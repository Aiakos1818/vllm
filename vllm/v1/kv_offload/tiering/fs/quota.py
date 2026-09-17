# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Byte budget for the filesystem KV offload tier.

A promotion copies a block from disk into the primary tier and leaves the file
in place, so a block that keeps being restored keeps being useful. Eviction is
therefore LRU over file recency: a successful load refreshes the file's mtime,
and a restart resumes with the ordering already recorded on disk.

Eviction runs on the tier's own I/O threads (never on the scheduler thread) and
unlinks whole files, oldest first, until the incoming batch fits. Lookups are
file-existence checks, so an evicted block is simply a miss.
"""

import contextlib
import os
import threading
import time
from collections.abc import Collection

from vllm.logger import init_logger

logger = init_logger(__name__)

_TMP_SUFFIX = ".tmp"


def _by_recency(entry: tuple[str, tuple[int, float]]) -> float:
    return entry[1][1]


class FileQuota:
    """LRU byte budget over a directory of KV block files.

    The inventory (path -> (size, recency)) is the source of truth for both the
    byte count and the eviction order. It is rebuilt from disk on construction,
    and its recency values come from ``os.utime`` on load, so a restarted
    process keeps evicting in the same order.

    All public methods are safe to call from multiple I/O threads.

    Args:
        root_dir: Directory holding this rank's block files.
        max_bytes: Upper bound on the bytes the tier may hold on disk.

    """

    def __init__(self, root_dir: str, max_bytes: int) -> None:
        self.root_dir = root_dir
        self.max_bytes = max_bytes
        self._lock = threading.Lock()
        self._entries: dict[str, tuple[int, float]] = {}
        self._used_bytes = 0
        # Bytes reserved by a store batch that has not committed yet.
        self._pending_bytes = 0
        # Observations since the last take_counters().
        self._evictions = 0
        self._evicted_bytes = 0
        self._skipped_bytes = 0
        self._scan()

    @property
    def used_bytes(self) -> int:
        return self._used_bytes

    @property
    def num_files(self) -> int:
        return len(self._entries)

    def reserve(
        self,
        paths: Collection[str],
        bytes_per_file: int,
        skip: Collection[str] = (),
    ) -> int | None:
        """Account for a store batch, evicting oldest-first to make room.

        Args:
            paths: Block files the batch would write.
            bytes_per_file: Size of one block file.
            skip: Paths that must not be evicted (in-flight loads).

        Returns:
            The bytes reserved, or None if the batch does not fit under
            ``max_bytes`` even after evicting everything evictable. The caller
            must not write the batch in that case.

        """
        with self._lock:
            # Blocks already on disk are skipped by the store path, so they
            # cost nothing and must not evict anything.
            n_bytes = sum(bytes_per_file for path in paths if path not in self._entries)
            if self._used_bytes + self._pending_bytes + n_bytes > self.max_bytes:
                self._evict_locked(
                    self._used_bytes + self._pending_bytes + n_bytes - self.max_bytes,
                    skip,
                )
            if self._used_bytes + self._pending_bytes + n_bytes > self.max_bytes:
                self._skipped_bytes += n_bytes
                return None
            self._pending_bytes += n_bytes
            return n_bytes

    def commit(self, paths: Collection[str], reserved_bytes: int) -> None:
        """Replace a reservation with the bytes actually on disk.

        Called after the write attempt, successful or not: blocks written
        before a mid-batch failure are readable, so they are counted.
        """
        with self._lock:
            self._pending_bytes -= reserved_bytes
            now = time.time()
            for path in paths:
                if path in self._entries:
                    continue
                try:
                    size = os.stat(path).st_size
                except OSError:
                    continue
                self._entries[path] = (size, now)
                self._used_bytes += size

    def evict_bytes(self, n_bytes: int, skip: Collection[str] = ()) -> int:
        """Evict oldest-first until at least *n_bytes* are reclaimed.

        For the retry path: a write can fail with a full-disk errno even though
        the budget says there is room, e.g. when other data (or another
        instance) shares the filesystem.

        Returns:
            The number of bytes actually reclaimed.

        """
        with self._lock:
            return self._evict_locked(n_bytes, skip)

    def touch(self, paths: Collection[str]) -> None:
        """Refresh the recency of blocks that were just restored."""
        now = time.time()
        with self._lock:
            for path in paths:
                entry = self._entries.get(path)
                if entry is None:
                    continue
                with contextlib.suppress(OSError):
                    os.utime(path, (now, now))
                    self._entries[path] = (entry[0], now)

    def take_counters(self) -> tuple[int, int, int]:
        """Return and reset (evictions, evicted bytes, skipped store bytes)."""
        with self._lock:
            counters = (self._evictions, self._evicted_bytes, self._skipped_bytes)
            self._evictions = self._evicted_bytes = self._skipped_bytes = 0
            return counters

    def _evict_locked(self, target_bytes: int, skip: Collection[str]) -> int:
        """Unlink oldest files until *target_bytes* are reclaimed."""
        if target_bytes <= 0:
            return 0
        freed = 0
        evicted = 0
        for path, (size, _recency) in sorted(self._entries.items(), key=_by_recency):
            if freed >= target_bytes:
                break
            if path in skip:
                continue
            try:
                os.remove(path)
            except FileNotFoundError:
                pass
            except OSError as exc:
                logger.warning("Failed to evict block file %s: %s", path, exc)
                continue
            del self._entries[path]
            self._used_bytes -= size
            freed += size
            evicted += 1
        self._evictions += evicted
        self._evicted_bytes += freed
        if freed:
            logger.info(
                "Evicted %d KV block(s) (%.2f GiB) from '%s'",
                evicted,
                freed / 2**30,
                self.root_dir,
            )
        return freed

    def _scan(self) -> None:
        """Rebuild the inventory from disk, dropping orphaned temp files."""
        entries: dict[str, tuple[int, float]] = {}
        used = 0
        for dirpath, _dirnames, filenames in os.walk(self.root_dir):
            for name in filenames:
                path = os.path.join(dirpath, name)
                if name.endswith(_TMP_SUFFIX):
                    # A temp file exists only while a store is in flight, and
                    # block files are renamed into place atomically, so
                    # anything left here is an orphan from a killed process.
                    with contextlib.suppress(OSError):
                        os.remove(path)
                    continue
                try:
                    stat = os.stat(path)
                except OSError:
                    continue
                entries[path] = (stat.st_size, stat.st_mtime)
                used += stat.st_size
        self._entries = entries
        self._used_bytes = used
        self._pending_bytes = 0
