# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Session-level SSD parking for the host-tier offload (two-tier GPU/SSD).

Sessions evicted from GPU under admission pressure are staged through the
connector's CPU buffer and written to a session-scoped directory on a
filesystem (a real SSD in production, tmpfs for tests). On resume the session
is read back into a staging buffer and copied to fresh GPU blocks.

The CPU tier is a transient bounce buffer only: parked sessions live on disk.
The in-process index is keyed by the parked chain's tail hash / per-group
hashes, mirroring the RAM parking lookup (``KVCacheManager.find_ram_session``).
"""

import functools
import hashlib
import os
import re
import shutil
import threading
import time
from dataclasses import dataclass
from typing import Any

from vllm import envs
from vllm.logger import init_logger
from vllm.v1.kv_offload.tiering.fs.io import (
    batch_load_block,
    batch_store_block,
    probe_o_direct,
)
from vllm.v1.kv_offload.tiering.fs.thread_pool import DualQueueThreadPool


def evict_sort_key(tokens: int, last_used: float) -> tuple[int, float]:
    """Host-tier eviction order: small sessions first, LRU within a tier.

    ``VLLM_HOSTTIER_EVICT_SMALL_TOKENS`` splits sessions into a small tier
    (evicted first) and a large tier; both tiers are ordered least-recently-used
    first. A non-positive threshold collapses the two tiers into one (pure LRU).
    """
    small = envs.VLLM_HOSTTIER_EVICT_SMALL_TOKENS
    return (1 if 0 < small <= tokens else 0, last_used)

logger = init_logger(__name__)

_UNSAFE = re.compile(r"[^A-Za-z0-9_.-]")


def _safe_name(name: str) -> str:
    safe = _UNSAFE.sub("_", name)[:48]
    digest = hashlib.sha1(name.encode("utf-8")).hexdigest()[:12]
    return f"{safe}_{digest}"


class _RateLimiter:
    """Aggregate byte-rate limiter shared by all I/O threads."""

    def __init__(self, max_mbps: float) -> None:
        self._interval_per_byte = (
            1.0 / (max_mbps * (1 << 20)) if max_mbps > 0 else 0.0
        )
        self._lock = threading.Lock()
        self._next = time.monotonic()

    def acquire(self, nbytes: int) -> None:
        if self._interval_per_byte <= 0.0:
            return
        with self._lock:
            now = time.monotonic()
            start = max(now, self._next)
            self._next = start + nbytes * self._interval_per_byte
            wait = start - now
        if wait > 0:
            time.sleep(wait)


@dataclass
class SSDJobResult:
    job_id: int
    kind: str  # "store" | "load"
    sid: str
    slots: list[int]
    nbytes: int
    success: bool
    session: dict[str, Any] | None = None
    start_idx: int = 0
    commit: bool = False


class HostTierSSDStore:
    """Parked-session store backed by files on disk (SSD / tmpfs)."""

    def __init__(
        self,
        root_dir: str,
        quota_bytes: int,
        kv_view: memoryview,
        engine_id: str = "0",
        read_threads: int = 8,
        write_threads: int = 8,
        max_mbps: float = 0.0,
        clean_start: bool = True,
        row_bytes: int | None = None,
        use_o_direct: bool | None = None,
        region: Any | None = None,
    ) -> None:
        self._kv_view = kv_view
        if row_bytes is None:
            assert kv_view.strides is not None, "kv_view must be 2-D"
            row_bytes = int(kv_view.strides[0])
        self._row_bytes = int(row_bytes)
        self._quota = int(quota_bytes)
        self._region = region

        self._engine_dir = os.path.join(root_dir, _safe_name(engine_id))
        self._sessions_dir = os.path.join(self._engine_dir, "sessions")
        if clean_start:
            shutil.rmtree(self._sessions_dir, ignore_errors=True)
        os.makedirs(self._sessions_dir, exist_ok=True)
        self._use_o_direct = (
            probe_o_direct(self._sessions_dir)
            if use_o_direct is None
            else bool(use_o_direct)
        )
        logger.info(
            "HostTierSSDStore: root=%s quota=%.2fGiB O_DIRECT=%s",
            self._sessions_dir,
            self._quota / (1 << 30),
            self._use_o_direct,
        )

        self._pool = DualQueueThreadPool(
            read_threads,
            write_threads,
            thread_name_prefix="vllm_host_tier_ssd",
        )
        self._limiter = _RateLimiter(max_mbps)

        self._lock = threading.Lock()
        # Committed sessions: sid -> metadata (incl. grp_hashes / tail / dir).
        self._sessions: dict[str, dict[str, Any]] = {}
        # Sessions being restored: taken out of the index so a resumed request
        # is not matched again while its load is in flight; files are deleted
        # once the load completes.
        self._restoring: dict[str, dict[str, Any]] = {}
        # In-flight jobs: job_id -> (kind, sid, slots, nbytes, job_meta).
        self._jobs: dict[int, tuple[str, str, list[int], int, dict | None]] = {}
        self._next_job_id = 0
        self._pending_bytes = 0
        # Whole-session reservations for multi-chunk stores (held from
        # ``begin_store`` until the commit chunk completes).
        self._reserved_bytes = 0
        self._inflight_meta: dict[str, dict[str, Any]] = {}
        self._bytes_used = 0
        self._inflight_stores: set[str] = set()
        self._inflight_loads: set[str] = set()

    # ------------------------------------------------------------------ #
    # Introspection.
    # ------------------------------------------------------------------ #
    @property
    def row_bytes(self) -> int:
        return self._row_bytes

    @property
    def quota_bytes(self) -> int:
        return self._quota

    @property
    def num_sessions(self) -> int:
        with self._lock:
            return len(self._sessions)

    @property
    def bytes_used(self) -> int:
        with self._lock:
            return self._bytes_used

    def session_dir(self, sid: str) -> str:
        return os.path.join(self._sessions_dir, _safe_name(sid))

    # ------------------------------------------------------------------ #
    # Lookup / eviction.
    # ------------------------------------------------------------------ #
    def find(self, block_hashes: list[bytes]) -> dict[str, Any] | None:
        """Parked session that this request resumes/extends.

        Same matching rule as the RAM parking path: tail-hash signal first,
        then a strict per-group hash-prefix match; the longest chain wins.
        """
        if not block_hashes:
            return None
        bh_set = set(block_hashes)
        with self._lock:
            sessions = list(self._sessions.values())
        best: dict[str, Any] | None = None
        best_len = 0
        for sess in sessions:
            n = max((len(h) for h in sess["grp_hashes"]), default=0)
            if n <= best_len:
                continue
            hit = sess["tail"] is not None and sess["tail"] in bh_set
            if not hit:
                for hashes in sess["grp_hashes"]:
                    if (
                        hashes
                        and len(block_hashes) >= len(hashes)
                        and list(block_hashes[: len(hashes)]) == list(hashes)
                    ):
                        hit = True
                        break
            if hit:
                best = sess
                best_len = n
        return best

    def touch(self, sid: str) -> None:
        with self._lock:
            sess = self._sessions.get(sid)
            if sess is not None:
                sess["last_used"] = time.monotonic()

    def take_for_restore(self, sid: str) -> dict[str, Any] | None:
        """Remove a session from the lookup index ahead of a restore load.

        The files stay on disk until the load completes; the session is not
        matchable by ``find`` while the restore is in flight.
        """
        with self._lock:
            sess = self._sessions.pop(sid, None)
            if sess is None:
                return None
            self._bytes_used -= sess["bytes"]
            self._restoring[sid] = sess
            return dict(sess)

    def discard(self, sid: str) -> None:
        """Drop a restoring session and its files (aborted/failed restore)."""
        with self._lock:
            meta = self._restoring.pop(sid, None)
        if meta is not None:
            shutil.rmtree(meta["dir"], ignore_errors=True)

    def remove(self, sid: str) -> None:
        """Drop a session and delete its files (e.g. unreadable data)."""
        with self._lock:
            sess = self._sessions.pop(sid, None)
            if sess is not None:
                self._bytes_used -= sess["bytes"]
            directory = self.session_dir(sid)
        shutil.rmtree(directory, ignore_errors=True)

    def evict_for(self, need_bytes: int) -> int:
        """Two-tier evict committed sessions until ``need_bytes`` fits quota.

        Small sessions (< ``VLLM_HOSTTIER_EVICT_SMALL_TOKENS``) go first, then
        large ones; within each tier the oldest (``last_used``) is dropped
        first. In-flight stores/loads are never evicted. Returns the number of
        sessions dropped.
        """
        victims: list[dict[str, Any]] = []
        with self._lock:
            while (
                self._bytes_used
                + self._reserved_bytes
                + self._pending_bytes
                + need_bytes
                > self._quota
            ):
                candidates = [
                    s
                    for s in self._sessions.values()
                    if s["sid"] not in self._inflight_stores
                    and s["sid"] not in self._inflight_loads
                ]
                if not candidates:
                    break
                victim = min(
                    candidates,
                    key=lambda s: evict_sort_key(s["tokens"], s["last_used"]),
                )
                self._sessions.pop(victim["sid"], None)
                self._bytes_used -= victim["bytes"]
                victims.append(victim)
        for victim in victims:
            shutil.rmtree(victim["dir"], ignore_errors=True)
            logger.info(
                "HostTierSSDStore: evicted session %s (%d tokens, %d bytes)",
                victim["sid"],
                victim["tokens"],
                victim["bytes"],
            )
        return len(victims)

    # ------------------------------------------------------------------ #
    # Async transfer submission.
    # ------------------------------------------------------------------ #
    def begin_store(
        self,
        sid: str,
        total_slots: int,
        grp_hashes: list[list[bytes]],
        tail: bytes | None,
        tokens: int,
    ) -> bool:
        """Reserve quota for a session that will be written in chunks.

        Returns False when the session can never fit the quota or no space
        could be freed (caller runs ``evict_for`` first).
        """
        total_bytes = total_slots * self._row_bytes
        if total_slots <= 0 or total_bytes > self._quota:
            logger.warning(
                "HostTierSSDStore: session %s needs %d bytes > quota %d",
                sid,
                total_bytes,
                self._quota,
            )
            return False
        with self._lock:
            if (
                self._bytes_used
                + self._reserved_bytes
                + self._pending_bytes
                + total_bytes
                > self._quota
            ):
                return False
            # Re-parking the same session: drop the previous copy so files are
            # rewritten from scratch (a stale prefix file must never be reused).
            old = self._sessions.pop(sid, None)
            if old is not None:
                self._bytes_used -= old["bytes"]
        if old is not None:
            shutil.rmtree(old["dir"], ignore_errors=True)

        directory = self.session_dir(sid)
        os.makedirs(directory, exist_ok=True)
        meta = {
            "sid": sid,
            "req_id": sid,
            "tail": tail,
            "grp_hashes": [list(h) for h in grp_hashes],
            "tokens": tokens,
            "n_slots": total_slots,
            "bytes": total_bytes,
            "dir": directory,
            "last_used": time.monotonic(),
        }
        with self._lock:
            self._reserved_bytes += total_bytes
            self._inflight_stores.add(sid)
            self._inflight_meta[sid] = meta
        return True

    def append_store(
        self,
        sid: str,
        start_idx: int,
        slots: list[int],
        commit: bool,
    ) -> int | None:
        """Write one chunk of staging slots to files ``start_idx..``.

        ``commit`` marks the final chunk; only then does the session become
        discoverable via :meth:`find`.
        """
        with self._lock:
            meta = self._inflight_meta.get(sid)
        if meta is None or not slots or start_idx < 0:
            return None
        n = len(slots)
        paths = [
            os.path.join(meta["dir"], f"{start_idx + i:05d}.bin") for i in range(n)
        ]
        offsets = [s * self._row_bytes for s in slots]
        nbytes = n * self._row_bytes

        job_id = self._new_job_id()
        job_meta = {"meta": meta, "start_idx": start_idx, "commit": bool(commit)}
        with self._lock:
            self._jobs[job_id] = ("store", sid, list(slots), nbytes, job_meta)
            self._pending_bytes += nbytes
        task = functools.partial(self._store_task, paths, offsets, nbytes)
        self._pool.enqueue_store(job_id, 1, [task])
        return job_id

    def abort_store(self, sid: str) -> None:
        """Drop a partial/failed multi-chunk store and release its quota."""
        with self._lock:
            meta = self._inflight_meta.pop(sid, None)
            if meta is not None:
                self._reserved_bytes -= meta["bytes"]
            self._inflight_stores.discard(sid)
        if meta is not None:
            shutil.rmtree(meta["dir"], ignore_errors=True)

    def submit_store(
        self,
        sid: str,
        slots: list[int],
        grp_hashes: list[list[bytes]],
        tail: bytes | None,
        tokens: int,
    ) -> int | None:
        """Whole-session store (single chunk) used by tests/small sessions."""
        if not self.begin_store(sid, len(slots), grp_hashes, tail, tokens):
            return None
        job_id = self.append_store(sid, 0, slots, commit=True)
        if job_id is None:
            self.abort_store(sid)
        return job_id

    def submit_load_range(
        self,
        sid: str,
        start_idx: int,
        slots: list[int],
        keep_files: bool = False,
    ) -> int | None:
        """Read files ``start_idx..`` of a restoring session into ``slots``."""
        with self._lock:
            sess = self._restoring.get(sid)
            if sess is None:
                return None
            directory = sess["dir"]
            session_copy = dict(sess)
        n = len(slots)
        if not slots or start_idx < 0 or start_idx + n > session_copy["n_slots"]:
            return None
        with self._lock:
            self._inflight_loads.add(sid)

        paths = [
            os.path.join(directory, f"{start_idx + i:05d}.bin") for i in range(n)
        ]
        offsets = [s * self._row_bytes for s in slots]
        nbytes = n * self._row_bytes

        job_id = self._new_job_id()
        job_meta = {
            "session": session_copy,
            "start_idx": start_idx,
            "keep_files": bool(keep_files),
        }
        with self._lock:
            self._jobs[job_id] = ("load", sid, list(slots), nbytes, job_meta)
        task = functools.partial(self._load_task, paths, offsets, nbytes)
        self._pool.enqueue_load(job_id, 1, [task])
        return job_id

    def submit_load(self, sid: str, slots: list[int]) -> int | None:
        """Whole-session load; deletes the files when it completes."""
        with self._lock:
            sess = self._restoring.get(sid)
            if sess is None or len(slots) != sess["n_slots"]:
                return None
        return self.submit_load_range(sid, 0, slots, keep_files=False)

    def finish_restore(self, sid: str) -> None:
        """Drop a restoring session and its files after the final chunk."""
        with self._lock:
            meta = self._restoring.pop(sid, None)
            self._inflight_loads.discard(sid)
        if meta is not None:
            shutil.rmtree(meta["dir"], ignore_errors=True)

    @staticmethod
    def _write_commit_marker(meta: dict[str, Any]) -> None:
        path = os.path.join(meta["dir"], ".commit")
        try:
            with open(path, "w") as f:
                f.write(
                    f"{meta['sid']} {meta['n_slots']} {meta['bytes']} "
                    f"{meta['tokens']}\n"
                )
        except OSError:
            logger.warning(
                "HostTierSSDStore: commit marker write failed for %s",
                meta["sid"],
                exc_info=True,
            )

    def _store_task(self, paths: list[str], offsets: list[int], nbytes: int) -> None:
        self._limiter.acquire(nbytes)
        batch_store_block(
            paths, self._kv_view, offsets, self._row_bytes, self._use_o_direct
        )

    def _load_task(self, paths: list[str], offsets: list[int], nbytes: int) -> None:
        self._limiter.acquire(nbytes)
        batch_load_block(
            paths, self._kv_view, offsets, self._row_bytes, self._use_o_direct
        )

    # ------------------------------------------------------------------ #
    # Completion polling.
    # ------------------------------------------------------------------ #
    def poll(self) -> list[SSDJobResult]:
        results: list[SSDJobResult] = []
        for job_id, success in self._pool.get_finished():
            with self._lock:
                job = self._jobs.pop(job_id, None)
            if job is None:
                continue
            kind, sid, slots, nbytes, job_meta = job
            if kind == "store":
                meta = (job_meta or {}).get("meta")
                start_idx = int((job_meta or {}).get("start_idx", 0))
                commit = bool((job_meta or {}).get("commit", False))
                session = None
                with self._lock:
                    self._pending_bytes -= nbytes
                    if success and commit and meta is not None:
                        session = dict(meta)
                        session["last_used"] = time.monotonic()
                        self._sessions[sid] = session
                        self._bytes_used += meta["bytes"]
                        self._reserved_bytes -= meta["bytes"]
                        self._inflight_stores.discard(sid)
                        self._inflight_meta.pop(sid, None)
                if success and commit and meta is not None:
                    self._write_commit_marker(meta)
                if not success:
                    self.abort_store(sid)
                    logger.warning(
                        "HostTierSSDStore: store failed for session %s", sid
                    )
                results.append(
                    SSDJobResult(
                        job_id, kind, sid, slots, nbytes, success, session,
                        start_idx, commit,
                    )
                )
            else:
                keep_files = bool((job_meta or {}).get("keep_files", False))
                start_idx = int((job_meta or {}).get("start_idx", 0))
                session = None
                with self._lock:
                    self._inflight_loads.discard(sid)
                    meta = None
                    if not keep_files:
                        meta = self._restoring.pop(sid, None)
                    if success and meta is not None:
                        session = dict(meta)
                if meta is not None:
                    # The staging copy now holds the data; the parked files
                    # have served their purpose.
                    shutil.rmtree(meta["dir"], ignore_errors=True)
                if not success:
                    logger.warning(
                        "HostTierSSDStore: load failed for session %s", sid
                    )
                results.append(
                    SSDJobResult(
                        job_id, kind, sid, slots, nbytes, success, session,
                        start_idx, False,
                    )
                )
        return results

    def has_pending_work(self) -> bool:
        with self._lock:
            return bool(self._jobs)

    def _new_job_id(self) -> int:
        with self._lock:
            job_id = self._next_job_id
            self._next_job_id += 1
            return job_id

    def shutdown(self) -> None:
        self._pool.wait_idle()
        self._pool.shutdown(wait=True)
        if self._region is not None:
            try:
                self._kv_view.release()
            except Exception:
                pass
            try:
                self._region.cleanup()
            except Exception:
                logger.warning("HostTierSSDStore: region cleanup failed", exc_info=True)
            self._region = None
