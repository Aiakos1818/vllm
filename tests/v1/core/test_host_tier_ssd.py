# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the two-tier GPU/SSD host tier (``HostTierSSDStore``).

Covers the session index matching, quota LRU, slot/file lifetime and failure
handling without spinning up an engine.
"""

import mmap
import os
import threading
import time

import pytest

from vllm.v1.core.host_tier_ssd import HostTierSSDStore

pytestmark = pytest.mark.cpu_test

ROW = 4096


def make_view(num_blocks: int, row_bytes: int = ROW):
    buf = mmap.mmap(-1, num_blocks * row_bytes)
    flat = memoryview(buf)
    view = flat.cast("B", shape=(num_blocks, row_bytes))
    return buf, view, flat


def close_view(buf, view, flat) -> None:
    view.release()
    flat.release()
    buf.close()


def make_store(tmp_path, view, quota_bytes: int = 1 << 20) -> HostTierSSDStore:
    return HostTierSSDStore(
        root_dir=str(tmp_path),
        quota_bytes=quota_bytes,
        kv_view=view,
        engine_id="test",
        read_threads=2,
        write_threads=2,
        use_o_direct=False,
        row_bytes=ROW,
    )


def wait_n(store: HostTierSSDStore, n: int, timeout: float = 10.0):
    out = []
    deadline = time.monotonic() + timeout
    while len(out) < n and time.monotonic() < deadline:
        out.extend(store.poll())
        time.sleep(0.005)
    assert len(out) >= n, f"timed out waiting for {n} SSD jobs"
    return out


def row_bytes_at(flat, slot: int, row: int = ROW) -> bytes:
    return bytes(flat[slot * row : (slot + 1) * row])


def tail(byte: int) -> bytes:
    return bytes([byte]) * 16


def test_store_find_load_roundtrip(tmp_path) -> None:
    buf, view, flat = make_view(16)
    store = make_store(tmp_path, view)
    originals = {}
    for slot in (0, 1, 2):
        data = bytes([(slot + 1) * 7 % 256]) * ROW
        flat[slot * ROW : (slot + 1) * ROW] = data
        originals[slot] = data

    grp_hashes = [[b"\x01" * 16, b"\x02" * 16], [b"\x03" * 16]]
    job = store.submit_store("sid1", [0, 1, 2], grp_hashes, tail(0xAA), 96)
    assert job is not None
    [res] = wait_n(store, 1)
    assert res.success and res.kind == "store"
    assert store.num_sessions == 1
    assert store.bytes_used == 3 * ROW

    sess = store.find([tail(0xAA)])
    assert sess is not None
    assert sess["n_slots"] == 3
    assert sess["grp_hashes"] == grp_hashes

    # Taking it for restore removes it from the lookup index (a resumed
    # request must not match the same parked session again while loading).
    taken = store.take_for_restore("sid1")
    assert taken is not None
    assert store.find([tail(0xAA)]) is None
    assert store.num_sessions == 0

    for slot in (5, 6, 7):
        flat[slot * ROW : (slot + 1) * ROW] = b"\x00" * ROW
    job = store.submit_load("sid1", [5, 6, 7])
    assert job is not None
    [res] = wait_n(store, 1)
    assert res.success and res.kind == "load"
    assert row_bytes_at(flat, 5) == originals[0]
    assert row_bytes_at(flat, 6) == originals[1]
    assert row_bytes_at(flat, 7) == originals[2]
    # Files are deleted once the load completed.
    assert not os.path.exists(taken["dir"])

    store.shutdown()
    close_view(buf, view, flat)


def test_find_matches_tail_and_prefix_longest(tmp_path) -> None:
    buf, view, flat = make_view(8)
    store = make_store(tmp_path, view)

    short = [[b"\x10" * 16, b"\x11" * 16]]
    long = [[b"\x10" * 16, b"\x11" * 16, b"\x12" * 16]]
    store.submit_store("short", [0, 1], short, b"\x11" * 16, 32)
    store.submit_store("long", [2, 3, 4], long, b"\x12" * 16, 48)
    wait_n(store, 2)

    # Tail-hash hit selects the longer chain when both match.
    sess = store.find([b"\x12" * 16])
    assert sess is not None and sess["sid"] == "long"
    # Prefix match works without the tail.
    sess = store.find([b"\x10" * 16, b"\x11" * 16])
    assert sess is not None and sess["sid"] == "short"
    # No match.
    assert store.find([b"\x99" * 16]) is None

    store.shutdown()
    close_view(buf, view, flat)


def test_quota_lru_eviction(tmp_path) -> None:
    buf, view, flat = make_view(16)
    store = make_store(tmp_path, view, quota_bytes=2 * 2 * ROW)
    nbytes = 2 * ROW
    for i, sid in enumerate(("a", "b", "c")):
        store.evict_for(nbytes)
        assert (
            store.submit_store(
                sid, [i, i + 1], [[bytes([i]) * 16]], tail(i), 32
            )
            is not None
        )
        wait_n(store, 1)

    assert store.num_sessions == 2
    assert store.find([tail(0)]) is None  # LRU evicted
    assert store.find([tail(1)]) is not None
    assert store.find([tail(2)]) is not None
    assert not os.path.exists(store.session_dir("a"))
    assert store.bytes_used == 2 * nbytes

    store.shutdown()
    close_view(buf, view, flat)


def test_evict_prefers_small_then_oldest(tmp_path) -> None:
    buf, view, flat = make_view(16)
    store = make_store(tmp_path, view, quota_bytes=2 * 2 * ROW)
    nbytes = 2 * ROW
    # "big_old" is parked first (oldest); "small_new" is parked after it.
    store.evict_for(nbytes)
    assert (
        store.submit_store(
            "big_old", [0, 1], [[b"\x01" * 16]], tail(1), 80_000
        )
        is not None
    )
    wait_n(store, 1)
    store.evict_for(nbytes)
    assert (
        store.submit_store(
            "small_new", [2, 3], [[b"\x02" * 16]], tail(2), 1_000
        )
        is not None
    )
    wait_n(store, 1)
    assert store.num_sessions == 2

    # Making room must drop the small session first despite it being newer.
    store.evict_for(nbytes)
    assert store.find([tail(1)]) is not None  # big_old survives
    assert store.find([tail(2)]) is None  # small_new evicted first
    assert os.path.exists(store.session_dir("big_old"))
    assert not os.path.exists(store.session_dir("small_new"))

    store.shutdown()
    close_view(buf, view, flat)


def test_touch_makes_session_most_recent(tmp_path) -> None:
    buf, view, flat = make_view(16)
    store = make_store(tmp_path, view, quota_bytes=2 * 2 * ROW)
    nbytes = 2 * ROW
    for i, sid in enumerate(("a", "b")):
        store.evict_for(nbytes)
        assert (
            store.submit_store(
                sid, [i, i + 1], [[bytes([i]) * 16]], tail(i), 80_000
            )
            is not None
        )
        wait_n(store, 1)

    # "a" was parked first but is touched afterwards, so "b" is the LRU victim.
    store.touch("a")
    store.evict_for(nbytes)
    assert store.find([tail(0)]) is not None
    assert store.find([tail(1)]) is None

    store.shutdown()
    close_view(buf, view, flat)


def test_store_failure_cleans_up(tmp_path, monkeypatch) -> None:
    buf, view, flat = make_view(8)
    store = make_store(tmp_path, view)

    def boom(*args, **kwargs):
        raise RuntimeError("disk exploded")

    monkeypatch.setattr("vllm.v1.core.host_tier_ssd.batch_store_block", boom)
    store.submit_store("bad", [0, 1], [[b"\x01" * 16]], tail(1), 16)
    [res] = wait_n(store, 1)
    assert not res.success
    assert store.num_sessions == 0
    assert store.find([tail(1)]) is None
    assert not os.path.exists(store.session_dir("bad"))

    store.shutdown()
    close_view(buf, view, flat)


def test_inflight_load_not_evicted(tmp_path, monkeypatch) -> None:
    buf, view, flat = make_view(16)
    store = make_store(tmp_path, view, quota_bytes=4 * ROW)
    # Store "victim" first so it is the LRU entry, then "target".
    store.submit_store("victim", [0, 1], [[b"\x01" * 16]], tail(1), 16)
    wait_n(store, 1)
    store.submit_store("target", [2, 3], [[b"\x02" * 16]], tail(2), 16)
    wait_n(store, 1)

    release = threading.Event()

    def slow_load(paths, offsets, nbytes_):
        release.wait(5)

    monkeypatch.setattr(store, "_load_task", slow_load)
    assert store.take_for_restore("target") is not None
    assert store.submit_load("target", [4, 5]) is not None
    assert "target" in store._inflight_loads

    # Forcing room must evict the LRU "victim"; the in-flight "target" is out
    # of the index (restoring) and its files are protected.
    store.evict_for(3 * ROW)
    assert "target" in store._restoring
    assert store.find([tail(2)]) is None
    assert store.find([tail(1)]) is None

    release.set()
    [res] = wait_n(store, 1)
    assert res.success
    store.shutdown()
    close_view(buf, view, flat)


def test_session_larger_than_quota_rejected(tmp_path) -> None:
    buf, view, flat = make_view(4)
    store = make_store(tmp_path, view, quota_bytes=ROW // 2)
    assert store.submit_store("big", [0], [[b"\x01" * 16]], tail(1), 8) is None
    assert store.num_sessions == 0
    store.shutdown()
    close_view(buf, view, flat)


def test_chunked_store_roundtrip(tmp_path) -> None:
    """begin/append/commit then range-load/finish_restore across chunks."""
    buf, view, flat = make_view(16)
    store = make_store(tmp_path, view)
    for slot in range(6):
        flat[slot * ROW : (slot + 1) * ROW] = bytes([slot + 1]) * ROW

    grp_hashes = [[b"\x01" * 16, b"\x02" * 16], [b"\x03" * 16]]
    assert store.begin_store("chunked", 6, grp_hashes, tail(0xAB), 96)
    # Not discoverable until the commit chunk lands.
    assert store.find([tail(0xAB)]) is None

    assert store.append_store("chunked", 0, [0, 1, 2], commit=False) is not None
    [r1] = wait_n(store, 1)
    assert r1.success and not r1.commit and r1.start_idx == 0
    assert store.find([tail(0xAB)]) is None

    assert store.append_store("chunked", 3, [3, 4, 5], commit=True) is not None
    [r2] = wait_n(store, 1)
    assert r2.success and r2.commit and r2.start_idx == 3
    sess = store.find([tail(0xAB)])
    assert sess is not None and sess["n_slots"] == 6
    assert os.path.exists(os.path.join(store.session_dir("chunked"), ".commit"))
    assert store.bytes_used == 6 * ROW

    taken = store.take_for_restore("chunked")
    assert taken is not None
    for slot in range(10, 16):
        flat[slot * ROW : (slot + 1) * ROW] = b"\x00" * ROW

    assert (
        store.submit_load_range("chunked", 0, [10, 11, 12], keep_files=True)
        is not None
    )
    [rl1] = wait_n(store, 1)
    assert rl1.success
    # Chunked restores keep the files/session until finish_restore.
    assert "chunked" in store._restoring
    assert os.path.exists(taken["dir"])

    assert (
        store.submit_load_range("chunked", 3, [13, 14, 15], keep_files=True)
        is not None
    )
    [rl2] = wait_n(store, 1)
    assert rl2.success
    for i, slot in enumerate(range(10, 16)):
        assert row_bytes_at(flat, slot) == bytes([i + 1]) * ROW

    store.finish_restore("chunked")
    assert not os.path.exists(taken["dir"])
    assert "chunked" not in store._restoring

    store.shutdown()
    close_view(buf, view, flat)


def test_chunked_store_abort_releases_quota(tmp_path) -> None:
    buf, view, flat = make_view(8)
    store = make_store(tmp_path, view, quota_bytes=4 * ROW)
    assert store.begin_store("bad", 4, [[b"\x05" * 16]], tail(5), 64)
    # The whole session is reserved up front, so a second one cannot start.
    assert not store.begin_store("other", 1, [[b"\x06" * 16]], tail(6), 16)
    assert store.append_store("bad", 0, [0, 1], commit=False) is not None
    [r] = wait_n(store, 1)
    assert r.success

    store.abort_store("bad")
    assert not os.path.exists(store.session_dir("bad"))
    assert store.find([tail(5)]) is None
    # Reservation released -> another session fits.
    assert store.begin_store("other", 1, [[b"\x06" * 16]], tail(6), 16)
    store.abort_store("other")

    store.shutdown()
    close_view(buf, view, flat)


def test_load_range_bounds_checked(tmp_path) -> None:
    buf, view, flat = make_view(8)
    store = make_store(tmp_path, view)
    store.begin_store("s", 2, [[b"\x07" * 16]], tail(7), 32)
    store.append_store("s", 0, [0, 1], commit=True)
    wait_n(store, 1)
    assert store.take_for_restore("s") is not None
    assert store.submit_load_range("s", 1, [2, 3], keep_files=True) is None
    assert store.submit_load_range("s", -1, [2], keep_files=True) is None
    store.discard("s")
    store.shutdown()
    close_view(buf, view, flat)


def test_begin_store_records_anchors_and_snapshot(tmp_path) -> None:
    """The parked session keeps its anchor count for the monitor snapshot."""
    buf, view, flat = make_view(8)
    store = make_store(tmp_path, view)
    assert store.begin_store("sidA", 3, [[b"\x01" * 16]], tail(0xAA), 96, 2)
    assert store.append_store("sidA", 0, [0, 1, 2], commit=True) is not None
    wait_n(store, 1)

    [sess] = store.snapshot()
    assert sess["sid"] == "sidA"
    assert sess["tokens"] == 96
    assert sess["blocks"] == 3
    assert sess["bytes"] == 3 * ROW
    assert sess["anchors"] == 2
    assert "last_used" in sess

    store.shutdown()
    close_view(buf, view, flat)
