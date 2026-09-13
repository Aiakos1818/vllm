# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the session-level host-tier (RAM) spill/restore helpers.

These cover the pure bookkeeping in ``KVCacheManager`` (slot allocator,
eviction order, session matching) and the scheduler's restore-slot lifetime
contract without spinning up an engine.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.sched import scheduler as scheduler_module
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)

pytestmark = pytest.mark.cpu_test


def make_manager(num_blocks: int = 32, block_size: int = 4) -> KVCacheManager:
    spec = FullAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
    )
    config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(["layer"], spec)],
    )
    return KVCacheManager(
        config,
        max_model_len=block_size * num_blocks,
        scheduler_block_size=block_size,
        hash_block_size=block_size,
    )


def test_alloc_and_free_ram_slots() -> None:
    manager = make_manager()
    manager.set_ram_capacity(8)
    assert manager.ram_slots_free() == 8

    a = manager.alloc_ram_slots(3)
    assert a == [0, 1, 2]
    b = manager.alloc_ram_slots(2)
    assert b == [3, 4]
    assert manager.ram_slots_free() == 3

    # Freeing coalesces back so a later large request can reuse the range.
    manager.free_ram_slots(a)
    assert manager.ram_slots_free() == 6
    manager.free_ram_slots(b)
    assert manager.ram_slots_free() == 8
    assert manager.alloc_ram_slots(8) == list(range(8))


def test_alloc_ram_slots_insufficient_returns_none() -> None:
    manager = make_manager()
    manager.set_ram_capacity(4)
    manager.alloc_ram_slots(3)
    assert manager.alloc_ram_slots(2) is None
    assert manager.alloc_ram_slots(0) is None


def test_evict_ram_for_two_tier_then_oldest() -> None:
    manager = make_manager()
    manager.set_ram_capacity(12)
    big_old = manager.alloc_ram_slots(4)
    big_new = manager.alloc_ram_slots(4)
    small_old = manager.alloc_ram_slots(2)
    small_new = manager.alloc_ram_slots(2)
    manager._ram_sessions["big_old"] = {
        "req_id": "big_old",
        "tokens": 80_000,
        "slots": big_old,
        "parked_at": 1.0,
        "last_used": 1.0,
    }
    manager._ram_sessions["big_new"] = {
        "req_id": "big_new",
        "tokens": 90_000,
        "slots": big_new,
        "parked_at": 2.0,
        "last_used": 2.0,
    }
    manager._ram_sessions["small_old"] = {
        "req_id": "small_old",
        "tokens": 1_000,
        "slots": small_old,
        "parked_at": 3.0,
        "last_used": 3.0,
    }
    manager._ram_sessions["small_new"] = {
        "req_id": "small_new",
        "tokens": 2_000,
        "slots": small_new,
        "parked_at": 4.0,
        "last_used": 4.0,
    }
    assert manager.ram_slots_free() == 0

    # Small tier drains first (LRU-first), then the large tier.
    manager.evict_ram_for(1)
    assert set(manager._ram_sessions) == {"big_old", "big_new", "small_new"}
    manager.evict_ram_for(3)
    assert set(manager._ram_sessions) == {"big_old", "big_new"}
    manager.evict_ram_for(5)
    assert set(manager._ram_sessions) == {"big_new"}

    # Protected session is never dropped, even when nothing else remains.
    manager.evict_ram_for(100, protect="big_new")
    assert "big_new" in manager._ram_sessions


def test_evict_ram_for_uses_last_used_not_park_time() -> None:
    """A session parked long ago but used recently outlives a newer park."""
    manager = make_manager()
    manager.set_ram_capacity(8)
    old_but_hot = manager.alloc_ram_slots(4)
    new_but_cold = manager.alloc_ram_slots(4)
    manager._ram_sessions["old_but_hot"] = {
        "req_id": "old_but_hot",
        "tokens": 80_000,
        "slots": old_but_hot,
        "parked_at": 1.0,
        "last_used": 9.0,
    }
    manager._ram_sessions["new_but_cold"] = {
        "req_id": "new_but_cold",
        "tokens": 90_000,
        "slots": new_but_cold,
        "parked_at": 2.0,
        "last_used": 2.0,
    }
    manager.evict_ram_for(4)
    assert set(manager._ram_sessions) == {"old_but_hot"}


def test_find_ram_session_refreshes_recency() -> None:
    manager = make_manager()
    manager._ram_sessions["s"] = {
        "req_id": "s",
        "tail": b"h2",
        "tokens": 20,
        "grp_hashes": [[b"h0", b"h1", b"h2"]],
        "slots": [0, 1, 2],
        "parked_at": 1.0,
        "last_used": 1.0,
    }
    assert manager.find_ram_session([b"h0", b"h1", b"h2"]) is not None
    assert manager._ram_sessions["s"]["last_used"] > 1.0


def test_find_ram_session_tail_and_prefix() -> None:
    manager = make_manager()
    manager._ram_sessions["s"] = {
        "req_id": "s",
        "tail": b"h2",
        "tokens": 20,
        "grp_hashes": [[b"h0", b"h1", b"h2"]],
        "slots": [0, 1, 2],
    }
    # Tail hash reappears in the resumed prompt.
    assert manager.find_ram_session([b"h0", b"h1", b"h2", b"h3"]) is not None
    # A strict prefix match also works.
    assert manager.find_ram_session([b"h0", b"h1", b"h2"]) is not None
    # No overlap -> miss.
    assert manager.find_ram_session([b"x0", b"x1"]) is None
    assert manager.find_ram_session([]) is None


def test_take_spill_candidates_two_tier_then_oldest() -> None:
    manager = make_manager(num_blocks=4)
    entries = (
        ("big_old", 80_000, 1.0),
        ("big_new", 90_000, 2.0),
        ("small_old", 1_000, 3.0),
        ("small_new", 2_000, 4.0),
    )
    for req_id, tokens, parked_at in entries:
        manager._auto_pin_entries.append(
            {
                "req_id": req_id,
                "tokens": tokens,
                "parked_at": parked_at,
                "last_used": parked_at,
                "blocks": [],
                "grp_blocks": [],
            }
        )
    # Ask for more free blocks than the pool can ever provide to force taking
    # every candidate; small tier first, then oldest within each tier.
    chosen = manager.take_spill_candidates(need_free_blocks=10_000)
    assert [e["req_id"] for e in chosen] == [
        "small_old",
        "small_new",
        "big_old",
        "big_new",
    ]
    assert not manager._auto_pin_entries
    assert set(manager._spill_hold) == {
        "small_old",
        "small_new",
        "big_old",
        "big_new",
    }


def test_confirm_spill_parks_metadata_without_blocks() -> None:
    from vllm.v1.core.kv_cache_utils import KVCacheBlock

    manager = make_manager()
    manager.set_ram_capacity(4)
    slots = manager.alloc_ram_slots(2)
    block = KVCacheBlock(block_id=7)
    block._block_hash = b"tailhash"
    block.is_null = False
    manager._spill_hold["r"] = {
        "req_id": "r",
        "tail": b"tailhash",
        "tokens": 8,
        "num_blocks": 2,
        "anchors": 3,
        "blocks": [],
        "grp_blocks": [[block]],
    }

    manager.confirm_spill("r", slots)

    session = manager.get_ram_sessions()["r"]
    assert session["slots"] == slots
    assert session["grp_hashes"] == [[b"tailhash"]]
    assert session["tokens"] == 8
    assert session["anchors"] == 3
    assert "r" not in manager._spill_hold


def test_matches_pinned_chain_refreshes_recency() -> None:
    """A resumed keep-alive chain becomes most-recently-used (true LRU)."""
    manager = make_manager()
    manager._auto_pin_entries.append(
        {
            "req_id": "s",
            "tail": b"t2",
            "tokens": 20,
            "blocks": [],
            "grp_blocks": [],
            "anchors": 1,
            "parked_at": 1.0,
            "last_used": 1.0,
        }
    )
    assert manager.matches_pinned_chain([b"t0", b"t1", b"t2"])
    assert manager._auto_pin_entries[0]["last_used"] > 1.0

    # A miss (or an empty prompt) leaves recency untouched.
    manager._auto_pin_entries[0]["last_used"] = 5.0
    assert not manager.matches_pinned_chain([b"x0"])
    assert not manager.matches_pinned_chain([])
    assert manager._auto_pin_entries[0]["last_used"] == 5.0


def test_host_tier_snapshots_report_anchors() -> None:
    manager = make_manager()
    manager._auto_pin_entries.append(
        {
            "req_id": "g",
            "tail": b"t",
            "tokens": 40,
            "blocks": [object(), object()],
            "grp_blocks": [],
            "anchors": 2,
            "parked_at": 1.0,
            "last_used": 3.0,
        }
    )
    manager._ram_sessions["r"] = {
        "req_id": "r",
        "tokens": 20,
        "num_blocks": 5,
        "slots": [0, 1],
        "anchors": 1,
        "parked_at": 2.0,
        "last_used": 4.0,
    }
    assert manager.pinned_chain_snapshot() == [
        {"id": "g", "tokens": 40, "blocks": 2, "anchors": 2, "last_used": 3.0}
    ]
    assert manager.ram_session_snapshot() == [
        {
            "id": "r",
            "tokens": 20,
            "blocks": 5,
            "slots": 2,
            "anchors": 1,
            "last_used": 4.0,
        }
    ]


def test_pinned_hit_blocks_are_not_evictable() -> None:
    """Keep-alive pinned hits must not inflate the admission gate."""
    from vllm.v1.core.kv_cache_utils import KVCacheBlock
    from vllm.v1.core.single_type_kv_cache_manager import SingleTypeKVCacheManager

    idle = KVCacheBlock(block_id=0)
    pinned = KVCacheBlock(block_id=1)
    pinned.pinned = 1
    null = KVCacheBlock(block_id=2)
    null.is_null = True
    in_use = KVCacheBlock(block_id=3)
    in_use.ref_cnt = 1

    assert (
        SingleTypeKVCacheManager._get_num_evictable_blocks(
            [idle, pinned, null, in_use]
        )
        == 1
    )


class _FakeConnector:
    def __init__(self, completed: list[int]) -> None:
        self._completed = completed

    def take_external_completed(self) -> list[int]:
        done = self._completed
        self._completed = []
        return done


class _FakeScheduler:
    """Minimal host exposing the attributes ``_drain_spill_completions`` reads."""

    def __init__(self, connector: _FakeConnector) -> None:
        self.connector = connector
        self.kv_cache_manager = make_manager()
        self.kv_cache_manager.set_ram_capacity(8)
        self._spill_job_to_req: dict[int, str] = {}
        self._spill_slots: dict[str, list[int]] = {}
        self._restore_job_to_req: dict[int, str] = {}
        self._restore_jobs: dict[str, dict] = {}
        self._restored_req_ids: set[str] = set()
        self.requests: dict[str, object] = {}
        self._host_tier_spills = 0
        self._host_tier_restores = 0
        self._ssd_store = None

    def _ramtrace(self, msg: str) -> None:
        pass

    def drain(self) -> None:
        Scheduler._drain_spill_completions(self)  # type: ignore[arg-type]


def test_restore_slots_held_until_load_completes() -> None:
    """P0: source host slots are only freed once the restore load is done."""
    scheduler = _FakeScheduler(_FakeConnector(completed=[0]))
    kvm = scheduler.kv_cache_manager
    slots = kvm.alloc_ram_slots(3)
    assert slots == [0, 1, 2]
    assert kvm.ram_slots_free() == 5

    scheduler._restore_jobs["r"] = {
        "job_id": 0,
        "session": {"req_id": "parked"},
        "per_group_blocks": [[]],
        "per_group_hashes": [[]],
        "slots": list(slots),
    }
    scheduler._restore_job_to_req[0] = "r"
    scheduler.requests["r"] = object()

    # In flight: slots must NOT be available for a concurrent spill.
    assert kvm.ram_slots_free() == 5

    scheduler.drain()

    assert kvm.ram_slots_free() == 8
    assert not scheduler._restore_jobs
    assert "r" in scheduler._restored_req_ids


def test_aborted_restore_does_not_leak_restored_req_id() -> None:
    scheduler = _FakeScheduler(_FakeConnector(completed=[1]))
    kvm = scheduler.kv_cache_manager
    slots = kvm.alloc_ram_slots(3)

    scheduler._restore_jobs["gone"] = {
        "job_id": 1,
        "session": {"req_id": "parked"},
        "per_group_blocks": [[]],
        "per_group_hashes": [[]],
        "slots": list(slots),
    }
    scheduler._restore_job_to_req[1] = "gone"
    # Request not in self.requests -> aborted while the load was in flight.

    scheduler.drain()

    assert kvm.ram_slots_free() == 8
    assert not scheduler._restore_jobs
    assert "gone" not in scheduler._restored_req_ids


def test_release_spill_blocks_partial_then_abort() -> None:
    """Chunked spills unpin a prefix; abort only releases the rest."""
    manager = make_manager(num_blocks=8)
    blocks = manager.block_pool.get_new_blocks(4)
    for b in blocks:
        manager.block_pool.pin_block(b)
    manager._spill_hold["r"] = {
        "req_id": "r",
        "tail": b"t",
        "tokens": 16,
        "num_blocks": 4,
        "blocks": list(blocks),
        "grp_blocks": [[blocks[0], blocks[1]], [blocks[2], blocks[3]]],
    }
    done = manager.release_spill_blocks("r", list(blocks[:2]))
    assert not done
    assert [b.pinned for b in blocks] == [0, 0, 1, 1]
    # Releasing the rest completes the hold.
    assert manager.release_spill_blocks("r", list(blocks[2:]))
    assert [b.pinned for b in blocks] == [0, 0, 0, 0]

    # Abort after a partial release must not double-unpin.
    manager._spill_hold["r2"] = {
        "req_id": "r2",
        "tail": b"t2",
        "tokens": 16,
        "num_blocks": 4,
        "blocks": list(blocks),
        "grp_blocks": [[blocks[0], blocks[1]], [blocks[2], blocks[3]]],
        "released_ids": {id(blocks[0]), id(blocks[1])},
    }
    for b in blocks[2:]:
        manager.block_pool.pin_block(b)
    manager.abort_spill("r2")
    assert [b.pinned for b in blocks] == [0, 0, 0, 0]
    assert "r2" not in manager._spill_hold
    manager.block_pool.free_blocks(blocks)


def test_hold_and_release_restored_blocks() -> None:
    """Chunked restores pin adopted blocks until the whole chain is loaded."""
    manager = make_manager(num_blocks=8)
    blocks = manager.allocate_restore_blocks([2])[0]
    free_before = manager.block_pool.get_num_free_blocks()
    manager.hold_restored_blocks([[b"h0", b"h1"]], [blocks])
    # Held blocks cannot be handed to another request.
    assert manager.block_pool.get_num_free_blocks() == free_before
    manager.release_restored_hold(blocks)
    assert manager.block_pool.get_num_free_blocks() == free_before + 2


class _InfoScheduler:
    """Minimal host for ``Scheduler.host_tier_info``."""

    host_tier_info = Scheduler.host_tier_info
    _ram_slot_bytes = Scheduler._ram_slot_bytes

    def __init__(self, manager: KVCacheManager, connector=None, store=None) -> None:
        self.kv_cache_manager = manager
        self._ssd_store = store
        self.connector = connector
        self._ssd_chunk_slots = 7
        self.block_size = 4
        self.kv_cache_config = object()
        self.vllm_config = SimpleNamespace(
            model_config=SimpleNamespace(model="m", max_model_len=1000),
            scheduler_config=SimpleNamespace(max_num_seqs=1),
            cache_config=SimpleNamespace(kv_cache_memory_bytes=123),
            kv_transfer_config=None,
        )


class _SlotConnector:
    def cpu_slot_bytes(self) -> int:
        return 100


def test_host_tier_info(monkeypatch) -> None:
    manager = make_manager()
    manager.set_ram_capacity(4)
    manager._auto_pin_entries.append(
        {
            "req_id": "g",
            "tail": b"t",
            "tokens": 40,
            "blocks": [object(), object()],
            "grp_blocks": [],
            "anchors": 2,
            "parked_at": 1.0,
            "last_used": 1.0,
        }
    )
    manager._ram_sessions["r"] = {
        "req_id": "r",
        "tokens": 20,
        "num_blocks": 5,
        "slots": [0, 1],
        "anchors": 1,
        "parked_at": 2.0,
        "last_used": 2.0,
    }
    monkeypatch.setattr(
        scheduler_module, "get_kv_cache_capacity", lambda *_: (999, 1.0)
    )
    scheduler = _InfoScheduler(manager, connector=_SlotConnector())

    info = scheduler.host_tier_info()

    assert info["config"]["gpu_total_tokens"] == 999
    assert info["config"]["staging_slots"] == 4
    assert info["config"]["chunk_slots"] == 7
    assert info["config"]["block_size"] == 4
    assert info["config"]["ram_slot_bytes"] == 100
    assert info["config"]["ssd"]["enabled"] is False

    sessions = info["sessions"]
    assert [s["id"] for s in sessions["gpu"]] == ["g"]
    assert sessions["gpu"][0]["anchors"] == 2
    assert sessions["gpu"][0]["bytes"] == 2 * 100
    assert sessions["ram"][0]["bytes"] == 2 * 100
    assert sessions["ssd"] == []
    # last_used is exported as an absolute epoch, not a monotonic tick.
    assert sessions["gpu"][0]["last_used"] > 1_000_000_000.0
