# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mamba "align" prefill chunk splitting (`_mamba_block_aligned_split`).

Invariant: slot `p` holds the state after exactly `(p + 1) * block_size` tokens.
State is written at chunk ends, so a chunk ending mid-block leaves its slot at
the wrong offset, and a later chunk crossing that boundary publishes it anyway.
Requests resuming from it then restore a truncated state (#43559).
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.utils.math_utils import cdiv
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
)
from vllm.v1.request import Request

from .utils import create_requests

pytestmark = pytest.mark.cpu_test

# Mirrors the deployment where the poisoning was observed (Qwen3.6-27B): mamba
# block 1600, MTP with 3 draft tokens, prompts shorter than 2 mamba blocks.
ATTN_BLOCK_SIZE = 16
MAMBA_BLOCK_SIZE = 1600
NUM_SPEC = 3
PROMPT_LEN = 2002
MAMBA_GROUP_ID = 1


def _make_hybrid_kv_cache_manager() -> KVCacheManager:
    config = KVCacheConfig(
        num_blocks=10000,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["full_layer"],
                FullAttentionSpec(
                    block_size=ATTN_BLOCK_SIZE,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            ),
            KVCacheGroupSpec(
                ["mamba_layer"],
                MambaSpec(
                    block_size=MAMBA_BLOCK_SIZE,
                    shapes=((1, 1),),
                    dtypes=(torch.float32,),
                    mamba_cache_mode="align",
                    num_speculative_blocks=NUM_SPEC,
                ),
            ),
        ],
    )
    return KVCacheManager(
        config,
        max_model_len=262144,
        scheduler_block_size=MAMBA_BLOCK_SIZE,
        hash_block_size=ATTN_BLOCK_SIZE,
        enable_caching=True,
        use_eagle=True,
    )


def _split(
    request: Request,
    num_new_tokens: int,
    use_eagle: bool = True,
    partial_hit: bool = False,
    num_new_local_computed_tokens: int = 0,
    num_external_computed_tokens: int = 0,
) -> int:
    """Call the real `Scheduler._mamba_block_aligned_split` on a stub self."""
    stub = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=MAMBA_BLOCK_SIZE),
        use_eagle=use_eagle,
        max_num_scheduled_tokens=16384,
        scheduler_config=SimpleNamespace(long_prefill_token_threshold=0),
        # `prefix_match_unit` finer than the block size (#46384).
        mamba_partial_cache_hit=partial_hit,
        hash_block_size=ATTN_BLOCK_SIZE,
    )
    return Scheduler._mamba_block_aligned_split(
        stub,
        request,
        num_new_tokens,
        num_new_local_computed_tokens,
        num_external_computed_tokens,
    )


def _run_chunked_prefill(
    manager: KVCacheManager, request: Request, budgets: list[int]
) -> dict[int, int]:
    """Prefill `request`, one step per entry in `budgets`.

    A zero-token split means the budget cannot fund an aligned chunk; the
    scheduler defers the request to a later step, so this does too.

    Returns physical mamba block id -> the token offset of the state it holds,
    mirroring the GDN kernel: the running slot ends up at the chunk end.
    """
    mamba_manager = manager.coordinator.single_type_managers[MAMBA_GROUP_ID]
    state_at: dict[int, int] = {}
    # `budgets` fragments the first steps; afterwards the request is alone and
    # gets as much as it can use, so the prefill always finishes.
    for step in range(len(budgets) + 64):
        computed = request.num_computed_tokens
        if computed >= request.num_tokens:
            break
        budget = budgets[step] if step < len(budgets) else request.num_tokens
        num_new = _split(request, min(request.num_tokens - computed, budget))
        if num_new == 0:
            continue
        assert (
            manager.allocate_slots(request, num_new, num_lookahead_tokens=NUM_SPEC)
            is not None
        )
        request.num_computed_tokens = computed + num_new
        blocks = mamba_manager.req_to_blocks[request.request_id]
        running = cdiv(request.num_computed_tokens, MAMBA_BLOCK_SIZE) - 1
        state_at[blocks[running].block_id] = request.num_computed_tokens
    return state_at


def _count_cached_boundary_states(
    manager: KVCacheManager, request: Request, state_at: dict[int, int]
) -> int:
    """Assert every hash-cached mamba slot holds the state its hash claims.

    Covers both full-block snapshots (`(p + 1) * block_size`) and the
    partial-tail entries align mode registers at an exact token count.

    Returns the number of cached slots checked.
    """
    mamba_manager = manager.coordinator.single_type_managers[MAMBA_GROUP_ID]
    checked = 0
    for pos, block in enumerate(mamba_manager.req_to_blocks[request.request_id]):
        if block.is_null or block.block_hash is None:
            continue
        claimed = block.block_hash_num_tokens
        assert state_at.get(block.block_id) == claimed, (
            f"mamba slot {pos} is hashed as state@{claimed} but holds "
            f"state@{state_at.get(block.block_id)}"
        )
        checked += 1
    return checked


def _prefill(prompt_len: int, budgets: list[int]) -> int:
    manager = _make_hybrid_kv_cache_manager()
    (request,) = create_requests(1, num_tokens=prompt_len, block_size=ATTN_BLOCK_SIZE)
    state_at = _run_chunked_prefill(manager, request, budgets)
    assert request.num_computed_tokens == prompt_len, "prefill did not complete"
    return _count_cached_boundary_states(manager, request, state_at)


def test_fragmented_first_chunk_does_not_poison_mamba_prefix_cache() -> None:
    """EAGLE zeroes `last_cache_position` for prompts under two blocks.

    Past that point any chunk end used to be accepted, so a short first chunk
    (concurrent prefills sharing the budget) left slot 0 at state@364 while the
    next chunk crossed 1600 and published slot 0 as state@1600.
    """
    _prefill(PROMPT_LEN, budgets=[364, PROMPT_LEN])


def test_fragmented_tail_chunk_does_not_poison_mamba_prefix_cache() -> None:
    """Same poisoning one block in, where a hit is still cacheable.

    `last_cache_position` is 1600, so the chunk ending there is cached. The next
    chunk used to be free to stop mid-block (slot 1 at state@2600) and the one
    after it crossed 3200, publishing slot 1 as state@3200.
    """
    assert _prefill(3602, budgets=[1600, 1000, 3602]) > 0


@pytest.mark.parametrize("first_chunk", [800, 900, 1599, 1601, 2000])
def test_intermediate_chunk_ends_stay_block_aligned(first_chunk: int) -> None:
    """Every non-final prefill chunk must end on a mamba block boundary."""
    _prefill(PROMPT_LEN, budgets=[first_chunk, PROMPT_LEN, PROMPT_LEN])


@pytest.mark.parametrize(
    ("block_size", "prompt_len", "budgets"),
    [
        # Kimi-K3-scale mamba blocks: TP8 shards the recurrent state 8 ways
        # (~1.5k block), DEP16 keeps it whole (~12k block). The first budget
        # walks the request up to `last_cache_position`; the second is the
        # sub-block fragment that lands in the unguarded tail region.
        (1536, 30000, [27648, 1024, 30000]),
        (12288, 30000, [12288, 4000, 30000]),
        (12288, 41000, [24576, 8000, 41000]),
    ],
)
def test_poisoning_is_block_size_independent(
    monkeypatch: pytest.MonkeyPatch,
    block_size: int,
    prompt_len: int,
    budgets: list[int],
) -> None:
    """The invariant is per-block, so large mamba blocks are not safer."""
    import sys

    monkeypatch.setattr(sys.modules[__name__], "MAMBA_BLOCK_SIZE", block_size)
    assert _prefill(prompt_len, budgets=budgets) > 0


def test_durable_boundaries_survive_head_free(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The head-prefix free must not drop durable snapshot boundaries.

    `remove_skipped_blocks` frees the whole skipped range before the manager's
    retention logic runs, so a state block sitting at `cadence - block_size`
    (the pre-cadence anchor MTP needs) was freed before it could be pinned.
    A restored request then found no anchor and replayed the prefix.
    """
    from vllm import envs

    monkeypatch.setattr(envs, "VLLM_MAMBA_CKPT_TOKENS", 2 * MAMBA_BLOCK_SIZE)
    manager = _make_hybrid_kv_cache_manager()
    mamba = manager.coordinator.single_type_managers[MAMBA_GROUP_ID]
    assert mamba._ckpt_tokens == 2 * MAMBA_BLOCK_SIZE

    (request,) = create_requests(1, num_tokens=3602, block_size=ATTN_BLOCK_SIZE)
    _run_chunked_prefill(manager, request, [])
    assert request.num_computed_tokens == 3602

    # A later step frees the head prefix up to the prompt end.
    mamba.remove_skipped_blocks(request.request_id, 3602, 3602)

    win = mamba.take_durable_window(request.request_id)
    assert sorted(b.block_hash_num_tokens for b in win) == [
        MAMBA_BLOCK_SIZE,
        2 * MAMBA_BLOCK_SIZE,
    ]
    assert all(b.pinned for b in win)
    # The retained entries are detached from the table like any freed block.
    blocks = mamba.req_to_blocks[request.request_id]
    assert blocks[0].is_null and blocks[1].is_null


def test_align_ckpt_tokens_floors_to_block_size() -> None:
    from vllm.v1.core.kv_cache_utils import align_ckpt_tokens

    assert align_ckpt_tokens(0, 1600) == 0
    assert align_ckpt_tokens(32000, 1600) == 32000
    assert align_ckpt_tokens(31680, 1584) == 31680
    assert align_ckpt_tokens(32000, 1584) == 31680
    assert align_ckpt_tokens(1000, 1584) == 1584
    assert align_ckpt_tokens(32000, 0) == 0


def test_mtp1_block_size_1584_cadence_aligns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MTP1 selects mamba block 1584, so cadence 32000 must floor to 31680."""
    import sys

    from vllm import envs

    monkeypatch.setattr(sys.modules[__name__], "MAMBA_BLOCK_SIZE", 1584)
    monkeypatch.setattr(envs, "VLLM_MAMBA_CKPT_TOKENS", 32000)
    manager = _make_hybrid_kv_cache_manager()
    mamba = manager.coordinator.single_type_managers[MAMBA_GROUP_ID]
    assert mamba.block_size == 1584
    assert mamba._ckpt_tokens == 31680


def test_scheduler_cadence_aligns_to_block_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The scheduler's cadence boundaries are block-aligned too (MTP1/1584)."""
    import sys

    from vllm import envs
    from vllm.v1.core.kv_cache_utils import align_ckpt_tokens

    monkeypatch.setattr(sys.modules[__name__], "MAMBA_BLOCK_SIZE", 1584)
    monkeypatch.setattr(envs, "VLLM_MAMBA_CKPT_TOKENS", 32000)
    (request,) = create_requests(1, num_tokens=40000, block_size=ATTN_BLOCK_SIZE)
    # The first chunk stops at the aligned pre-cadence (31680 - 1584), not at
    # the raw 32000 - 1584 = 30416.
    assert _split(request, 33000) == align_ckpt_tokens(32000, 1584) - 1584


def test_resumed_session_reclaims_cached_anchors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A request resuming from the prefix cache re-claims the session's durable
    anchors instead of leaving them as unowned idle blocks.

    A mamba prefix hit adopts only the single state block at the resume
    boundary; without the re-claim the older anchors are neither protected by
    the resumed request's window nor carried into its keep-alive entry, so a
    second park/restore cycle loses them and a deep revert recomputes.
    """
    from vllm import envs

    monkeypatch.setattr(envs, "VLLM_MAMBA_CKPT_TOKENS", 2 * MAMBA_BLOCK_SIZE)
    manager = _make_hybrid_kv_cache_manager()
    mamba = manager.coordinator.single_type_managers[MAMBA_GROUP_ID]

    (producer,) = create_requests(
        1,
        num_tokens=3602,
        block_size=ATTN_BLOCK_SIZE,
        same_prompt=True,
        req_ids=["producer"],
    )
    _run_chunked_prefill(manager, producer, [])
    mamba.remove_skipped_blocks(producer.request_id, 3602, 3602)
    anchors = mamba.take_durable_window(producer.request_id)
    assert sorted(b.block_hash_num_tokens for b in anchors) == [
        MAMBA_BLOCK_SIZE,
        2 * MAMBA_BLOCK_SIZE,
    ]
    # The session goes to the host tier and comes back: its anchors are
    # re-loaded as ordinary idle cached blocks (hash kept, no owner).
    for blk in anchors:
        mamba._release_durable_anchor(blk)
    manager.free(producer)

    (consumer,) = create_requests(
        1,
        num_tokens=3602,
        block_size=ATTN_BLOCK_SIZE,
        same_prompt=True,
        req_ids=["consumer"],
    )
    # The consumer is scheduled in a later step than the producer.
    manager.new_step_starts()
    computed_blocks, num_computed, _ = manager.get_computed_blocks(consumer)
    assert num_computed >= 2 * MAMBA_BLOCK_SIZE
    num_new = _split(
        consumer,
        consumer.num_tokens - num_computed,
        num_new_local_computed_tokens=num_computed,
    )
    assert (
        manager.allocate_slots(
            consumer,
            num_new,
            num_new_computed_tokens=num_computed,
            new_computed_blocks=computed_blocks,
            num_lookahead_tokens=NUM_SPEC,
        )
        is not None
    )

    win = mamba._durable_win.get(consumer.request_id, [])
    assert all(b.pinned for b in win)
    adopted = mamba.take_durable_window(consumer.request_id)
    assert sorted((b.block_id, b.block_hash_num_tokens) for b in adopted) == sorted(
        (b.block_id, b.block_hash_num_tokens) for b in anchors
    )


def test_preempted_request_does_not_re_adopt_anchors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A re-scheduled (preempted) request must not re-claim its own anchors.

    After preemption the cached anchors are this request's own idle blocks;
    pinning them would withdraw them from the free queue and starve the
    request's next allocation (pool-boundary livelock).
    """
    from vllm import envs

    monkeypatch.setattr(envs, "VLLM_MAMBA_CKPT_TOKENS", 2 * MAMBA_BLOCK_SIZE)
    manager = _make_hybrid_kv_cache_manager()
    mamba = manager.coordinator.single_type_managers[MAMBA_GROUP_ID]

    (producer,) = create_requests(
        1,
        num_tokens=3602,
        block_size=ATTN_BLOCK_SIZE,
        same_prompt=True,
        req_ids=["producer"],
    )
    _run_chunked_prefill(manager, producer, [])
    mamba.remove_skipped_blocks(producer.request_id, 3602, 3602)
    for blk in mamba.take_durable_window(producer.request_id):
        mamba._release_durable_anchor(blk)
    manager.free(producer)

    (consumer,) = create_requests(
        1,
        num_tokens=3602,
        block_size=ATTN_BLOCK_SIZE,
        same_prompt=True,
        req_ids=["consumer"],
    )
    manager.new_step_starts()
    computed_blocks, num_computed, _ = manager.get_computed_blocks(consumer)
    assert (
        manager.allocate_slots(
            consumer,
            _split(
                consumer,
                consumer.num_tokens - num_computed,
                num_new_local_computed_tokens=num_computed,
            ),
            num_new_computed_tokens=num_computed,
            new_computed_blocks=computed_blocks,
            num_lookahead_tokens=NUM_SPEC,
        )
        is not None
    )
    assert mamba._durable_win.get(consumer.request_id)

    # Preempt: the window and the request's blocks are released.
    manager.free(consumer)
    consumer.num_preemptions = 1

    # Re-schedule: the prefix hits again, but the anchors stay unowned.
    manager.new_step_starts()
    computed_blocks, num_computed, _ = manager.get_computed_blocks(consumer)
    assert num_computed >= 2 * MAMBA_BLOCK_SIZE
    assert (
        manager.allocate_slots(
            consumer,
            _split(
                consumer,
                consumer.num_tokens - num_computed,
                num_new_local_computed_tokens=num_computed,
            ),
            num_new_computed_tokens=num_computed,
            new_computed_blocks=computed_blocks,
            num_lookahead_tokens=NUM_SPEC,
        )
        is not None
    )
    assert not mamba._durable_win.get(consumer.request_id)


@pytest.mark.parametrize("partial_hit", [False, True])
@pytest.mark.parametrize("resume_at", [331, 1599, 1601, 2531, 3011])
def test_unaligned_resume_never_runs_past_its_block(
    partial_hit: bool, resume_at: int
) -> None:
    """A prefill resuming mid-block must re-align before crossing a boundary.

    Reachable with a finer `prefix_match_unit` (its partial-tail stop ends a
    chunk off-grid by design) and with unaligned external tokens from a KV
    connector.
    """
    prompt_len = 3602
    (request,) = create_requests(1, num_tokens=prompt_len, block_size=ATTN_BLOCK_SIZE)
    tail_boundary = prompt_len // ATTN_BLOCK_SIZE * ATTN_BLOCK_SIZE

    pos, ends = resume_at, []
    while pos < prompt_len:
        request.num_computed_tokens = pos
        num_new = _split(request, prompt_len - pos, partial_hit=partial_hit)
        assert num_new > 0, f"no progress at {pos}"
        if pos % MAMBA_BLOCK_SIZE != 0:
            block_end = (pos // MAMBA_BLOCK_SIZE + 1) * MAMBA_BLOCK_SIZE
            assert pos + num_new <= block_end, (
                f"chunk [{pos}, {pos + num_new}) starts mid-block and runs past "
                f"{block_end}; the slot holding state@{pos} gets hashed as "
                f"state@{block_end}"
            )
        pos += num_new
        ends.append(pos)

    for end in ends[:-1]:
        aligned = end % MAMBA_BLOCK_SIZE == 0
        assert aligned or (partial_hit and end == tail_boundary), (
            f"intermediate chunk end {end} is neither block-aligned nor the "
            f"partial-tail boundary"
        )
