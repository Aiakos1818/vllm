# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import itertools
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, overload

from vllm import envs
from vllm.distributed.kv_events import BlockStored, KVCacheEvent
from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv
from vllm.v1.core.host_tier_ssd import evict_sort_key
from vllm.v1.core.kv_cache_coordinator import (
    HybridKVCacheCoordinator,
    get_kv_cache_coordinator,
)
from vllm.v1.core.kv_cache_metrics import KVCacheMetricsCollector
from vllm.v1.core.kv_cache_utils import (
    KVCacheBlock,
    KVCacheBlockCopy,
    resolve_block_hashes,
)
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    CrossAttentionSpec,
    EncoderOnlyAttentionSpec,
    KVCacheConfig,
    get_kv_cache_spec_kind,
    get_kv_cache_spec_sliding_window,
)
from vllm.v1.metrics.stats import PrefixCacheStats
from vllm.v1.request import Request, RequestStatus

logger = init_logger(__name__)


@dataclass
class KVCacheBlocks:
    """
    The allocation result of KVCacheManager, work as the interface between
    Scheduler and KVCacheManager, to hide KVCacheManager's internal data
    structure from the Scheduler.
    """

    blocks: tuple[Sequence[KVCacheBlock], ...]
    """
    `blocks[i][j]` refers to the i-th kv_cache_group
    and the j-th block of tokens.We don't use block of
    tokens as the outer dimension because it assumes all
    kv_cache_groups have the same number of blocks, which is true for now but
    will be broken if we want to give different block_size to different
    kv_cache_groups in the future.

    Each single type KVCacheBlocks could be represented as:
    - list[KVCacheBlock] for more than one KVCacheBlock
    - an empty tuple for requests without KVCacheBlock
      (a precomputed KVCacheBlocks is in KVCacheManager to avoid GC overhead)
    """

    def __add__(self, other: "KVCacheBlocks") -> "KVCacheBlocks":
        """Adds two KVCacheBlocks instances."""
        return KVCacheBlocks(
            tuple(
                list(itertools.chain(blk1, blk2))
                for blk1, blk2 in zip(self.blocks, other.blocks)
            )
        )

    @overload
    def get_block_ids(
        self,
        allow_none: Literal[False] = False,
    ) -> tuple[list[int], ...]: ...

    @overload
    def get_block_ids(
        self,
        allow_none: Literal[True] = True,
    ) -> tuple[list[int], ...] | None: ...

    def get_block_ids(
        self,
        allow_none: bool = False,
    ) -> tuple[list[int], ...] | None:
        """
        Converts the KVCacheBlocks instance to block_ids.

        Returns:
            tuple[list[int], ...]: A tuple of lists where:
                - the outer tuple corresponds to KV cache groups
                - each inner list contains the block_ids of the blocks in that
                  group
        """
        if allow_none and all(len(group) == 0 for group in self.blocks):
            return None
        return tuple([blk.block_id for blk in group] for group in self.blocks)

    def get_unhashed_block_ids(self) -> list[int]:
        """Get block_ids of unhashed blocks from KVCacheBlocks instance."""
        assert len(self.blocks) == 1, "Only one group is supported"
        return [block.block_id for block in self.blocks[0] if block.block_hash is None]

    def get_unhashed_block_ids_all_groups(self) -> list[list[int]]:
        """Get block_ids of unhashed blocks from KVCacheBlocks instance."""
        # Skip padding blocks.
        return [
            [
                block.block_id
                for block in group
                if block.block_hash is None and not block.is_null
            ]
            for group in self.blocks
        ]

    def new_empty(self) -> "KVCacheBlocks":
        """
        Creates a new KVCacheBlocks instance with no blocks.
        """
        return KVCacheBlocks(tuple(() for _ in range(len(self.blocks))))


class KVCacheManager:
    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        max_model_len: int,
        scheduler_block_size: int,
        hash_block_size: int,
        max_in_flight_tokens: int | None = None,
        enable_caching: bool = True,
        use_eagle: bool = False,
        log_stats: bool = False,
        enable_kv_cache_events: bool = False,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
        metrics_collector: KVCacheMetricsCollector | None = None,
        watermark: float = 0.0,
    ) -> None:
        self.max_model_len = max_model_len
        # When unset, fall back to `max_model_len` so the recycling-aware cap
        # collapses to the prior (uncapped) admission behavior. The scheduler
        # always supplies the real value at runtime.
        if max_in_flight_tokens is None:
            max_in_flight_tokens = max_model_len

        self.enable_caching = enable_caching
        self.enable_kv_cache_events = enable_kv_cache_events
        self.use_eagle = use_eagle
        self.log_stats = log_stats
        self.metrics_collector = metrics_collector
        # FIXME: make prefix cache stats conditional on log_stats. We still need
        # this comment because when the log stats is enabled there are still
        # potential configs we could expose in the future.
        self.prefix_cache_stats = PrefixCacheStats() if log_stats else None

        self.coordinator = get_kv_cache_coordinator(
            kv_cache_config=kv_cache_config,
            max_model_len=self.max_model_len,
            max_in_flight_tokens=max_in_flight_tokens,
            use_eagle=self.use_eagle,
            enable_caching=self.enable_caching,
            enable_kv_cache_events=enable_kv_cache_events,
            dcp_world_size=dcp_world_size,
            pcp_world_size=pcp_world_size,
            scheduler_block_size=scheduler_block_size,
            hash_block_size=hash_block_size,
            metrics_collector=self.metrics_collector,
        )
        self.num_kv_cache_groups = len(kv_cache_config.kv_cache_groups)
        self.block_pool = self.coordinator.block_pool
        self.kv_cache_config = kv_cache_config

        # Watermark: minimum number of KV cache blocks to keep free when
        # admitting waiting/preempted requests, to avoid frequent preemptions.
        assert watermark >= 0.0, "watermark must be non-negative"
        self.watermark_blocks = int(watermark * kv_cache_config.num_blocks)
        self.kv_cache_event_metadata = tuple(
            (
                get_kv_cache_spec_kind(group.kv_cache_spec).value,
                get_kv_cache_spec_sliding_window(group.kv_cache_spec),
            )
            for group in kv_cache_config.kv_cache_groups
        )

        # Pre-constructed KVCacheBlocks with no blocks, callers should use this
        # via create_kv_cache_blocks instead of creating new ones to avoid GC
        # overhead.
        #
        # We use nested tuples to ensure the empty KVCacheBlocks is immutable.
        self.empty_kv_cache_blocks = KVCacheBlocks(
            tuple(() for _ in range(self.num_kv_cache_groups))
        )

        # Off-table cow blocks handed to a KV connector for partial-tail
        # offload; pinned until the request's blocks are freed.
        self._partial_tail_pins: dict[str, list[KVCacheBlock]] = {}

        # Auto keep-alive (session protection). Each entry holds the cached
        # prefix-chain blocks of one long finished request. Pinned entries are
        # withdrawn from the free queue so later cache pressure cannot evict /
        # invalidate them; under admission pressure the SMALLEST entries are
        # released first ("缓存最小的先出").
        self._auto_pin_entries: list[dict[str, Any]] = []

        # --- Host-tier session spill (RAM parking) state. ---
        # Entries chosen for spill that are still GPU-pinned while their
        # GPU->CPU store is in flight. Blocks are only unpinned/freed after the
        # store completes (correctness-first ordering).
        self._spill_hold: dict[str, dict[str, Any]] = {}
        # Parked sessions living on the host tier: keyed by request_id.
        self._ram_sessions: dict[str, dict[str, Any]] = {}
        # Host-tier slot allocator, keyed by store/load job id to the entry
        # being moved, so completion callbacks can drive the transition.
        self._ram_capacity: int = 0
        self._ram_free: list[tuple[int, int]] = []  # disjoint (start, len)
        self._ram_ext_job_to_req: dict[int, str] = {}
        # Resume requests waiting for a spill/restore load to finish.
        self._ram_waiting: dict[str, int] = {}  # request_id -> load job id

    @property
    def usage(self) -> float:
        """Get the KV cache usage.

        Returns:
            The KV cache usage (between 0.0 and 1.0).
        """
        return self.block_pool.get_usage()

    def make_prefix_cache_stats(self) -> PrefixCacheStats | None:
        """Get (and reset) the prefix cache stats.

        Returns:
            The current prefix caching stats, or None if logging is disabled.
        """
        if not self.log_stats:
            return None
        stats = self.prefix_cache_stats
        self.prefix_cache_stats = PrefixCacheStats()
        return stats

    def prefix_cache_lookup_enabled(self, request: Request) -> bool:
        """Whether a local prefix cache lookup may be run for this request."""
        return self.enable_caching and not request.skip_reading_prefix_cache

    def record_prefix_cache_stats(self, request: Request, num_hits: int) -> None:
        # Don't count a request that skipped the cache lookup.
        if not self.log_stats or not self.prefix_cache_lookup_enabled(request):
            return
        assert self.prefix_cache_stats is not None
        self.prefix_cache_stats.record(
            num_tokens=request.num_tokens,
            num_hits=num_hits,
            preempted=request.num_preemptions > 0,
        )

    def get_computed_blocks(self, request: Request) -> tuple[KVCacheBlocks, int, int]:
        """Get the computed (cached) blocks for the request.
        Note that the computed blocks must be full.

        Args:
            request: The request to get the computed blocks.

        Returns:
            A tuple containing:
                - A list of blocks that are computed for the request.
                - The number of computed tokens.
                - ``shared_prefix_boundary``: the block-aligned token position of
                  a shared prefix that a sparse-retention group (Mamba / sliding
                  window) has not cached yet (Marconi-style APC), or 0 if none.
                  Pinned so ``VLLM_PREFIX_CACHE_RETENTION_INTERVAL`` does not drop
                  the junction and defeat cross-request reuse.
        """
        # We skip finding the prefix cache hit when prefix caching is
        # disabled or the request is marked as skipping kv cache read
        # (which happens when the request requires prompt logprobs
        # or calls a pooling model with all pooling).
        if not self.prefix_cache_lookup_enabled(request):
            return self.empty_kv_cache_blocks, 0, 0

        # NOTE: When all tokens hit the cache, we must recompute the last token
        # to obtain logits. Thus, set max_cache_hit_length to prompt_length - 1.
        # This can trigger recomputation of an entire block, rather than just
        # the single last token, because allocate_slots() requires
        # num_computed_tokens to be block-size aligned. Removing this limitation
        # could slightly improve performance in the future.
        max_cache_hit_length = request.num_tokens - 1
        computed_blocks, num_new_computed_tokens, num_uncached = (
            self.coordinator.find_longest_cache_hit(
                request.block_hashes, max_cache_hit_length
            )
        )
        if envs.RAMTRACE:
            try:
                with open(
                    envs.RAMTRACE_LOG, "a"
                ) as _f:
                    _f.write(
                        f"diag reconciled req={request.request_id} "
                        f"hit={num_new_computed_tokens} uncached={num_uncached}\n"
                    )
            except Exception:
                pass

        # When kv_cache_report_mode is "full", emit BlockStored events
        # for the reused prefix cache blocks so that external consumers
        # (e.g. gateway) can learn about them.
        if (
            num_new_computed_tokens > 0
            and self.enable_kv_cache_events
            and getattr(request, "kv_cache_report_mode", "incremental") == "full"
        ):
            for group_idx, group_blocks in enumerate(computed_blocks):
                num_blocks = len(group_blocks)
                if num_blocks > 0:
                    group = self.kv_cache_config.kv_cache_groups[group_idx]
                    block_size = group.kv_cache_spec.block_size
                    self.block_pool.emit_cached_block_events(
                        request,
                        num_blocks,
                        block_size,
                        group_idx,
                    )

        # The junction to pin is where the lagging sparse-retention group stops
        # (``num_new_computed_tokens``) plus the uncached shared prefix -- i.e.
        # the longest single-group hit. Sub-block gaps are left to the mask,
        # which floors to the alignment boundary (a no-op there).
        shared_prefix_boundary = (
            num_new_computed_tokens + num_uncached if num_uncached else 0
        )

        blocks = self.create_kv_cache_blocks(computed_blocks)
        return blocks, num_new_computed_tokens, shared_prefix_boundary

    def get_computed_blocks_for_connector(
        self, request: Request
    ) -> tuple[KVCacheBlocks, int, int, bool]:
        """Local prefix-cache lookup for a request scheduled with a KV connector.

        Hybrid (Mamba + full-attention) models can have per-group prefix hits
        diverge under block pressure: the full-attention tail may be evicted
        while a deeper Mamba state survives, or vice versa. Report the
        full-attention hit as the local prefix - the connector transfers the
        remaining suffix and the Mamba state is transferred unconditionally by
        nixl's ``_apply_prefix_caching`` - and flag when that hit ran deeper
        than a lagging group. Such a hit only has a valid Mamba state at its
        boundary if the connector supplies it, so the caller must fall back to
        ``get_computed_blocks`` to reconcile when no external tokens are found.

        Non-hybrid models and already-convergent hits use ``get_computed_blocks``.

        Returns:
            The ``get_computed_blocks`` triple (blocks, number of local computed
            tokens, shared-prefix boundary) plus ``hit_diverged``.
        """
        coordinator = self.coordinator
        if not (
            self.kv_cache_config.has_mamba_layers
            and isinstance(coordinator, HybridKVCacheCoordinator)
            and coordinator.full_attention_group_id is not None
        ):
            return *self.get_computed_blocks(request), False

        if not self.prefix_cache_lookup_enabled(request):
            return self.empty_kv_cache_blocks, 0, 0, False

        fa_group_id = coordinator.full_attention_group_id
        computed, per_group_hits = coordinator.find_longest_cache_hit_per_group(
            request.block_hashes, request.num_tokens - 1
        )
        if envs.RAMTRACE:
            try:
                with open(
                    envs.RAMTRACE_LOG, "a"
                ) as _f:
                    _f.write(
                        f"diag lookup req={request.request_id} "
                        f"groups={len(per_group_hits)} fa={fa_group_id} "
                        f"hits={list(per_group_hits)}\n"
                    )
            except Exception:
                pass
        if any(hit > per_group_hits[fa_group_id] for hit in per_group_hits):
            # A lagging group hit deeper than full attention means its
            # full-attention blocks were evicted; use the reconciled boundary
            # that every group agrees on.
            return *self.get_computed_blocks(request), False

        num_local = per_group_hits[fa_group_id]
        blocks = self.create_kv_cache_blocks(computed)
        # Per-group lookups do not detect an uncached shared prefix (boundary 0).
        return blocks, num_local, 0, min(per_group_hits) < num_local

    def allocate_slots(
        self,
        request: Request,
        num_new_tokens: int,
        num_new_computed_tokens: int = 0,
        new_computed_blocks: KVCacheBlocks | None = None,
        num_lookahead_tokens: int = 0,
        num_external_computed_tokens: int = 0,
        delay_cache_blocks: bool = False,
        num_encoder_tokens: int = 0,
        full_sequence_must_fit: bool = False,
        reserved_blocks: int = 0,
        has_scheduled_reqs: bool = True,
    ) -> KVCacheBlocks | None:
        """Add slots for a request with new tokens to append.

        Args:
            request: The request to allocate slots.
            num_new_tokens: The number of new tokens to be allocated and computed.
            num_new_computed_tokens: The number of new computed tokens just
                hitting the prefix caching, excluding external tokens.
            new_computed_blocks: The cached blocks for the above new computed
                tokens, grouped as a tuple by kv cache groups.
            num_lookahead_tokens: The number of speculative tokens to allocate.
                This is used by spec decode proposers with kv-cache such
                as eagle.
            num_external_computed_tokens: The number of tokens that their
                KV caches are not cached by vLLM but cached by the connector.
            delay_cache_blocks: Whether to skip caching the blocks. This is
                used by P/D when allocating blocks used in a KV transfer
                which will complete in a future step.
            num_encoder_tokens: The number of encoder tokens to allocate for
                cross-attention in encoder-decoder models(e.g., Whisper).
                For decoder-only models, this should be 0.
            full_sequence_must_fit: Only allocate blocks if the KV cache has enough
                free blocks to hold the full sequence, accounting for prefix cache hits
                and sliding window. Used as an admission gate to prevent over-admitting
                requests when chunked prefill would otherwise only check the first chunk
            reserved_blocks: Number of free blocks that must be left available for
                other in-flight sequences to complete. The actual allocation is only
                made if it fits within (free blocks - reserved_blocks). Used to gate
                async KV-connector loads so their initial allocation cannot consume
                blocks an already in-flight (prefilling) sequence is relying on.
            has_scheduled_reqs: Whether any requests are already scheduled to run
                this step, controls whether watermark is applied.

        Blocks layout:
        ```
        ----------------------------------------------------------------------
        | < comp > | < new_comp > | < ext_comp >  | < new >  | < lookahead > |
        ----------------------------------------------------------------------
                                                  |   < to be computed >     |
        ----------------------------------------------------------------------
                                  |            < to be allocated >           |
        ----------------------------------------------------------------------
                                  | < to be cached (roughly, |
                                  | details below)>          |
        ----------------------------------------------------------------------
        | Prefix-cached tokens from either vLLM   |
        | or connector. Can be safely removed if  |
        | they are outside sliding window.        |
        ----------------------------------------------------------------------
        |   < cached by vLLM >    | not cached by |
                                  | vLLM, but     |
        | ref_cnt  | ref_cnt not  | cached by     |
        | increased| increased yet| connector     |
        ----------------------------------------------------------------------
        ```

        Abbrivations:

        ```
        comp      = request.num_computed_tokens
        new_comp  = num_new_computed_tokens
                  = len(new_computed_blocks) * block_size
        ext_comp  = num_external_computed_tokens, cached by the connector
        new       = num_new_tokens, including unverified draft tokens
        lookahead = num_lookahead_tokens
        ```

        NOTE: for new tokens which include both verified and unverified draft
        tokens, we only cache the verified tokens (by capping the number at
        `request.num_tokens`).

        The allocation has three stages:
        - Free unnecessary blocks in `comp` and check
           if we have sufficient free blocks (return None if not).
        - Handle prefix tokens (`comp + new_comp + ext_comp`):
            - Free unnecessary blocks (e.g. outside sliding window)
            - Allocate new blocks for `ext_comp` tokens inside
              sliding window
        - Allocate new blocks for tokens to be computed (`new + lookahead`)

        Returns:
            A list of new allocated blocks.
        """
        # When loading KV data asynchronously, we may have zero new tokens to
        # compute while still allocating slots for externally computed tokens.
        if num_new_tokens == 0 and num_external_computed_tokens == 0:
            raise ValueError(
                "num_new_tokens must be greater than 0 when there are no "
                "external computed tokens"
            )

        if new_computed_blocks is not None:
            new_computed_block_list = new_computed_blocks.blocks
        else:
            new_computed_block_list = self.empty_kv_cache_blocks.blocks

        # The number of computed tokens is the number of computed tokens plus
        # the new prefix caching hits
        num_local_computed_tokens = (
            request.num_computed_tokens + num_new_computed_tokens
        )
        total_computed_tokens = min(
            num_local_computed_tokens + num_external_computed_tokens,
            self.max_model_len,
        )

        watermark_blocks = 0
        # The watermark is applied to waiting/preempted requests only, and only
        # when there's at least one request already scheduled.
        if has_scheduled_reqs and request.status in (
            RequestStatus.WAITING,
            RequestStatus.PREEMPTED,
        ):
            watermark_blocks = self.watermark_blocks

        if full_sequence_must_fit:
            # First check and fail if the full request sequence won't fit.
            full_num_tokens = min(request.num_tokens, self.max_model_len)

            num_blocks_to_allocate = self.coordinator.get_num_blocks_to_allocate(
                request_id=request.request_id,
                num_tokens=full_num_tokens,
                new_computed_blocks=new_computed_block_list,
                num_encoder_tokens=num_encoder_tokens,
                total_computed_tokens=total_computed_tokens,
                num_local_computed_tokens=num_local_computed_tokens,
                num_tokens_main_model=full_num_tokens,
                apply_admission_cap=True,
            )
            required_blocks = num_blocks_to_allocate + watermark_blocks
            if required_blocks > self.block_pool.get_num_free_blocks():
                if envs.RAMTRACE:
                    try:
                        with open(
                            envs.RAMTRACE_LOG, "a"
                        ) as _f:
                            _f.write(
                                f"alloc gate full req={request.request_id} "
                                f"need={required_blocks} "
                                f"free={self.block_pool.get_num_free_blocks()} "
                                f"nalloc={num_blocks_to_allocate} "
                                f"wm={watermark_blocks}\n"
                            )
                    except Exception:
                        pass
                return None

        num_tokens_main_model = total_computed_tokens + num_new_tokens
        num_tokens_need_slot = min(
            num_tokens_main_model + num_lookahead_tokens, self.max_model_len
        )

        # Free the blocks that are skipped during the attention computation
        # (e.g., tokens outside the sliding window).
        # We can do this even if we cannot schedule this request due to
        # insufficient free blocks.
        # Should call this function before allocating new blocks to reduce
        # the number of evicted blocks.
        # Free on the processed-token basis: in-flight steps' attention windows
        # still read blocks below the optimistic boundary, and rejected spec
        # tokens can roll it back.
        self.coordinator.remove_skipped_blocks(
            request.request_id,
            max(0, total_computed_tokens - request.num_in_flight_tokens),
            num_prompt_tokens=request.num_prompt_tokens,
        )

        num_blocks_to_allocate = self.coordinator.get_num_blocks_to_allocate(
            request_id=request.request_id,
            num_tokens=num_tokens_need_slot,
            new_computed_blocks=new_computed_block_list,
            num_encoder_tokens=num_encoder_tokens,
            total_computed_tokens=num_local_computed_tokens
            + num_external_computed_tokens,
            num_local_computed_tokens=num_local_computed_tokens,
            num_tokens_main_model=num_tokens_main_model,
        )

        # Keep `reserved_blocks` free for other in-flight sequences, and an
        # additional watermark of headroom for waiting/preempted admissions.
        available_blocks = self.block_pool.get_num_free_blocks() - reserved_blocks
        required_blocks = num_blocks_to_allocate + watermark_blocks
        if required_blocks > available_blocks:
            # Cannot allocate new blocks
            if envs.RAMTRACE:
                try:
                    with open(
                        envs.RAMTRACE_LOG, "a"
                    ) as _f:
                        _f.write(
                            f"alloc gate req={request.request_id} "
                            f"need={required_blocks} avail={available_blocks} "
                            f"free={self.block_pool.get_num_free_blocks()} "
                            f"nalloc={num_blocks_to_allocate} "
                            f"reserved={reserved_blocks} wm={watermark_blocks} "
                            f"ntok={num_tokens_need_slot}\n"
                        )
                except Exception:
                    pass
            return None

        if (
            new_computed_block_list is not self.empty_kv_cache_blocks.blocks
            or num_external_computed_tokens > 0
        ):
            # Append the new computed blocks to the request blocks until now to
            # avoid the case where the new blocks cannot be allocated.
            self.coordinator.allocate_new_computed_blocks(
                request_id=request.request_id,
                new_computed_blocks=new_computed_block_list,
                num_local_computed_tokens=num_local_computed_tokens,
                num_external_computed_tokens=num_external_computed_tokens,
            )

        new_blocks = self.coordinator.allocate_new_blocks(
            request.request_id,
            num_tokens_need_slot,
            num_tokens_main_model,
            num_encoder_tokens,
        )

        # P/D: delay caching blocks if we have to recv from
        # remote. Update state for locally cached blocks.
        if not self.enable_caching or delay_cache_blocks:
            return self.create_kv_cache_blocks(new_blocks)

        # NOTE(woosuk): We want to commit (cache) up to num_local_computed_tokens
        # + num_external_computed_tokens + num_new_tokens, but must exclude
        # "non-committable" tokens (e.g., draft tokens that could be rejected).
        # Therefore, we cap the number at `request.num_tokens`, ensuring only
        # "finalized" tokens are cached.
        num_tokens_to_cache = min(
            total_computed_tokens + num_new_tokens,
            request.num_tokens,
        )
        self.coordinator.cache_blocks(request, num_tokens_to_cache)

        return self.create_kv_cache_blocks(new_blocks)

    def free(self, request: Request) -> None:
        """Free the blocks allocated for the request.
        We free the blocks in reverse order so that the tail blocks are evicted
        first when caching is enabled.

        Args:
            request: The request to free the blocks.
        """
        pins = self._partial_tail_pins.pop(request.request_id, None)
        if pins:
            self.block_pool.free_blocks(pins)
        self.coordinator.free(request.request_id)

    # ------------------------------------------------------------------ #
    # Auto keep-alive (session protection). See ``_auto_pin_entries``.
    # ------------------------------------------------------------------ #
    def pin_request_auto(self, request: Request) -> int:
        """Keep alive the whole cached prefix-chain of a finished request.

        The request's own blocks (still alive here, before ``free``) become
        pinned: they are withdrawn from the free queue and can no longer be
        reused/evicted by other requests, so a later re-send/resume of the
        same conversation keeps hitting the full prefix.

        Returns the number of blocks pinned (0 if nothing eligible).
        """
        if not self.enable_caching:
            return 0

        entries: list[KVCacheBlock] = []
        total_tokens = 0
        managers = self.coordinator.single_type_managers
        per_mgr: list[list[KVCacheBlock]] = [[] for _ in managers]
        # Pin the chain across ALL single-type groups (attention + GDN/mamba):
        # a prefix-cache chain only survives if none of its pages is reused, and
        # reuse of a page in any group invalidates the whole chain in this build.
        # ``block_size`` (per group, may differ) weights the entry size so the
        # eviction sort can compare sessions by occupied cache tokens.
        for mgr_idx, manager in enumerate(managers):
            blocks = manager.req_to_blocks.get(request.request_id, ())
            if envs.RAMTRACE:
                try:
                    n_nonnull = sum(1 for b in blocks if not b.is_null)
                    n_hash = sum(1 for b in blocks if b.block_hash is not None)
                    with open(
                        envs.RAMTRACE_LOG, "a"
                    ) as _f:
                        _f.write(
                            f"pin g{mgr_idx} all={len(blocks)} "
                            f"nonnull={n_nonnull} hash={n_hash}\n"
                        )
                except Exception:
                    pass
            block_size = manager.block_size
            for block in blocks:
                if block.is_null or block.block_hash is None:
                    continue
                entries.append(block)
                per_mgr[mgr_idx].append(block)
                total_tokens += block_size

        # Durable Mamba/GDN state at the reusable boundary: a restored chain
        # resumes at the full-attention hit, but GDN is recurrent, so it needs
        # the SSM state snapshot at that boundary. The state block is not part
        # of ``req_to_blocks`` (superseded states are nulled there); it lives in
        # the prefix cache keyed by the chain hash. Capture the highest cached
        # state block at or below the attention reusable boundary so the
        # restored request resumes with a valid recurrent state.
        non_mamba_tokens = [
            len(per_mgr[i]) * managers[i].block_size
            for i, m in enumerate(managers)
            if type(getattr(m, "kv_cache_spec", None)).__name__ != "MambaSpec"
            and per_mgr[i]
        ]
        if non_mamba_tokens and not envs.VLLM_SPILL_NO_MAMBA:
            attn_tokens = min(non_mamba_tokens)
            use_eagle = any(getattr(m, "use_eagle", False) for m in managers)
            if use_eagle:
                # EAGLE reuses one block short of the matched boundary.
                attn_tokens = max(0, attn_tokens - managers[0].block_size)
            for mgr_idx, manager in enumerate(managers):
                if type(getattr(manager, "kv_cache_spec", None)).__name__ != "MambaSpec":
                    continue
                try:
                    rhs = resolve_block_hashes(
                        request.block_hashes,
                        self.block_pool.hash_block_size,
                        manager.block_size,
                    )
                    max_idx = (
                        min(len(rhs), attn_tokens // manager.block_size) - 1
                    )
                    state_blk = None
                    for j in range(max_idx, -1, -1):
                        found = self.block_pool.get_cached_block(rhs[j], [mgr_idx])
                        if found:
                            state_blk = found[0]
                            break
                except Exception:
                    state_blk = None
                if state_blk is None or state_blk.block_hash is None:
                    continue
                if any(b is state_blk for b in entries):
                    continue
                per_mgr[mgr_idx].append(state_blk)
                entries.append(state_blk)
                total_tokens += manager.block_size
                if envs.RAMTRACE:
                    try:
                        with open(
                            envs.RAMTRACE_LOG, "a"
                        ) as _f:
                            _f.write(
                                f"mamba cap g{mgr_idx} idx={j} "
                                f"blk={state_blk.block_id}\n"
                            )
                    except Exception:
                        pass

        # Durable Mamba anchors share the SAME protection as their chain: they
        # are folded into this keep-alive entry so they are released together
        # with the chain (only under admission pressure) and never become an
        # independent, unbounded pin. Anchors are already pinned during the
        # owner's run (MambaManager window); hand them over here and make them
        # ref-idle-pinned (free without enqueue, exactly like the chain blocks
        # become right after this, in ``_free_request_blocks``).
        anchor_ids: set[int] = set()
        for mgr_idx, manager in enumerate(managers):
            take = getattr(manager, "take_durable_window", None)
            if take is None:
                continue
            anchors = take(request.request_id)
            if not anchors:
                continue
            block_size = manager.block_size
            for blk in anchors:
                # ref is 1 (allocated, never freed during the run); this drops
                # it to 0 while pinned>0 keeps it out of the free queue.
                self.block_pool.free_blocks([blk])
                anchor_ids.add(id(blk))
                if any(b is blk for b in entries):
                    # The boundary state block captured above can itself be a
                    # cadence anchor. It is already in the entry; drop only its
                    # allocation ref (done above) to avoid a duplicate unpin.
                    continue
                entries.append(blk)
                total_tokens += block_size
                # Include the anchor in the per-group chain so it is stored and
                # restored with the session; this keeps truncated/deep reverts
                # hitting their mamba anchor after a host-tier restore.
                per_mgr[mgr_idx].append(blk)

        if not entries:
            return 0

        bh_set = set(request.block_hashes)
        # Subsumption: if an existing keep-alive chain is a prefix of this new
        # chain (same conversation resumed / re-sent, tail hash reappears), drop
        # the old entry and replace it with the longer chain.
        subsumed = [
            entry for entry in self._auto_pin_entries if entry["tail"] in bh_set
        ]
        for entry in subsumed:
            self._unpin_entry(entry)

        tail = request.block_hashes[-1] if request.block_hashes else None
        for block in entries:
            # Anchors were already pinned during the owner's run; only chain
            # blocks (and any other non-anchor blocks) need pinning here.
            if id(block) in anchor_ids:
                continue
            self.block_pool.pin_block(block)
        self._auto_pin_entries.append(
            {
                "tail": tail,
                "req_id": request.request_id,
                "blocks": entries,
                "grp_blocks": per_mgr,
                "tokens": total_tokens,
                "num_blocks": len(entries),
                "anchors": len(anchor_ids),
                "parked_at": time.monotonic(),
                "last_used": time.monotonic(),
            }
        )
        return len(entries)

    def _unpin_entry(self, entry: dict[str, Any]) -> None:
        for block in entry["blocks"]:
            self.block_pool.unpin_block(block)
        try:
            self._auto_pin_entries.remove(entry)
        except ValueError:
            pass

    def num_pinned_entries(self) -> int:
        return len(self._auto_pin_entries)

    def num_pinned_tokens(self) -> int:
        return sum(e["tokens"] for e in self._auto_pin_entries)

    def num_pinned_blocks(self) -> int:
        return sum(len(e["blocks"]) for e in self._auto_pin_entries)

    def num_pinned_anchors(self) -> int:
        return sum(e.get("anchors", 0) for e in self._auto_pin_entries)

    def num_pinned_anchor_sessions(self) -> int:
        return sum(1 for e in self._auto_pin_entries if e.get("anchors", 0) > 0)

    def ram_capacity(self) -> int:
        return self._ram_capacity

    def ram_slots_used(self) -> int:
        # Slots held by an in-flight restore are counted as used (they are not
        # in the free list until the load completes).
        return self._ram_capacity - self.ram_slots_free()

    def num_ram_sessions(self) -> int:
        return len(self._ram_sessions)

    # ------------------------------------------------------------------ #
    # Host-tier session spill (RAM parking).                             #
    # ------------------------------------------------------------------ #
    def set_ram_capacity(self, capacity: int) -> None:
        """Configure the host-tier slot capacity (0 disables the tier)."""
        self._ram_capacity = capacity
        self._ram_free = [(0, capacity)] if capacity > 0 else []
        if envs.RAMTRACE:
            try:
                with open(
                    envs.RAMTRACE_LOG, "a"
                ) as _f:
                    _f.write(f"ram capacity slots={capacity}\n")
            except Exception:
                pass

    def matches_pinned_chain(self, block_hashes: list[bytes]) -> bool:
        """True if a keep-alive pinned chain's tail is in this request's hashes."""
        if not block_hashes:
            return False
        bh = set(block_hashes)
        return any(
            e["tail"] is not None and e["tail"] in bh for e in self._auto_pin_entries
        )

    def ram_slots_free(self) -> int:
        return sum(length for _, length in self._ram_free)

    def alloc_ram_slots(self, n: int) -> list[int] | None:
        """Allocate n contiguous host slots; None if insufficient space."""
        if n <= 0 or n > self.ram_slots_free():
            return None
        for i, (start, length) in enumerate(self._ram_free):
            if length >= n:
                slots = list(range(start, start + n))
                if length == n:
                    del self._ram_free[i]
                else:
                    self._ram_free[i] = (start + n, length - n)
                return slots
        return None

    def free_ram_slots(self, slots: list[int]) -> None:
        if not slots:
            return
        slots = sorted(slots)
        self._ram_free.append((slots[0], len(slots)))
        # coalesce adjacent ranges
        self._ram_free.sort()
        merged: list[tuple[int, int]] = []
        for start, length in self._ram_free:
            if merged and merged[-1][0] + merged[-1][1] == start:
                s0, l0 = merged[-1]
                merged[-1] = (s0, l0 + length)
            else:
                merged.append((start, length))
        self._ram_free = merged

    def take_spill_candidates(self, need_free_blocks: int) -> list[dict[str, Any]]:
        """Move keep-alive entries into spill hold in host-tier eviction order.

        Small sessions (< ``VLLM_HOSTTIER_EVICT_SMALL_TOKENS``) are chosen
        first, then large ones; within each tier the least recently used is
        chosen first. This does NOT unpin/free: blocks stay pinned until their
        host store completes (correctness-first ordering), then are released by
        ``confirm_spill``. Returns the chosen entries.
        """
        chosen: list[dict[str, Any]] = []
        while self._auto_pin_entries:
            if self.block_pool.get_num_free_blocks() >= need_free_blocks:
                break
            victim = min(
                self._auto_pin_entries,
                key=lambda e: evict_sort_key(
                    e["tokens"], e.get("last_used", e["parked_at"])
                ),
            )
            req_id = victim["req_id"]
            self._auto_pin_entries.remove(victim)
            self._spill_hold[req_id] = victim
            chosen.append(victim)
        return chosen

    def abort_spill(self, req_id: str) -> None:
        """Store did not fit / failed: release the held entry back to the free
        pool (blocks were already unpinned-eligible; unpin makes them idle)."""
        entry = self._spill_hold.pop(req_id, None)
        if entry is not None:
            released = entry.get("released_ids", set())
            for block in entry["blocks"]:
                if id(block) in released:
                    continue
                self.block_pool.unpin_block(block)

    def has_spill_hold(self, req_id: str) -> bool:
        return req_id in self._spill_hold

    def confirm_spill_ssd(self, req_id: str) -> dict[str, Any] | None:
        """SSD-mode spill: free GPU blocks and return the chain metadata.

        Unlike ``confirm_spill`` this does not park the session in RAM; the
        caller writes the staging slots to disk and indexes them there.
        """
        entry = self._spill_hold.pop(req_id, None)
        if entry is None:
            return None
        for block in entry["blocks"]:
            self.block_pool.unpin_block(block)
        grp_hashes: list[list[bytes]] = []
        for grp_blocks in entry["grp_blocks"]:
            grp_hashes.append([b.block_hash for b in grp_blocks if not b.is_null])
        return {
            "req_id": req_id,
            "tail": entry["tail"],
            "grp_hashes": grp_hashes,
            "tokens": entry["tokens"],
        }

    def release_spill_blocks(
        self, req_id: str, blocks: list[KVCacheBlock]
    ) -> bool:
        """Free one chunk of a held spill; True when the hold is fully freed.

        Used by chunked SSD spills: GPU blocks are unpinned/freed as soon as
        their chunk has been copied to staging, so admission pressure is
        relieved progressively instead of waiting for the whole session.
        """
        entry = self._spill_hold.get(req_id)
        if entry is None:
            return True
        for block in blocks:
            self.block_pool.unpin_block(block)
        entry.setdefault("released_ids", set()).update(id(b) for b in blocks)
        entry["released"] = entry.get("released", 0) + len(blocks)
        return entry["released"] >= len(entry["blocks"])

    def spill_hold_done(self, req_id: str) -> dict[str, Any] | None:
        """Pop a fully-released spill hold and return its chain metadata."""
        entry = self._spill_hold.pop(req_id, None)
        if entry is None:
            return None
        grp_hashes: list[list[bytes]] = []
        for grp_blocks in entry["grp_blocks"]:
            grp_hashes.append([b.block_hash for b in grp_blocks if not b.is_null])
        return {
            "req_id": req_id,
            "tail": entry["tail"],
            "grp_hashes": grp_hashes,
            "tokens": entry["tokens"],
        }

    def hold_restored_blocks(
        self,
        per_group_hashes: list[list[bytes]],
        per_group_blocks: list[list[KVCacheBlock]],
    ) -> None:
        """Adopt loaded GPU blocks into the prefix cache and pin them.

        Chunked restores hold each chunk until the whole chain is loaded:
        otherwise the idle blocks would be evicted by other requests before
        the resumed request is scheduled.
        """
        for hashes, blocks in zip(per_group_hashes, per_group_blocks):
            for h, b in zip(hashes, blocks):
                self.block_pool._insert_block_hash(h, b, None)
        for blocks in per_group_blocks:
            for b in blocks:
                self.block_pool.pin_block(b)
        if envs.RAMTRACE:
            try:
                with open(
                    envs.RAMTRACE_LOG, "a"
                ) as _f:
                    _f.write(
                        "hold_restored n=%d grp_lens=%s hashes=%s\n"
                        % (
                            sum(len(g) for g in per_group_blocks),
                            [len(g) for g in per_group_blocks],
                            [[repr(h)[:24] for h in grp] for grp in per_group_hashes],
                        )
                    )
            except Exception:
                pass

    def release_restored_hold(self, blocks: list[KVCacheBlock]) -> None:
        """Drop a restored-chunk hold, making the blocks idle-cached again."""
        for b in blocks:
            self.block_pool.free_blocks([b])
            self.block_pool.unpin_block(b)

    def confirm_spill(self, req_id: str, slots: list[int] | None) -> None:
        """Host store finished: unpin/free GPU blocks and park the session."""
        entry = self._spill_hold.pop(req_id, None)
        if entry is None:
            return
        for block in entry["blocks"]:
            self.block_pool.unpin_block(block)
        # Parked record keeps only immutable chain metadata (per-group hashes +
        # counts) and the CPU slot ranges, so a later restore can rebuild the
        # chain in fresh GPU blocks without depending on the recycled originals.
        grp_hashes: list[list[bytes]] = []
        for grp_blocks in entry["grp_blocks"]:
            grp_hashes.append(
                [b.block_hash for b in grp_blocks if not b.is_null]
            )
        self._ram_sessions[req_id] = {
            "req_id": req_id,
            "tail": entry["tail"],
            "grp_hashes": grp_hashes,
            "tokens": entry["tokens"],
            "num_blocks": entry["num_blocks"],
            "slots": slots or [],
            "parked_at": time.monotonic(),
            "last_used": time.monotonic(),
        }
        if envs.RAMTRACE:
            try:
                with open(envs.RAMTRACE_LOG, "a") as _f:
                    _f.write(
                        f"confirm_spill req={req_id} slots={len(slots or [])} "
                        f"grp_lens={[len(g) for g in grp_hashes]} tokens={entry['tokens']}\n"
                    )
            except Exception:
                pass

    def unpark_session(self, req_id: str) -> dict[str, Any] | None:
        return self._ram_sessions.pop(req_id, None)

    def get_ram_sessions(self) -> dict[str, dict[str, Any]]:
        return self._ram_sessions

    def evict_ram_for(self, need: int, protect: str | None = None) -> int:
        """Free host slots by dropping parked sessions in eviction order.

        Small sessions go first, then large ones; within each tier the least
        recently used session (LRU, by ``last_used``) is dropped first.
        ``protect`` (the session being restored, i.e. X) is never dropped.
        Returns the number of sessions dropped.
        """
        dropped = 0
        while self.ram_slots_free() < need:
            candidates = [
                s
                for rid, s in self._ram_sessions.items()
                if rid != protect
            ]
            if not candidates:
                break
            victim = min(
                candidates,
                key=lambda s: evict_sort_key(
                    s["tokens"], s.get("last_used", s["parked_at"])
                ),
            )
            self._ram_sessions.pop(victim["req_id"], None)
            self.free_ram_slots(victim["slots"])
            dropped += 1
            if envs.RAMTRACE:
                try:
                    with open(
                        envs.RAMTRACE_LOG, "a"
                    ) as _f:
                        _f.write(
                            f"ram evict dropped={victim['req_id']} "
                            f"tokens={victim['tokens']} "
                            f"slots={len(victim['slots'])} need={need}\n"
                        )
                except Exception:
                    pass
        return dropped

    def find_ram_session(self, block_hashes: list[bytes]) -> dict[str, Any] | None:
        """Parked session that this request resumes/extends.

        Uses the same tail-hash signal as keep-alive subsumption: if a request
        carries a parked session's tail hash, that session's chain is its
        prefix. Falls back to a strict hash-prefix match.
        """
        if not block_hashes:
            return None
        bh_set = set(block_hashes)
        best: dict[str, Any] | None = None
        best_len = 0
        for sess in self._ram_sessions.values():
            n = max((len(h) for h in sess["grp_hashes"]), default=0)
            if n <= best_len:
                continue
            hit = sess["tail"] is not None and sess["tail"] in bh_set
            if not hit:
                for hashes in sess["grp_hashes"]:
                    if hashes and len(block_hashes) >= len(hashes) and list(
                        block_hashes[: len(hashes)]
                    ) == list(hashes):
                        hit = True
                        break
            if hit:
                best = sess
                best_len = n
        if best is not None:
            # LRU recency: a session that is being probed for restore counts as
            # used, so it is not the first victim of the next eviction.
            best["last_used"] = time.monotonic()
        return best

    def allocate_restore_blocks(
        self, per_group_lens: list[int]
    ) -> list[list[KVCacheBlock]]:
        """Allocate fresh GPU blocks (one run per group) for a restore."""
        total = sum(per_group_lens)
        flat = self.block_pool.get_new_blocks(total)
        out: list[list[KVCacheBlock]] = []
        idx = 0
        for n in per_group_lens:
            out.append(flat[idx : idx + n])
            idx += n
        return out

    def register_restored_blocks(
        self,
        per_group_hashes: list[list[bytes]],
        per_group_blocks: list[list[KVCacheBlock]],
    ) -> None:
        """Make loaded GPU blocks adoptable by the prefix cache.

        Blocks are attached to their original chain hashes and returned to the
        idle cached pool, so a subsequent lookup of the resumed request finds
        the whole prefix on GPU.
        """
        for hashes, blocks in zip(per_group_hashes, per_group_blocks):
            for h, b in zip(hashes, blocks):
                self.block_pool._insert_block_hash(h, b, None)
        for blocks in per_group_blocks:
            if blocks:
                self.block_pool.free_blocks(blocks)
        if envs.RAMTRACE:
            try:
                probe = []
                for hashes in per_group_hashes:
                    if hashes:
                        probe = self.block_pool.get_cached_block([hashes[0]])
                        break
                nums = [
                    [getattr(b, "block_hash_num_tokens", None) for b in blks]
                    for blks in per_group_blocks
                ]
                with open(envs.RAMTRACE_LOG, "a") as _f:
                    _f.write(
                        f"register_restored n={sum(len(b) for b in per_group_blocks)} "
                        f"grp_lens={[len(b) for b in per_group_blocks]} "
                        f"nums={nums} probe_found={len(probe)}\n"
                    )
            except Exception:
                pass

    def remove_skipped_blocks(
        self,
        request_id: str,
        processed_computed_tokens: int,
        num_prompt_tokens: int | None = None,
    ) -> None:
        """Remove the blocks that are no longer needed from `blocks` and replace
        the removed blocks with null_block.

        Args:
            request_id: The request ID.
            processed_computed_tokens: Computed-token prefix length covering
                fully processed and committed tokens only (safe to free).
            num_prompt_tokens: Optional prompt length for R-SWA gap eviction.
        """
        self.coordinator.remove_skipped_blocks(
            request_id, processed_computed_tokens, num_prompt_tokens
        )

    def pop_blocks_for_free(self, request: Request) -> list[KVCacheBlock]:
        """Pop the request's bookkeeping and return its blocks without
        returning them to the block pool. The caller must eventually free
        them in reverse order (so that tail blocks are evicted first).

        Args:
            request: The request to pop the blocks for.

        Returns:
            The request's blocks in allocation order.
        """
        blocks = self.coordinator.pop_blocks_for_free(request.request_id)
        # Pins ride the same (possibly deferred) free as the request blocks.
        # Preemption may release a pin under a still-queued offload — the same
        # exposure normal saves of table blocks already have.
        pins = self._partial_tail_pins.pop(request.request_id, None)
        if pins:
            blocks = pins + blocks
        return blocks

    def evict_blocks(self, block_ids: set[int]) -> None:
        """evict blocks from the prefix cache by their block IDs.

        Args:
            block_ids: Set of block IDs to evict from cache.
        """
        self.block_pool.evict_blocks(block_ids)

    def reset_prefix_cache(self) -> bool:
        """Reset prefix cache. This function may be used in RLHF
        flows to invalidate prefix caching after the weights are updated,
        or used for resetting prefix caching status for benchmarking.

        Returns:
            bool: True if the prefix cache is successfully reset,
            False otherwise.
        """
        if not self.block_pool.reset_prefix_cache():
            return False
        if self.log_stats:
            assert self.prefix_cache_stats is not None
            self.prefix_cache_stats.reset = True
        return True

    def get_num_common_prefix_blocks(self, running_request_id: str) -> list[int]:
        """Calculate the number of common prefix blocks for each kv cache group.

        The function selects a running request and iterates through its blocks.
        A block is considered a common prefix block if ALL requests with
        allocated KV cache share it (i.e., ref_cnt equals the number of entries
        in req_to_blocks).

        NOTE(woosuk): The number of requests with allocated KV cache is **greater
        than or equal to** the number of requests scheduled in the current step.
        This is because having allocated KV cache only indicates that:
        1. The request has not yet finished, and
        2. The request holds its blocks unfreed.

        While all scheduled requests must have allocated KV cache, the inverse
        is not necessarily true. There may be requests with allocated KV cache
        that are not scheduled in the current step.

        This can result in an edge case where the number of common prefix blocks
        is 0, even though all scheduled requests share a common prefix. This
        occurs because there may be unscheduled requests that do not share the
        common prefix. Currently, this case cannot be easily detected, so the
        function returns 0 in such cases.

        Args:
            running_request_id: The request ID of any running request, used to
                identify the common prefix blocks.

        Returns:
            list[int]: The number of common prefix blocks for each kv cache
            group.
        """
        return self.coordinator.get_num_common_prefix_blocks(running_request_id)

    def take_events(self) -> list[KVCacheEvent]:
        """Take the KV cache events from the block pool.

        Returns:
            A list of KV cache events.
        """
        events = self.block_pool.take_events()
        for event in events:
            if not isinstance(event, BlockStored):
                continue
            if event.group_idx is None:
                continue
            if event.group_idx < 0 or event.group_idx >= len(
                self.kv_cache_event_metadata
            ):
                logger.warning(
                    "Group index `%s` not in KV cache metadata", event.group_idx
                )
                continue
            # Annotate here so BlockPool can keep emitting structural cache
            # events without owning semantic KV cache spec metadata.
            kind, sliding_window = self.kv_cache_event_metadata[event.group_idx]
            event.kv_cache_spec_kind = kind
            event.kv_cache_spec_sliding_window = sliding_window
        return events

    def get_blocks(self, request_id: str) -> KVCacheBlocks:
        """Get the blocks of a request."""
        return self.create_kv_cache_blocks(self.coordinator.get_blocks(request_id))

    def get_block_ids(self, request_id: str) -> tuple[list[int], ...]:
        """Get the block ids of a request."""
        return self.get_blocks(request_id).get_block_ids()

    def get_block_ids_for_computed_tokens(
        self,
        request_id: str,
        num_computed_tokens: int,
    ) -> tuple[list[int], ...]:
        """Get block ids covering the request's computed tokens."""
        block_ids = self.get_block_ids(request_id)
        clipped_block_ids: list[list[int]] = []
        for group, ids in zip(self.kv_cache_config.kv_cache_groups, block_ids):
            spec = group.kv_cache_spec
            if not isinstance(spec, AttentionSpec) or isinstance(
                spec, (CrossAttentionSpec, EncoderOnlyAttentionSpec)
            ):
                clipped_block_ids.append(ids)
                continue

            num_valid_blocks = cdiv(num_computed_tokens, spec.block_size)
            clipped_block_ids.append(ids[:num_valid_blocks])
        return tuple(clipped_block_ids)

    def estimate_cached_tokens(self, request: Request) -> int:
        """Estimate the number of tokens cached by the request."""
        cached_tokens: int | None = None
        for group, blocks in zip(
            self.kv_cache_config.kv_cache_groups,
            self.get_blocks(request.request_id).blocks,
        ):
            if isinstance(
                group.kv_cache_spec,
                (CrossAttentionSpec, EncoderOnlyAttentionSpec),
            ):
                # Cross-attention and encoder-only groups are not prefix cached.
                continue

            group_cached_tokens = 0
            for block in blocks:
                group_cached_tokens = max(
                    group_cached_tokens,
                    block.block_hash_num_tokens or 0,
                )

            cached_tokens = (
                group_cached_tokens
                if cached_tokens is None
                else min(cached_tokens, group_cached_tokens)
            )

        return cached_tokens or 0

    def cache_blocks(self, request: Request, num_computed_tokens: int) -> None:
        """Cache the blocks for the request, if enabled.

        Args:
            request: The request to cache the blocks.
            num_computed_tokens: The number of computed tokens, including tokens
                that are already cached and tokens to be cached.
        """
        if self.enable_caching:
            self.coordinator.cache_blocks(request, num_computed_tokens)

    def create_kv_cache_blocks(
        self, blocks: tuple[list[KVCacheBlock], ...]
    ) -> KVCacheBlocks:
        # Only create new KVCacheBlocks for non-empty blocks
        return KVCacheBlocks(blocks) if any(blocks) else self.empty_kv_cache_blocks

    def truncate_computed_blocks(
        self, blocks: KVCacheBlocks, num_computed_tokens: int
    ) -> KVCacheBlocks:
        """Return a lookup-result view truncated at an aligned token endpoint.

        Pure slicing: refcounts are untouched and ``blocks`` is not mutated.
        """
        truncated: list[list[KVCacheBlock]] = []
        for group_blocks, manager in zip(
            blocks.blocks,
            self.coordinator.single_type_managers,
            strict=True,
        ):
            assert num_computed_tokens % manager.block_size == 0
            num_blocks = num_computed_tokens // manager.block_size
            assert num_blocks <= len(group_blocks)
            truncated.append(list(group_blocks[:num_blocks]))
        return self.create_kv_cache_blocks(tuple(truncated))

    def take_new_block_ids(self) -> list[int]:
        """Drain and return new attention block IDs for zeroing."""
        ids: list[int] = []
        for mgr in self.coordinator.single_type_managers:
            ids.extend(mgr.take_new_block_ids())
        return ids

    def get_zeroing_block_ids_in_range(
        self, request_id: str, start_token: int, end_token: int
    ) -> list[int]:
        """The request's block ids covering [start_token, end_token), from
        the groups whose new blocks are zeroed by the worker."""
        ids: list[int] = []
        for mgr in self.coordinator.single_type_managers:
            if mgr.records_new_block_ids:
                start_idx = start_token // mgr.block_size
                end_idx = cdiv(end_token, mgr.block_size)
                blocks = mgr.req_to_blocks[request_id]
                ids.extend(blk.block_id for blk in blocks[start_idx:end_idx])
        return ids

    def record_blocks_for_zeroing(self, request_id: str, start_token: int) -> None:
        """Re-record the request's blocks from start_token onwards for
        zeroing, e.g. blocks a failed async KV load left unwritten.

        start_token must be block-aligned: zeroing a partially-valid block
        would wipe its valid prefix.
        """
        for mgr in self.coordinator.single_type_managers:
            if mgr.records_new_block_ids:
                assert start_token % mgr.block_size == 0
                start_idx = start_token // mgr.block_size
                blocks = mgr.req_to_blocks[request_id]
                mgr.new_block_ids.extend(blk.block_id for blk in blocks[start_idx:])

    def take_kv_cache_block_copies(
        self,
    ) -> tuple[list[KVCacheBlockCopy], list[KVCacheBlock]]:
        """Drain pending copies and return their retained endpoints."""
        pending_copies: list[tuple[KVCacheBlock, KVCacheBlock]] = []
        for mgr in self.coordinator.single_type_managers:
            pending_copies.extend(mgr.take_pending_cow_copies())
        copies = [
            KVCacheBlockCopy(
                src_block_id=source_block.block_id,
                dst_block_id=cow_block.block_id,
            )
            for source_block, cow_block in pending_copies
        ]
        retained_blocks = [block for pair in pending_copies for block in pair]
        return copies, retained_blocks

    def take_partial_tail_offloads(self) -> dict[str, list[tuple[int, int, int]]]:
        """Drain producer partial-tail offload hand-offs per request.

        Returns ``{request_id: [(group_id, block_id, boundary_tokens), ...]}``
        for the durable boundary blocks of producers' last-prompt-boundary
        partial tails. Only mamba "align" groups contribute; empty otherwise.
        A KV connector reads the referenced blocks and offloads them so a later
        request can hit the sub-block prefix.

        Each handed-off block lives off the request block table, so it is
        pinned here and unpinned when the request's blocks are freed — for a
        producer with saved tokens, after the connector reports sends done.
        """
        offloads: dict[str, list[tuple[int, int, int]]] = {}
        for mgr in self.coordinator.single_type_managers:
            for (
                req_id,
                group_id,
                block,
                boundary_tokens,
            ) in mgr.take_pending_partial_tail_offloads():
                self.block_pool.touch((block,))
                self._partial_tail_pins.setdefault(req_id, []).append(block)
                offloads.setdefault(req_id, []).append(
                    (group_id, block.block_id, boundary_tokens)
                )
        return offloads

    def new_step_starts(self) -> None:
        """Notify the coordinator that a new step is starting."""
        self.coordinator.new_step_starts()
