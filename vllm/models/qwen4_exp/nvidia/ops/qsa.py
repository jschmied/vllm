# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton kernels for Qwen4Exp QSA sparse attention and cache updates."""

from __future__ import annotations

import torch

from vllm.model_executor.warmup.jit_warmup_triton_helper import (
    TritonWarmupTensor,
    triton_scalar_specialization_rep,
)
from vllm.triton_utils import HAS_TRITON, tl, triton


@triton.jit(do_not_specialize=["num_rows", "num_requests"])
def _qsa_sparse_paged_gqa_splitk_kernel(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    indices_ptr,
    block_table_ptr,
    token_to_req_ptr,
    partial_output_ptr,
    partial_lse_ptr,
    output_ptr,
    stride_q_row,
    stride_q_head,
    stride_k_block,
    stride_k_token,
    stride_k_head,
    stride_v_block,
    stride_v_token,
    stride_v_head,
    stride_indices_row,
    stride_table_req,
    stride_output_row,
    stride_output_head,
    num_rows,
    num_cache_blocks,
    num_requests,
    TOPK: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    PAGE_TABLE_WIDTH: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NUM_QUERY_HEADS: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    NUM_TILES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    kv_head = tl.program_id(1)
    split_id = tl.program_id(2)
    request = tl.load(token_to_req_ptr + row)
    safe_request = tl.minimum(tl.maximum(request, 0), num_requests - 1)

    # The packed selection buffer carries one TRAILING COUNT COLUMN per row
    # (column TOPK of a TOPK+1-wide buffer): the row's valid-entry count,
    # written by the expand kernel. It is never a token index — the tile loop
    # and the index load below only ever cover columns [0, TOPK).
    valid_count = tl.load(indices_ptr + row * stride_indices_row + TOPK)

    head_offsets = tl.arange(0, BLOCK_M)
    dim_offsets = tl.arange(0, HEAD_DIM)
    column_offsets = tl.arange(0, BLOCK_N)
    first_head = kv_head * GROUP_SIZE
    query = tl.load(
        q_ptr
        + row * stride_q_row
        + (first_head + head_offsets[:, None]) * stride_q_head
        + dim_offsets[None, :],
        mask=head_offsets[:, None] < GROUP_SIZE,
        other=0.0,
    )

    max_value = tl.full((BLOCK_M,), -1.0e20, dtype=tl.float32)
    normalizer = tl.zeros((BLOCK_M,), dtype=tl.float32)
    accumulator = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)
    softmax_scale_log2: tl.constexpr = (HEAD_DIM**-0.5) * 1.4426950408889634

    tile_end = tl.minimum(NUM_TILES, tl.cdiv(tl.minimum(valid_count, TOPK), BLOCK_N))

    for tile in range(split_id, tile_end, NUM_SPLITS):
        columns = tile * BLOCK_N + column_offsets
        logical_token = tl.load(
            indices_ptr + row * stride_indices_row + columns,
            mask=columns < TOPK,
            other=-1,
        )
        safe_token = tl.maximum(logical_token, 0)
        logical_page = safe_token // PAGE_SIZE
        page_offset = safe_token % PAGE_SIZE
        valid = (
            (request >= 0)
            & (request < num_requests)
            & (logical_token >= 0)
            & (logical_page < PAGE_TABLE_WIDTH)
        )
        physical_page = tl.load(
            block_table_ptr
            + safe_request * stride_table_req
            + tl.minimum(logical_page, PAGE_TABLE_WIDTH - 1),
            mask=valid,
            other=-1,
        )
        valid &= (physical_page >= 0) & (physical_page < num_cache_blocks)
        # physical_page * block stride can overflow int32 for large caches.
        safe_page = tl.maximum(physical_page, 0).to(tl.int64)
        keys = tl.load(
            k_cache_ptr
            + safe_page[None, :] * stride_k_block
            + page_offset[None, :] * stride_k_token
            + kv_head * stride_k_head
            + dim_offsets[:, None],
            mask=valid[None, :],
            other=0.0,
        )
        values = tl.load(
            v_cache_ptr
            + safe_page[:, None] * stride_v_block
            + page_offset[:, None] * stride_v_token
            + kv_head * stride_v_head
            + dim_offsets[None, :],
            mask=valid[:, None],
            other=0.0,
        )
        scores = tl.dot(query, keys)
        # Scaling scores avoids re-quantizing a scaled query to BF16.
        scores *= softmax_scale_log2
        scores = tl.where(valid[None, :], scores, -1.0e20)
        next_max = tl.maximum(max_value, tl.max(scores, axis=1))
        alpha = tl.math.exp2(max_value - next_max)
        probabilities = tl.where(
            valid[None, :], tl.math.exp2(scores - next_max[:, None]), 0.0
        )
        accumulator = tl.dot(
            probabilities.to(values.dtype),
            values,
            acc=accumulator * alpha[:, None],
        )
        normalizer = normalizer * alpha + tl.sum(probabilities, axis=1)
        max_value = next_max

    has_values = normalizer > 0
    normalized_output = tl.where(
        has_values[:, None],
        accumulator / tl.maximum(normalizer[:, None], 1.0e-20),
        0.0,
    )
    output_mask = head_offsets[:, None] < GROUP_SIZE
    if NUM_SPLITS == 1:
        tl.store(
            output_ptr
            + row * stride_output_row
            + (first_head + head_offsets[:, None]) * stride_output_head
            + dim_offsets[None, :],
            normalized_output,
            mask=output_mask,
        )
    else:
        partial_lse = tl.where(
            has_values,
            max_value + tl.math.log2(tl.maximum(normalizer, 1.0e-20)),
            -float("inf"),
        )
        partial_row = (split_id * num_rows + row).to(tl.int64)
        tl.store(
            partial_output_ptr
            + (partial_row * NUM_QUERY_HEADS + first_head + head_offsets[:, None])
            * HEAD_DIM
            + dim_offsets[None, :],
            normalized_output,
            mask=output_mask,
        )
        tl.store(
            partial_lse_ptr + partial_row * NUM_QUERY_HEADS + first_head + head_offsets,
            partial_lse,
            mask=head_offsets < GROUP_SIZE,
        )


@triton.jit(do_not_specialize=["num_rows"])
def _qsa_merge_splitk_kernel(
    partial_output_ptr,
    partial_lse_ptr,
    output_ptr,
    stride_output_row,
    stride_output_head,
    num_rows,
    HEAD_DIM: tl.constexpr,
    NUM_QUERY_HEADS: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_SPLITS: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    head = tl.program_id(1)
    split_offsets = tl.arange(0, BLOCK_SPLITS)
    dim_offsets = tl.arange(0, HEAD_DIM)
    split_mask = split_offsets < NUM_SPLITS
    lse = tl.load(
        partial_lse_ptr + (split_offsets * num_rows + row) * NUM_QUERY_HEADS + head,
        mask=split_mask,
        other=-float("inf"),
    )
    lse_max = tl.max(lse, axis=0)
    has_values = lse_max > -float("inf")
    shifted = tl.where(split_mask & has_values, lse - lse_max, -float("inf"))
    weights = tl.math.exp2(shifted)
    denominator = tl.sum(weights, axis=0)
    split_rows = split_offsets.to(tl.int64) * num_rows + row
    partial_output = tl.load(
        partial_output_ptr
        + (split_rows[:, None] * NUM_QUERY_HEADS + head) * HEAD_DIM
        + dim_offsets[None, :],
        mask=split_mask[:, None],
        other=0.0,
    )
    merged = tl.sum(partial_output * weights[:, None], axis=0)
    merged = tl.where(denominator > 0, merged / denominator, 0.0)
    tl.store(
        output_ptr + row * stride_output_row + head * stride_output_head + dim_offsets,
        merged,
    )


@triton.jit
def _store_qsa_rows_kernel(
    cache_ptr,
    slots_ptr,
    rows_ptr,
    stride_cache_block,
    stride_cache_token,
    stride_cache_dim,
    stride_rows_row,
    stride_rows_dim,
    num_rows,
    num_blocks,
    PAGE_SIZE: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK_D: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    dims = tl.arange(0, BLOCK_D)
    slot = tl.load(slots_ptr + row)
    valid = (row < num_rows) & (slot >= 0) & (slot < num_blocks * PAGE_SIZE)
    block = tl.maximum(slot, 0) // PAGE_SIZE
    token = tl.maximum(slot, 0) % PAGE_SIZE
    values = tl.load(
        rows_ptr + row * stride_rows_row + dims * stride_rows_dim,
        mask=valid & (dims < WIDTH),
        other=0,
    )
    tl.store(
        cache_ptr
        + block * stride_cache_block
        + token * stride_cache_token
        + dims * stride_cache_dim,
        values,
        mask=valid & (dims < WIDTH),
    )


@triton.jit
def _compress_qsa_groups_kernel(
    raw_keys_ptr,  # this step's raw key rows, straight from activations
    raw_positions_ptr,  # this step's per-token positions
    compressor_state_cache_ptr,  # per-request ring of previous raw keys
    rope_cache_ptr,  # packed RoPE position tail of the ring
    compressor_state_table_ptr,
    token_to_req_ptr,
    query_start_loc_ptr,
    logical_positions_ptr,
    compressed_slots_ptr,
    pooled_ptr,
    first_positions_ptr,
    stride_raw_row,
    stride_raw_dim,
    stride_raw_positions_row,
    stride_raw_positions_dim,
    stride_compressor_state_block,
    stride_compressor_state_token,
    stride_compressor_state_dim,
    stride_rope_block,
    stride_rope_token,
    stride_rope_dim,
    stride_compressor_state_table_req,
    stride_pooled_row,
    stride_pooled_dim,
    stride_positions_row,
    stride_positions_dim,
    num_rows,
    num_compressor_state_blocks,
    num_requests,
    COMPRESSOR_STATE_SIZE: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    LOAD_ROPE_POSITIONS: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    dims = tl.arange(0, BLOCK_D)
    request = tl.load(token_to_req_ptr + row)
    end_position = tl.load(logical_positions_ptr + row)
    compressed_slot = tl.load(compressed_slots_ptr + row)
    valid_request = (request >= 0) & (request < num_requests)
    safe_request = tl.minimum(tl.maximum(request, 0), num_requests - 1)
    query_row_start = tl.load(
        query_start_loc_ptr + safe_request, mask=valid_request, other=0
    )
    query_row_end = tl.load(
        query_start_loc_ptr + safe_request + 1, mask=valid_request, other=0
    )
    chunk_start_position = end_position - (row - query_row_start)
    compressor_state_block = tl.load(
        compressor_state_table_ptr + safe_request * stride_compressor_state_table_req,
        mask=valid_request,
        other=-1,
    )
    valid_compressor_state_block = (compressor_state_block >= 0) & (
        compressor_state_block < num_compressor_state_blocks
    )
    valid_row = (
        (row < num_rows)
        & valid_request
        & (row >= query_row_start)
        & (row < query_row_end)
        & (end_position >= COMPRESS_RATIO - 1)
        & (compressed_slot >= 0)
    )
    accumulator = tl.zeros((BLOCK_D,), dtype=tl.float32)

    # A group can span the compressor-state ring (older members) and this
    # step's raw rows (members at positions >= chunk_start_position).
    for group_offset in tl.range(0, COMPRESS_RATIO):
        position = end_position - (COMPRESS_RATIO - 1 - group_offset)
        use_raw = position >= chunk_start_position
        raw_row = query_row_start + position - chunk_start_position
        raw_values = tl.load(
            raw_keys_ptr + raw_row * stride_raw_row + dims * stride_raw_dim,
            mask=valid_row
            & use_raw
            & (raw_row >= query_row_start)
            & (raw_row < query_row_end)
            & (raw_row < num_rows)
            & (dims < HEAD_DIM),
            other=0.0,
        ).to(tl.float32)
        compressor_state_values = tl.load(
            compressor_state_cache_ptr
            + tl.maximum(compressor_state_block, 0).to(tl.int64)
            * stride_compressor_state_block
            + (position % COMPRESSOR_STATE_SIZE) * stride_compressor_state_token
            + dims * stride_compressor_state_dim,
            mask=valid_row
            & ~use_raw
            & valid_compressor_state_block
            & (dims < HEAD_DIM),
            other=0.0,
        ).to(tl.float32)
        accumulator += tl.where(use_raw, raw_values, compressor_state_values)

    tl.store(
        pooled_ptr + row * stride_pooled_row + dims * stride_pooled_dim,
        accumulator / COMPRESS_RATIO,
        mask=(row < num_rows) & (dims < HEAD_DIM),
    )

    position_dims = tl.arange(0, 4)
    first_position = end_position - COMPRESS_RATIO + 1
    if LOAD_ROPE_POSITIONS:
        first_from_raw = first_position >= chunk_start_position
        raw_first_row = query_row_start + first_position - chunk_start_position
        raw_position_values = tl.load(
            raw_positions_ptr
            + raw_first_row * stride_raw_positions_row
            + position_dims * stride_raw_positions_dim,
            mask=valid_row
            & first_from_raw
            & (raw_first_row >= query_row_start)
            & (raw_first_row < query_row_end)
            & (raw_first_row < num_rows)
            & (position_dims < 3),
            other=0,
        )
        compressor_state_position_values = tl.load(
            rope_cache_ptr
            + tl.maximum(compressor_state_block, 0).to(tl.int64) * stride_rope_block
            + (first_position % COMPRESSOR_STATE_SIZE) * stride_rope_token
            + position_dims * stride_rope_dim,
            mask=valid_row
            & ~first_from_raw
            & valid_compressor_state_block
            & (position_dims < 3),
            other=0,
        )
        position_values = tl.where(
            first_from_raw,
            raw_position_values,
            compressor_state_position_values,
        )
    else:
        position_values = tl.where(valid_row, first_position, 0)
    tl.store(
        first_positions_ptr
        + row * stride_positions_row
        + position_dims * stride_positions_dim,
        position_values,
        mask=(row < num_rows) & (position_dims < 3),
    )


def _select_config(
    num_rows: int, num_kv_heads: int, use_prefill_config: bool, num_columns: int
) -> tuple[int, int, int, int]:
    """Select (block_n, num_warps, num_tiles, num_splits) for the kernel.

    Tuned on GB300 for the Qwen3.8-Flash-Next TP1/TP2/TP4 shapes, keyed on
    base_programs = num_rows * num_kv_heads. The bp > 2048 region splits on
    use_prefill_config (capture-stable: at FULL-graph capture max_query_len is the
    uniform decode/verify length).
    """
    base_programs = num_rows * num_kv_heads
    if base_programs > 2048:
        BLOCK_N, target_splits, num_warps = (
            (32, 1, 1) if use_prefill_config else (64, 1, 2)
        )
    elif base_programs <= 24:
        BLOCK_N, target_splits, num_warps = 32, 64, 4
    elif base_programs <= 32:
        BLOCK_N, target_splits, num_warps = 32, 16, 1
    elif base_programs <= 64:
        BLOCK_N, target_splits, num_warps = 32, 8, 1
    elif base_programs <= 128:
        BLOCK_N, target_splits, num_warps = 32, 4, 1
    elif base_programs <= 256:
        BLOCK_N, target_splits, num_warps = 32, 8, 1
    elif base_programs <= 512:
        BLOCK_N, target_splits, num_warps = 64, 4, 2
    else:
        BLOCK_N, target_splits, num_warps = 64, 1, 2
    num_tiles = triton.cdiv(num_columns, BLOCK_N)
    # Never more splits than tiles, never empty.
    num_splits = min(target_splits, num_tiles)
    return BLOCK_N, num_warps, num_tiles, num_splits


def qsa_sparse_paged_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    logical_indices: torch.Tensor,
    block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    use_prefill_config: bool,
    out: torch.Tensor | None = None,
    union: dict | None = None,  # QSA UNION: {compress_ratio, token_topk, num_requests, raw} or None
) -> torch.Tensor:
    """Run sparse GQA directly over paged BF16 K/V caches.

    logical_indices is the PACKED selection buffer: [rows, selection_width + 1]
    with the trailing column holding each row's valid-entry count (written by
    the expand kernel; never a token index). The kernel reads it as the
    tile-loop bound. use_prefill_config only steers the top of the config table; see
    _select_config.
    """
    if q.ndim != 3 or k_cache.ndim != 4 or v_cache.shape != k_cache.shape:
        raise ValueError("QSA sparse attention received invalid Q/K/V shapes")
    if logical_indices.ndim != 2 or logical_indices.shape[0] != q.shape[0]:
        raise ValueError("QSA indices must have one row per query")
    if token_to_req.shape != (q.shape[0],) or block_table.ndim != 2:
        raise ValueError("QSA sparse attention metadata has invalid shapes")
    if not all(k_cache.shape[:3]) or not all(block_table.shape):
        raise ValueError("QSA sparse attention cache and block table must be nonempty")
    if logical_indices.shape[1] < 2:
        raise ValueError(
            "QSA packed indices need selection columns plus the count column"
        )
    if q.shape[2] != k_cache.shape[3] or q.shape[1] % k_cache.shape[2]:
        raise ValueError("QSA sparse attention requires valid grouped-query heads")
    head_dim = q.shape[2]
    assert head_dim >= 16 and (head_dim & (head_dim - 1)) == 0
    assert q.dtype == k_cache.dtype == v_cache.dtype == torch.bfloat16
    assert logical_indices.dtype == block_table.dtype == torch.int32
    assert token_to_req.dtype == torch.int32
    assert q.device == k_cache.device == v_cache.device
    assert q.device == logical_indices.device == block_table.device
    assert q.device == token_to_req.device
    assert q.stride(2) == k_cache.stride(3) == v_cache.stride(3) == 1
    assert logical_indices.stride(1) == block_table.stride(1) == 1
    assert token_to_req.stride(0) == 1

    if out is None:
        out = torch.empty_like(q)
    if out.shape != q.shape:
        raise ValueError("QSA sparse output must match its query")
    assert out.dtype == q.dtype and out.device == q.device
    assert out.stride(2) == 1
    if not q.shape[0]:
        return out
    if union is not None and not qsa_union_layout_ok(k_cache, v_cache, union["compress_ratio"]):  # QSA UNION gate
        _qsa_union_log_once("stock fallback (page size not a multiple of the compress ratio)")
    elif union is not None:
        return qsa_sparse_paged_attention_union(q, k_cache, v_cache, logical_indices, block_table, token_to_req, out,
                                                compress_ratio=union["compress_ratio"], token_topk=union["token_topk"],
                                                num_requests=union["num_requests"], raw=union.get("raw"))

    group_size = q.shape[1] // k_cache.shape[2]
    block_m = triton.next_power_of_2(group_size)
    selection_width = logical_indices.shape[1] - 1  # trailing column is the count
    block_n, partial_warps, num_tiles, num_splits = _select_config(
        q.shape[0], k_cache.shape[2], use_prefill_config, selection_width
    )

    # Split=1 writes output directly and compiles out all workspace accesses.
    if num_splits == 1:
        partial_output = out
        partial_lse = out
    else:
        # FP32 partials preserve accuracy when merging independently normalized
        # splits.
        partial_output = torch.empty(
            (num_splits, *q.shape), dtype=torch.float32, device=q.device
        )
        partial_lse = torch.empty(
            (num_splits, q.shape[0], q.shape[1]),
            dtype=torch.float32,
            device=q.device,
        )

    partial_grid = (q.shape[0], k_cache.shape[2], num_splits)
    _qsa_sparse_paged_gqa_splitk_kernel[partial_grid](
        q,
        k_cache,
        v_cache,
        logical_indices,
        block_table,
        token_to_req,
        partial_output,
        partial_lse,
        out,
        q.stride(0),
        q.stride(1),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        v_cache.stride(0),
        v_cache.stride(1),
        v_cache.stride(2),
        logical_indices.stride(0),
        block_table.stride(0),
        out.stride(0),
        out.stride(1),
        q.shape[0],
        k_cache.shape[0],
        block_table.shape[0],
        TOPK=selection_width,
        PAGE_SIZE=k_cache.shape[1],
        PAGE_TABLE_WIDTH=block_table.shape[1],
        GROUP_SIZE=group_size,
        HEAD_DIM=q.shape[2],
        NUM_QUERY_HEADS=q.shape[1],
        NUM_SPLITS=num_splits,
        NUM_TILES=num_tiles,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=partial_warps,
        num_stages=2,
    )
    if num_splits == 1:
        return out

    _qsa_merge_splitk_kernel[(q.shape[0], q.shape[1])](
        partial_output,
        partial_lse,
        out,
        out.stride(0),
        out.stride(1),
        q.shape[0],
        HEAD_DIM=q.shape[2],
        NUM_QUERY_HEADS=q.shape[1],
        NUM_SPLITS=num_splits,
        BLOCK_SPLITS=triton.next_power_of_2(num_splits),
        num_warps=2,
        num_stages=1,
    )
    return out


def warmup_qsa_sparse_paged_attention(
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    *,
    num_query_heads: int,
    selection_width: int,
) -> tuple[tuple[int, int, int], ...]:
    """Compile every production-reachable split-K/merge specialization."""

    head_dim = kv_cache.shape[-1] // 2
    key_cache, value_cache = kv_cache.transpose(1, 2).split(head_dim, dim=-1)
    num_kv_heads = key_cache.shape[2]
    group_size = num_query_heads // num_kv_heads
    block_m = triton.next_power_of_2(group_size)

    # Every config the dispatch can pick for this group size.
    profiles = {
        _select_config(num_rows, num_kv_heads, use_prefill_config, selection_width)
        for num_rows in range(1, 8193)
        for use_prefill_config in (False, True)
    }

    # Scalars constant per deployment get their real values (their divisibility
    # specialization is wanted); the batch-varying ones are do_not_specialize'd
    # on the kernels, so any value here compiles the only variant.
    num_rows = 16
    num_requests = 16
    q_ptr = TritonWarmupTensor(
        torch.bfloat16, shape=(num_rows, num_query_heads, head_dim)
    )
    k_cache_ptr = TritonWarmupTensor(
        key_cache.dtype,
        shape=tuple(key_cache.shape),
        strides=tuple(key_cache.stride()),
    )
    v_cache_ptr = TritonWarmupTensor(
        value_cache.dtype,
        shape=tuple(value_cache.shape),
        strides=tuple(value_cache.stride()),
    )
    # +1: the packed buffer's trailing count column.
    indices_ptr = TritonWarmupTensor(torch.int32, shape=(num_rows, selection_width + 1))
    block_table_ptr = TritonWarmupTensor(
        block_table.dtype,
        shape=tuple(block_table.shape),
        strides=tuple(block_table.stride()),
    )
    token_to_req_ptr = TritonWarmupTensor(torch.int32)
    output_ptr = TritonWarmupTensor(
        torch.bfloat16, shape=(num_rows, num_query_heads, head_dim)
    )
    head_stride = head_dim
    row_stride = num_query_heads * head_dim
    num_cache_blocks = triton_scalar_specialization_rep(kv_cache.shape[0])

    warmed = []
    for block_n, warps, num_tiles, num_splits in sorted(profiles):
        if num_splits == 1:
            partial_output_ptr = output_ptr
            partial_lse_ptr = output_ptr
        else:
            partial_output_ptr = TritonWarmupTensor(
                torch.float32,
                shape=(num_splits, num_rows, num_query_heads, head_dim),
            )
            partial_lse_ptr = TritonWarmupTensor(
                torch.float32, shape=(num_splits, num_rows, num_query_heads)
            )
        _qsa_sparse_paged_gqa_splitk_kernel.warmup(
            q_ptr,
            k_cache_ptr,
            v_cache_ptr,
            indices_ptr,
            block_table_ptr,
            token_to_req_ptr,
            partial_output_ptr,
            partial_lse_ptr,
            output_ptr,
            row_stride,
            head_stride,
            key_cache.stride(0),
            key_cache.stride(1),
            key_cache.stride(2),
            value_cache.stride(0),
            value_cache.stride(1),
            value_cache.stride(2),
            selection_width + 1,
            block_table.stride(0),
            row_stride,
            head_stride,
            num_rows,
            num_cache_blocks,
            num_requests,
            TOPK=selection_width,
            PAGE_SIZE=key_cache.shape[1],
            PAGE_TABLE_WIDTH=block_table.shape[1],
            GROUP_SIZE=group_size,
            HEAD_DIM=head_dim,
            NUM_QUERY_HEADS=num_query_heads,
            NUM_SPLITS=num_splits,
            NUM_TILES=num_tiles,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            num_warps=warps,
            num_stages=2,
            grid=(num_rows, num_kv_heads, num_splits),
        )
        if num_splits > 1:
            _qsa_merge_splitk_kernel.warmup(
                partial_output_ptr,
                partial_lse_ptr,
                output_ptr,
                row_stride,
                head_stride,
                num_rows,
                HEAD_DIM=head_dim,
                NUM_QUERY_HEADS=num_query_heads,
                NUM_SPLITS=num_splits,
                BLOCK_SPLITS=triton.next_power_of_2(num_splits),
                num_warps=2,
                num_stages=1,
                grid=(num_rows, num_query_heads),
            )
        warmed.append((block_n, num_splits, warps))
    return tuple(warmed)


def qsa_store_cache_rows(
    cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    rows: torch.Tensor,
) -> None:
    """Store fixed-width rows in a QSA cache without boolean indexing."""

    if not cache.is_cuda or not HAS_TRITON:
        raise RuntimeError("QSA CUDA cache stores require Triton")
    if cache.ndim != 4 or cache.shape[2] != 1:
        raise ValueError("QSA cache must be [pages, page_size, 1, width]")
    if not all(cache.shape):
        raise ValueError("QSA cache dimensions must be nonzero")
    if rows.ndim == 3:
        if rows.shape[1] != 1:
            raise ValueError("QSA cache rows must have one head")
        rows = rows[:, 0]
    if rows.shape != (slot_mapping.numel(), cache.shape[3]):
        raise ValueError("QSA cache rows and slots have incompatible shapes")
    if not rows.shape[0]:
        return
    _store_qsa_rows_kernel[(rows.shape[0],)](
        cache,
        slot_mapping,
        rows,
        cache.stride(0),
        cache.stride(1),
        cache.stride(3),
        rows.stride(0),
        rows.stride(1),
        rows.shape[0],
        cache.shape[0],
        PAGE_SIZE=cache.shape[1],
        WIDTH=cache.shape[3],
        BLOCK_D=triton.next_power_of_2(cache.shape[3]),
        num_warps=4,
    )


def qsa_compress_groups_with_ratio(
    raw_keys: torch.Tensor,  # this step's raw key rows [rows, 1, head_size]
    raw_positions: torch.Tensor,  # this step's positions [rows, 1, 3] int64
    compressor_state_cache: torch.Tensor,
    compressor_state_block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    query_start_loc: torch.Tensor,
    logical_positions: torch.Tensor,
    compressed_slots: torch.Tensor,
    compress_ratio: int,
    rope_cache: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pool completed groups from the compressor-state ring and raw token rows."""

    if not raw_keys.is_cuda or not HAS_TRITON:
        raise RuntimeError("QSA CUDA compression requires Triton")
    rows = token_to_req.numel()
    if compress_ratio <= 0:
        raise ValueError("QSA compression ratio must be positive")
    if raw_keys.ndim != 3 or raw_keys.shape[:2] != (rows, 1):
        raise ValueError("QSA raw keys must be [rows, 1, head_size]")
    if raw_positions.shape != (rows, 1, 3) or raw_positions.dtype != torch.int64:
        raise ValueError("QSA raw positions must be [rows, 1, 3] int64")
    if logical_positions.shape != (rows,) or compressed_slots.shape != (rows,):
        raise ValueError("QSA compression metadata must match token rows")
    if compressor_state_cache.ndim != 4 or compressor_state_cache.shape[2] != 1:
        raise ValueError("QSA compressor-state cache has an invalid shape")
    if (
        # The ring is wider than one group so speculative rows cannot alias
        # onto the committed keys of the group still being collected.
        compressor_state_cache.shape[1] < compress_ratio
        or compressor_state_cache.shape[3] != raw_keys.shape[2]
        or compressor_state_cache.dtype != raw_keys.dtype
    ):
        raise ValueError(
            "QSA compressor-state cache does not match the compression layout"
        )
    if (
        compressor_state_block_table.ndim != 2
        or compressor_state_block_table.shape[1] < 1
    ):
        raise ValueError(
            "QSA compressor-state block table must contain one block per request"
        )
    if query_start_loc.ndim != 1 or query_start_loc.shape[0] < 2:
        raise ValueError("QSA query starts must contain a terminal offset")
    num_requests = query_start_loc.shape[0] - 1
    if compressor_state_block_table.shape[0] < num_requests:
        raise ValueError("QSA compressor-state block table has too few request rows")
    if rope_cache is not None and (
        rope_cache.ndim != 4
        or rope_cache.shape[:3] != compressor_state_cache.shape[:3]
        or rope_cache.shape[3] != 3
        or rope_cache.dtype != torch.int64
    ):
        raise ValueError("QSA packed position view has an invalid shape or dtype")
    if rows and (
        not all(compressor_state_cache.shape)
        or not all(compressor_state_block_table.shape)
    ):
        raise ValueError("QSA compressor-state cache and block table must be nonempty")
    pooled = torch.empty(
        (rows, 1, raw_keys.shape[2]),
        dtype=raw_keys.dtype,
        device=raw_keys.device,
    )
    first_positions = torch.empty((rows, 3), dtype=torch.int64, device=raw_keys.device)
    if not rows:
        return pooled, first_positions
    if rope_cache is None:
        rope_cache = compressor_state_cache
        load_rope_positions = False
    else:
        load_rope_positions = True
    _compress_qsa_groups_kernel[(rows,)](
        raw_keys,
        raw_positions,
        compressor_state_cache,
        rope_cache,
        compressor_state_block_table,
        token_to_req,
        query_start_loc,
        logical_positions,
        compressed_slots,
        pooled,
        first_positions,
        raw_keys.stride(0),
        raw_keys.stride(2),
        raw_positions.stride(0),
        raw_positions.stride(2),
        compressor_state_cache.stride(0),
        compressor_state_cache.stride(1),
        compressor_state_cache.stride(3),
        rope_cache.stride(0),
        rope_cache.stride(1),
        rope_cache.stride(3),
        compressor_state_block_table.stride(0),
        pooled.stride(0),
        pooled.stride(2),
        first_positions.stride(0),
        first_positions.stride(1),
        rows,
        compressor_state_cache.shape[0],
        num_requests,
        COMPRESSOR_STATE_SIZE=compressor_state_cache.shape[1],
        COMPRESS_RATIO=compress_ratio,
        HEAD_DIM=raw_keys.shape[2],
        LOAD_ROPE_POSITIONS=load_rope_positions,
        BLOCK_D=triton.next_power_of_2(raw_keys.shape[2]),
        num_warps=4,
    )
    return pooled, first_positions


__all__ = [
    "qsa_compress_groups_with_ratio",
    "qsa_sparse_paged_attention",
    "qsa_store_cache_rows",
    "warmup_qsa_sparse_paged_attention",
]


# ---- QSA UNION (jschmied 2026-09-04) ----
import os as _os
import torch
import triton
import triton.language as tl

_QSA_UNION = _os.environ.get("VLLM_QSA_UNION", "0") not in ("0", "", "false", "False")
_QSA_UNION_MIN_ROWS = int(_os.environ.get("VLLM_QSA_UNION_MIN_ROWS", "1024"))
_QSA_UNION_R = int(_os.environ.get("VLLM_QSA_UNION_R", "2"))        # finding 111: R=2, BN=32, 4 warps at every context
_QSA_UNION_BNB = int(_os.environ.get("VLLM_QSA_UNION_BNB", "8"))
_QSA_UNION_WARPS = int(_os.environ.get("VLLM_QSA_UNION_WARPS", "4"))
_QSA_UNION_TAIL_COLS = 16
if _QSA_UNION:
    print("QSAUNION active", flush=True)


@triton.jit
def _qsa_union_build_kernel(sorted_ptr, uni_ptr, mem_ptr, cnt_ptr, token_to_req_ptr, block_table_ptr, qpos_ptr, tail_ptr,
                            stride_sorted, stride_uni, stride_mem_t, stride_mem_r, stride_table_req, stride_tail,
                            num_rows, num_requests, table_width, num_cache_blocks,
                            N: tl.constexpr, R: tl.constexpr, CR: tl.constexpr, PAGE_SIZE: tl.constexpr,
                            TAIL_COLS: tl.constexpr, WRITE_TAILS: tl.constexpr):
    # sorted_ptr[t]: ascending packed (block*8 + row_in_tile), BIG*8+7 for padding; exactly N = R*block_topk.
    # Writes each union block as its physical token base (page * PAGE_SIZE + offset of the block's first token,
    # -1 if the page is invalid), so the attention loop addresses K/V without a page-table gather (finding 111).
    BPP: tl.constexpr = PAGE_SIZE // CR
    t = tl.program_id(0)
    i = tl.arange(0, N)
    packed = tl.load(sorted_ptr + t * stride_sorted + i)
    prev = tl.load(sorted_ptr + t * stride_sorted + i - 1, mask=i > 0, other=-8)
    blk = packed // 8
    r = packed % 8
    valid = blk < (1 << 27)
    first = (blk != prev // 8) & valid
    pos = tl.cumsum(first.to(tl.int32)) - 1
    # the tile's block-table row: any valid row of the tile (same rule as the attention kernel)
    rr = tl.arange(0, R)
    row = t * R + rr
    request = tl.load(token_to_req_ptr + tl.minimum(row, num_rows - 1), mask=row < num_rows, other=-1)
    req_ok = (request >= 0) & (request < num_requests)
    tile_request = tl.minimum(tl.max(tl.where(req_ok, request, 0), axis=0), num_requests - 1)
    logical_page = blk // BPP
    page_ok = valid & (logical_page < table_width)
    physical_page = tl.load(block_table_ptr + tile_request * stride_table_req + tl.minimum(logical_page, table_width - 1),
                            mask=page_ok, other=-1)
    page_ok &= (physical_page >= 0) & (physical_page < num_cache_blocks)
    phys = tl.where(page_ok, physical_page * PAGE_SIZE + (blk % BPP) * CR, -1)
    tl.store(uni_ptr + t * stride_uni + pos, phys, mask=first)
    if WRITE_TAILS:
        # the expansion kernel's rule: tail_start = ((q + 1) // CR) * CR, tail_count = q + 1 - tail_start (< CR)
        tt = tl.arange(0, TAIL_COLS)
        r_t = tt // (CR - 1)
        j_t = tt % (CR - 1)
        trow = t * R + r_t
        tmask = (r_t < R) & (trow < num_rows)
        qp = tl.load(qpos_ptr + tl.minimum(trow, num_rows - 1), mask=tmask, other=-1)
        tail_start = ((qp + 1) // CR) * CR
        tail_count = qp + 1 - tail_start
        tail_tok = tl.where(tmask & (j_t < tail_count) & (qp >= 0), tail_start + j_t, -1)
        tl.store(tail_ptr + t * stride_tail + tt, tail_tok)
    tl.store(mem_ptr + t * stride_mem_t + r * stride_mem_r + pos, tl.full((N,), 1, tl.int8), mask=valid)
    tl.store(cnt_ptr + t, tl.sum(first.to(tl.int32)))


@triton.jit
def _qsa_union_attn_kernel(q_ptr, k_cache_ptr, v_cache_ptr, uni_ptr, mem_ptr, cnt_ptr, tail_ptr, block_table_ptr,
                           token_to_req_ptr, out_ptr,
                           stride_q_row, stride_q_head, stride_k_block, stride_k_token, stride_k_head,
                           stride_v_block, stride_v_token, stride_v_head, stride_uni, stride_mem_t, stride_mem_r,
                           stride_tail, stride_table_req, stride_out_row, stride_out_head,
                           num_rows, num_requests, num_cache_blocks,
                           R: tl.constexpr, GP: tl.constexpr, GROUP_SIZE: tl.constexpr, HEAD_DIM: tl.constexpr,
                           BNB: tl.constexpr, CR: tl.constexpr, TAIL_COLS: tl.constexpr, PAGE_SIZE: tl.constexpr,
                           PAGE_TABLE_WIDTH: tl.constexpr):
    tile = tl.program_id(0)
    kv_head = tl.program_id(1)
    M: tl.constexpr = R * GP
    BN: tl.constexpr = BNB * CR
    TAIL_PER_ROW: tl.constexpr = CR - 1
    m_off = tl.arange(0, M)
    r_of_m = m_off // GP
    h_of_m = m_off % GP
    dim_offsets = tl.arange(0, HEAD_DIM)
    b_off = tl.arange(0, BNB)
    j_off = tl.arange(0, CR)
    row = tile * R + r_of_m
    # stock contract: rows with an invalid request are masked, the block-table row is clamped
    request = tl.load(token_to_req_ptr + tl.minimum(row, num_rows - 1), mask=row < num_rows, other=-1)
    req_ok = (request >= 0) & (request < num_requests)
    safe_request = tl.minimum(tl.maximum(request, 0), num_requests - 1)
    # the tile's block-table row: any valid row of the tile (the owner guarantees one request per batch; rows with
    # an invalid id, e.g. padding, are masked above and must not pick the table row)
    tile_request = tl.minimum(tl.max(tl.where(req_ok, request, 0), axis=0), num_requests - 1)
    qmask = (row < num_rows) & (h_of_m < GROUP_SIZE) & req_ok
    first_head = kv_head * GROUP_SIZE
    query = tl.load(q_ptr + row[:, None] * stride_q_row + (first_head + h_of_m[:, None]) * stride_q_head
                    + dim_offsets[None, :], mask=qmask[:, None], other=0.0)
    max_value = tl.full((M,), -1.0e20, dtype=tl.float32)
    normalizer = tl.zeros((M,), dtype=tl.float32)
    accumulator = tl.zeros((M, HEAD_DIM), dtype=tl.float32)
    softmax_scale_log2: tl.constexpr = (HEAD_DIM**-0.5) * 1.4426950408889634
    # ---- pass 1: the tile's union of whole compressed blocks (shared gather, per-row membership) ----
    ubound = tl.load(cnt_ptr + tile)
    for t in range(0, ubound, BNB):
        emask = (t + b_off) < ubound
        phys = tl.load(uni_ptr + tile * stride_uni + t + b_off, mask=emask, other=-1)   # page*PAGE + offset, or -1
        tok2 = tl.where((phys >= 0)[:, None], phys[:, None] + j_off[None, :], -1)
        physical_token = tl.reshape(tok2, (BN,))
        valid = physical_token >= 0
        safe_token = tl.maximum(physical_token, 0)
        safe_page = (safe_token // PAGE_SIZE).to(tl.int64)     # whole blocks never straddle a page (PAGE % CR == 0)
        page_offset = safe_token % PAGE_SIZE
        keys = tl.load(k_cache_ptr + safe_page[None, :] * stride_k_block + page_offset[None, :] * stride_k_token
                       + kv_head * stride_k_head + dim_offsets[:, None], mask=valid[None, :], other=0.0)
        values = tl.load(v_cache_ptr + safe_page[:, None] * stride_v_block + page_offset[:, None] * stride_v_token
                         + kv_head * stride_v_head + dim_offsets[None, :], mask=valid[:, None], other=0.0)
        memb = tl.load(mem_ptr + tile * stride_mem_t + r_of_m[:, None] * stride_mem_r + t + b_off[None, :],
                       mask=emask[None, :], other=0)
        memt = tl.reshape(tl.broadcast_to(memb[:, :, None], (M, BNB, CR)), (M, BN))
        active = (memt > 0) & valid[None, :] & req_ok[:, None]
        scores = tl.dot(query, keys) * softmax_scale_log2
        scores = tl.where(active, scores, -1.0e20)
        next_max = tl.maximum(max_value, tl.max(scores, axis=1))
        alpha = tl.math.exp2(max_value - next_max)
        probabilities = tl.where(active, tl.math.exp2(scores - next_max[:, None]), 0.0)
        accumulator = tl.dot(probabilities.to(values.dtype), values, acc=accumulator * alpha[:, None])
        normalizer = normalizer * alpha + tl.sum(probabilities, axis=1)
        max_value = next_max
    # ---- pass 2: each row's causal tail (<= CR-1 tokens of its open block), one 16-column tile per tile ----
    tt = tl.arange(0, TAIL_COLS)
    slot_row = tt // TAIL_PER_ROW
    tail_tok = tl.load(tail_ptr + tile * stride_tail + tt, mask=tt < R * TAIL_PER_ROW, other=-1)
    safe_token = tl.maximum(tail_tok, 0)
    logical_page = safe_token // PAGE_SIZE
    page_offset = safe_token % PAGE_SIZE
    valid = (tail_tok >= 0) & (logical_page < PAGE_TABLE_WIDTH)
    physical_page = tl.load(block_table_ptr + tile_request * stride_table_req
                            + tl.minimum(logical_page, PAGE_TABLE_WIDTH - 1), mask=valid, other=-1)
    valid &= (physical_page >= 0) & (physical_page < num_cache_blocks)
    safe_page = tl.maximum(physical_page, 0).to(tl.int64)
    keys = tl.load(k_cache_ptr + safe_page[None, :] * stride_k_block + page_offset[None, :] * stride_k_token
                   + kv_head * stride_k_head + dim_offsets[:, None], mask=valid[None, :], other=0.0)
    values = tl.load(v_cache_ptr + safe_page[:, None] * stride_v_block + page_offset[:, None] * stride_v_token
                     + kv_head * stride_v_head + dim_offsets[None, :], mask=valid[:, None], other=0.0)
    active = (r_of_m[:, None] == slot_row[None, :]) & valid[None, :] & req_ok[:, None]
    scores = tl.dot(query, keys) * softmax_scale_log2
    scores = tl.where(active, scores, -1.0e20)
    next_max = tl.maximum(max_value, tl.max(scores, axis=1))
    alpha = tl.math.exp2(max_value - next_max)
    probabilities = tl.where(active, tl.math.exp2(scores - next_max[:, None]), 0.0)
    accumulator = tl.dot(probabilities.to(values.dtype), values, acc=accumulator * alpha[:, None])
    normalizer = normalizer * alpha + tl.sum(probabilities, axis=1)
    has_values = normalizer > 0
    out = tl.where(has_values[:, None], accumulator / tl.maximum(normalizer[:, None], 1.0e-20), 0.0)
    # stock contract: rows with an invalid request are written as zeros (their query was masked to 0 above)
    smask = (row < num_rows) & (h_of_m < GROUP_SIZE)
    tl.store(out_ptr + row[:, None] * stride_out_row + (first_head + h_of_m[:, None]) * stride_out_head
             + dim_offsets[None, :], out.to(tl.bfloat16), mask=smask[:, None])


def _qsa_union_split(logical_indices: torch.Tensor, compress_ratio: int, token_topk: int):
    """Whole compressed blocks [rows, block_topk] (-1 padded) and the causal tail tokens [rows, CR-1]
    (-1 padded), from the expanded selection (layout of expand_qsa_block_indices: complete blocks first as CR
    consecutive tokens each, then the tail of the open block, then -1; newer builds append a count column)."""
    rows, width = logical_indices.shape
    block_topk = token_topk // compress_ratio
    selection_width = token_topk + compress_ratio - 1
    if width not in (selection_width, selection_width + 1):
        raise ValueError(f"QSA union: unexpected selection width {width} for token_topk {token_topk}, ratio {compress_ratio}")
    cols = logical_indices[:, : block_topk * compress_ratio].reshape(rows, block_topk, compress_ratio)
    first = cols[:, :, 0]
    whole = (first >= 0) & ((first % compress_ratio) == 0) & (cols[:, :, compress_ratio - 1] == first + compress_ratio - 1)
    blocks = torch.where(whole, first // compress_ratio, torch.full_like(first, -1)).to(torch.int32)
    c = whole.sum(dim=1, keepdim=True)
    tcols = (c * compress_ratio + torch.arange(compress_ratio - 1, device=logical_indices.device)[None, :]).clamp(max=selection_width - 1)
    tail = torch.gather(logical_indices, 1, tcols)
    tail = torch.where(tail >= 0, tail, torch.full_like(tail, -1)).to(torch.int32)
    return blocks, tail


_QSA_UNION_BIG = (1 << 27) * 8 + 7


def _qsa_union_launch(packed_rows: torch.Tensor, rows: int, R: int, token_to_req, block_table, num_requests: int,
                      k_cache, compress_ratio: int, qpos: torch.Tensor | None, tail: torch.Tensor | None):
    """packed_rows: [rows, E] int32 with (block*8 + row_in_tile) or _QSA_UNION_BIG; rows padded to T*R here."""
    E = packed_rows.shape[1]
    T = (rows + R - 1) // R
    N = R * E                                   # 2*512 = 1024 (or 4*512): exact powers of two
    assert N & (N - 1) == 0, "block_topk * R must be a power of two"
    dev = packed_rows.device
    if T * R != rows:
        packed = torch.full((T * R, E), _QSA_UNION_BIG, device=dev, dtype=torch.int32)
        packed[:rows] = packed_rows
    else:
        packed = packed_rows
    packed, _ = torch.sort(packed.view(T, N), dim=1)
    uni = torch.full((T, N), -1, device=dev, dtype=torch.int32)
    mem = torch.zeros((T, R, N), device=dev, dtype=torch.int8)
    cnt = torch.empty(T, device=dev, dtype=torch.int32)
    tails = torch.empty((T, _QSA_UNION_TAIL_COLS), device=dev, dtype=torch.int32)
    from_qpos = qpos is not None
    _qsa_union_build_kernel[(T,)](packed, uni, mem, cnt, token_to_req, block_table, qpos if from_qpos else tails, tails,
                                  packed.stride(0), uni.stride(0), mem.stride(0), mem.stride(1), block_table.stride(0),
                                  tails.stride(0), rows, num_requests, block_table.shape[1], k_cache.shape[0],
                                  N=N, R=R, CR=compress_ratio, PAGE_SIZE=k_cache.shape[1],
                                  TAIL_COLS=_QSA_UNION_TAIL_COLS, WRITE_TAILS=from_qpos, num_warps=4)
    if not from_qpos:  # tails given as [rows, CR-1] tokens (split path)
        tails.fill_(-1)
        w = tail.shape[1]
        if T * R != rows:
            tl_ = torch.full((T * R, w), -1, device=dev, dtype=torch.int32)
            tl_[:rows] = tail
        else:
            tl_ = tail
        tails[:, : R * w] = tl_.view(T, R * w)
    return uni, mem, cnt, tails, T


def _qsa_union_build(blocks: torch.Tensor, tail: torch.Tensor, R: int, token_to_req: torch.Tensor,
                     block_table: torch.Tensor, num_requests: int, k_cache: torch.Tensor, compress_ratio: int):
    """From the split of the expanded buffer (fallback path)."""
    rows = blocks.shape[0]
    rr = (torch.arange(rows, device=blocks.device, dtype=torch.int32) % R)[:, None]
    packed = torch.where(blocks >= 0, blocks * 8 + rr, torch.full_like(blocks, _QSA_UNION_BIG))
    return _qsa_union_launch(packed, rows, R, token_to_req, block_table, num_requests, k_cache, compress_ratio, None, tail)


def _qsa_union_build_raw(block_indices: torch.Tensor, query_positions: torch.Tensor, visible_blocks: torch.Tensor,
                         R: int, token_to_req: torch.Tensor, block_table: torch.Tensor, num_requests: int,
                         k_cache: torch.Tensor, compress_ratio: int):
    """From the indexer's selection directly (lever 1): the expansion kernel only expands ranks below
    min(visible_blocks, block_topk), everything else is padding; the tails come from the query positions."""
    rows, E = block_indices.shape
    dev = block_indices.device
    rank = torch.arange(E, device=dev, dtype=torch.int32)[None, :]
    rr = (torch.arange(rows, device=dev, dtype=torch.int32) % R)[:, None]
    bi = block_indices.to(torch.int32)
    keep = (bi >= 0) & (rank < visible_blocks.to(torch.int32)[:, None])
    packed = torch.where(keep, bi * 8 + rr, torch.full_like(bi, _QSA_UNION_BIG))
    return _qsa_union_launch(packed, rows, R, token_to_req, block_table, num_requests, k_cache, compress_ratio,
                             query_positions.to(torch.int32).contiguous(), None)


def qsa_union_layout_ok(k_cache: torch.Tensor, v_cache: torch.Tensor, compress_ratio: int) -> bool:
    """The pre-resolved page*PAGE + offset form needs whole compressed blocks inside a page (PAGE % CR == 0);
    strides are free (the server's K/V views are slices of a wider tensor). Otherwise the stock kernel runs."""
    return k_cache.shape[1] % compress_ratio == 0


_QSA_UNION_LOGGED = set()


def _qsa_union_log_once(what: str):
    if what not in _QSA_UNION_LOGGED:
        _QSA_UNION_LOGGED.add(what)
        print(f"QSAUNION path: {what}", flush=True)


def qsa_sparse_paged_attention_union(q, k_cache, v_cache, logical_indices, block_table, token_to_req, out, *,
                                     compress_ratio: int, token_topk: int, num_requests: int, R: int | None = None,
                                     raw: tuple | None = None):
    rows = q.shape[0]
    R = R or _QSA_UNION_R
    # the stash must be this step's selection: same rows, and its expansion is the buffer we were handed
    if (raw is not None and raw[0].shape[0] == rows and raw[0].shape[1] == token_topk // compress_ratio
            and raw[3].data_ptr() == logical_indices.data_ptr()):
        _qsa_union_log_once(f"raw (indexer selection), R={R} BNB={_QSA_UNION_BNB} warps={_QSA_UNION_WARPS}")
        uni, mem, cnt, tails, T = _qsa_union_build_raw(raw[0], raw[1], raw[2], R, token_to_req, block_table, num_requests,
                                                       k_cache, compress_ratio)
    else:
        _qsa_union_log_once(f"split (expanded buffer; raw stash {'absent' if raw is None else 'stale'}), R={R}")
        blocks, tail = _qsa_union_split(logical_indices, compress_ratio, token_topk)
        uni, mem, cnt, tails, T = _qsa_union_build(blocks, tail, R, token_to_req, block_table, num_requests, k_cache, compress_ratio)
    group_size = q.shape[1] // k_cache.shape[2]
    _qsa_union_attn_kernel[(T, k_cache.shape[2])](
        q, k_cache, v_cache, uni, mem, cnt, tails, block_table, token_to_req, out,
        q.stride(0), q.stride(1), k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2), uni.stride(0), mem.stride(0), mem.stride(1),
        tails.stride(0), block_table.stride(0), out.stride(0), out.stride(1), rows, num_requests, k_cache.shape[0],
        R=R, GP=triton.next_power_of_2(group_size), GROUP_SIZE=group_size, HEAD_DIM=q.shape[2],
        BNB=_QSA_UNION_BNB, CR=compress_ratio, TAIL_COLS=_QSA_UNION_TAIL_COLS, PAGE_SIZE=k_cache.shape[1],
        PAGE_TABLE_WIDTH=block_table.shape[1], num_warps=_QSA_UNION_WARPS, num_stages=1)
    return out


def qsa_union_eligible(num_rows: int, num_requests: int, compress_ratio: int, token_topk: int) -> bool:
    """Decided from CPU metadata only (no device reads): enabled, a prefill-sized single-request batch
    (tiles must not straddle requests), and a block_topk that keeps the sort width a power of two."""
    if not _QSA_UNION or num_rows < _QSA_UNION_MIN_ROWS or num_requests != 1:
        return False
    block_topk = token_topk // compress_ratio
    return compress_ratio & (compress_ratio - 1) == 0 and block_topk & (block_topk - 1) == 0
