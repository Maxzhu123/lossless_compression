"""Generic overflow counting, compaction, and decode scatter kernels."""

import triton
from triton import language as tl

from ...codec.autotune import (
    COMPACT_BAD_STREAMS_AUTOTUNE_CONFIGS,
    COMPACT_EXTRA_AUTOTUNE_CONFIGS,
    SCATTER_FALLBACK_AUTOTUNE_CONFIGS,
)
from ..primitives import pack_bf16


@triton.autotune(
    configs=COMPACT_BAD_STREAMS_AUTOTUNE_CONFIGS,
    key=["n_streams", "steps"],
    restore_value=["bad_count", "fallback_total"],
)
@triton.jit
def _count_bad_streams_kernel(
    extra_starts,
    bad_count, fallback_total, n_streams, steps,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_streams
    start = tl.load(extra_starts + offs, mask=mask, other=255)
    bad = mask & (start != 255)
    lengths = tl.where(bad, steps - start, 0)
    cnt = tl.sum(bad.to(tl.int32), axis=0)
    total_len = tl.sum(lengths, axis=0)
    tl.atomic_add(bad_count, cnt, mask=cnt != 0)
    tl.atomic_add(fallback_total, total_len, mask=total_len != 0)


@triton.autotune(
    configs=COMPACT_BAD_STREAMS_AUTOTUNE_CONFIGS,
    key=["n_streams", "steps"],
    restore_value=["bad_count", "fallback_total"],
)
@triton.jit
def _compact_bad_streams_kernel(
    extra_starts,
    bad_streams_out, bad_starts_out, fallback_offsets_out,
    metadata_buffer, allocation_descriptor, final_counts,
    bad_count, fallback_total, n_streams, steps,
    BUFFERED: tl.constexpr, BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_streams
    start = tl.load(extra_starts + offs, mask=mask, other=255)
    bad = mask & (start != 255)
    lengths = tl.where(bad, steps - start, 0)
    cnt = tl.sum(bad.to(tl.int32), axis=0)
    total_len = tl.sum(lengths, axis=0)
    block_start = tl.atomic_add(bad_count, cnt, mask=cnt != 0)
    fallback_block_start = tl.atomic_add(
        fallback_total, total_len, mask=total_len != 0
    )
    prefix = tl.cumsum(bad.to(tl.int32), axis=0) - bad.to(tl.int32)
    offset_prefix = tl.cumsum(lengths, axis=0) - lengths
    dense_offset = fallback_block_start + offset_prefix
    pos = block_start + prefix
    if BUFFERED:
        count = tl.load(final_counts).to(tl.int32)
        base = tl.load(allocation_descriptor).to(tl.int32)
        base_words = base // 4
        tl.store(metadata_buffer + base_words + pos, offs.to(tl.int32), mask=bad)
        tl.store(
            metadata_buffer + base_words + count + pos,
            dense_offset,
            mask=bad,
        )
        tl.store(bad_starts_out + base + 8 * count + pos, start, mask=bad)
    else:
        tl.store(bad_streams_out + pos, offs.to(tl.int32), mask=bad)
        tl.store(bad_starts_out + pos, start, mask=bad)
        tl.store(
            fallback_offsets_out + pos,
            dense_offset,
            mask=bad,
        )


@triton.jit
def _compact_extra_impl(
    source_bits,
    extra_streams, extra_starts,
    fallback_offsets, fallback_data,
    metadata_buffer, allocation_descriptor, final_counts, bad_count,
    n_elements,
    BUFFERED: tl.constexpr, PRECOMPUTED: tl.constexpr,
    LOGICAL_NUMEL: tl.constexpr,
    BLOCK: tl.constexpr, N_LANES: tl.constexpr, N_STEPS: tl.constexpr, TILE: tl.constexpr,
):
    pid = tl.program_id(0)
    tile = pid * TILE + tl.arange(0, TILE)
    if BUFFERED:
        count = tl.load(final_counts).to(tl.int32)
    else:
        count = tl.load(bad_count).to(tl.int32)
    if pid * TILE >= count:
        return
    valid = tile < count
    if BUFFERED:
        base = tl.load(allocation_descriptor).to(tl.int32)
        base_words = base // 4
        stream = tl.load(metadata_buffer + base_words + tile, mask=valid, other=0).to(tl.int32)
        start = tl.load(
            extra_starts + base + 8 * count + tile,
            mask=valid,
            other=N_STEPS,
        ).to(tl.int32)
        fallback_offset = tl.load(
            metadata_buffer + base_words + count + tile,
            mask=valid,
            other=0,
        ).to(tl.int32)
        fallback_base = base + 9 * count
    else:
        stream = tl.load(extra_streams + tile, mask=valid, other=0).to(tl.int32)
        start = tl.load(extra_starts + tile, mask=valid, other=N_STEPS).to(tl.int32)
        fallback_offset = tl.load(fallback_offsets + tile, mask=valid, other=0).to(tl.int32)
    block = stream // N_LANES
    lane = stream - block * N_LANES
    tail_steps = N_STEPS - start
    max_tail = tl.max(tl.where(valid, tail_steps, 0), axis=0)
    # Hoisted swizzle (see _encode_impl): block_shift is loop-invariant.
    block_shift = (block * N_STEPS) & 255
    block_base = block * BLOCK
    for step in tl.range(0, max_tail):
        source_offset = block_base + (step + start) * N_LANES + lane
        active = valid & (step < tail_steps) & (source_offset < n_elements)
        logical_n = block * N_STEPS + step + start
        logical_k = (lane + ((block_shift + step + start) & 255)) & 255
        input_offset = logical_n * N_LANES + logical_k
        input_active = active & (input_offset < LOGICAL_NUMEL)
        value = tl.load(
            source_bits + input_offset, mask=input_active, other=0,
        ).to(tl.int32)
        if PRECOMPUTED:
            values = (value - 127).to(tl.int8)
        else:
            values = (((value >> 7) & 0xFF) - 127).to(tl.int8)
        if BUFFERED:
            tl.store(
                fallback_data + fallback_base + fallback_offset + step,
                values,
                mask=active,
            )
        else:
            tl.store(fallback_data + fallback_offset + step, values, mask=active)


@triton.autotune(
    configs=COMPACT_EXTRA_AUTOTUNE_CONFIGS,
    key=["n_elements", "N_LANES", "N_STEPS", "BLOCK"],
)
@triton.jit
def _compact_components_extra_kernel(
    source_bits, extra_streams, extra_starts,
    fallback_offsets, fallback_data, metadata_buffer,
    allocation_descriptor, final_counts, bad_count, n_elements,
    BUFFERED: tl.constexpr, LOGICAL_NUMEL: tl.constexpr,
    BLOCK: tl.constexpr, N_LANES: tl.constexpr,
    N_STEPS: tl.constexpr, TILE: tl.constexpr,
):
    """Compact overflow exponents from flattened precomputed planes."""
    _compact_extra_impl(
        source_bits, extra_streams, extra_starts, fallback_offsets,
        fallback_data, metadata_buffer, allocation_descriptor,
        final_counts, bad_count, n_elements, BUFFERED, True,
        LOGICAL_NUMEL,
        BLOCK, N_LANES, N_STEPS, TILE,
    )


@triton.autotune(
    configs=COMPACT_EXTRA_AUTOTUNE_CONFIGS,
    key=["n_elements", "N_LANES", "N_STEPS", "BLOCK"],
)
@triton.jit
def _compact_extra_kernel(
    source_bits, extra_streams, extra_starts,
    fallback_offsets, fallback_data, metadata_buffer,
    allocation_descriptor, final_counts, bad_count, n_elements,
    BUFFERED: tl.constexpr, LOGICAL_NUMEL: tl.constexpr,
    BLOCK: tl.constexpr, N_LANES: tl.constexpr,
    N_STEPS: tl.constexpr, TILE: tl.constexpr,
):
    """Compact flattened overflow values through the 1D source mapping."""
    _compact_extra_impl(
        source_bits, extra_streams, extra_starts, fallback_offsets,
        fallback_data, metadata_buffer, allocation_descriptor,
        final_counts, bad_count, n_elements, BUFFERED, False,
        LOGICAL_NUMEL,
        BLOCK, N_LANES, N_STEPS, TILE,
    )


@triton.autotune(
    configs=SCATTER_FALLBACK_AUTOTUNE_CONFIGS,
    key=["n_elements", "N_LANES", "N_STEPS", "BLOCK"],
)
@triton.jit
def _scatter_blocked_fallback_kernel(
    bad_streams, bad_starts, fallback_offsets,
    fallback_buffer, fallback_base, metadata, descriptor, fallback_count,
    sign_mantissa, output, n_elements,
    BUFFERED: tl.constexpr,
    LOGICAL_NUMEL: tl.constexpr,
    TILE: tl.constexpr, BLOCK: tl.constexpr,
    N_LANES: tl.constexpr, N_STEPS: tl.constexpr,
):
    """Overwrite mapped decode output with compact fallback stream tails."""
    pid = tl.program_id(0)
    tile = pid * TILE + tl.arange(0, TILE)
    count = tl.load(fallback_count).to(tl.int32)
    if pid * TILE >= count:
        return
    valid = tile < count
    if BUFFERED:
        base = tl.load(descriptor).to(tl.int32)
        base_words = base // 4
        stream = tl.load(metadata + base_words + tile, mask=valid, other=0)
        start = tl.load(
            bad_starts + base + 8 * count + tile,
            mask=valid, other=N_STEPS,
        )
        fallback_offset = tl.load(
            metadata + base_words + count + tile, mask=valid, other=0,
        )
        fallback_base = base + 9 * count
    else:
        stream = tl.load(bad_streams + tile, mask=valid, other=0)
        start = tl.load(bad_starts + tile, mask=valid, other=N_STEPS)
        fallback_offset = tl.load(fallback_offsets + tile, mask=valid, other=0)
    stream = stream.to(tl.int32)
    start = start.to(tl.int32)
    fallback_offset = fallback_offset.to(tl.int32)
    block = stream // N_LANES
    lane = stream % N_LANES
    # Hoisted swizzle (see _encode_impl): block_shift is loop-invariant.
    block_shift = (block * N_STEPS) & 255
    block_base = block * BLOCK
    for step in tl.range(0, N_STEPS):
        storage_offset = block_base + step * N_LANES + lane
        logical_n = block * N_STEPS + step
        logical_k = (lane + ((block_shift + step) & 255)) & 255
        logical_offset = logical_n * N_LANES + logical_k
        active = (
            valid & (step >= start) & (storage_offset < n_elements)
            & (logical_offset < LOGICAL_NUMEL)
        )
        exponent = tl.load(
            fallback_buffer + fallback_base + fallback_offset + step - start,
            mask=active, other=0,
        ).to(tl.int32)
        sm = tl.load(sign_mantissa + storage_offset, mask=active, other=0)
        packed = pack_bf16(exponent, sm)
        tl.store(output + logical_offset, packed.to(tl.int16), mask=active)
