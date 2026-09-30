"""Generic BF16/component encoding with private or buffered overflow storage."""

import torch
import triton
from triton import language as tl

from ...codec.autotune import ENCODE_AUTOTUNE_CONFIGS
from ...codec.geometry import geometry
from ...comp_format import Distribution, StorageLayout
from ...comp_tensor import CompressedTensor
from ...compression.huffman_tables import get_distribution_tables
from ...tensor_buffer import TensorBuffer
from ..common.tables import _estimate_center_kernel, _shift_encoding_table_kernel
from .compaction import (
    _count_bad_streams_kernel, _compact_bad_streams_kernel,
    _compact_extra_kernel, _compact_components_extra_kernel,
)

CENTER_SAMPLE_SIZE = 4096


@triton.jit
def _encode_impl(
    source_bits, sign_mantissa, encoded, encode_table,
    extra_starts,
    n_elements, n_streams,
    PRECOMPUTED: tl.constexpr,
    LOGICAL_NUMEL: tl.constexpr,
    FIXED_WORDS: tl.constexpr,
    BLOCK: tl.constexpr, N_LANES: tl.constexpr, N_STEPS: tl.constexpr,
):
    """Encode one flattened contiguous block per program (1D storage).

    Storage offset ``block * BLOCK + step * N_LANES + lane`` maps to the
    flattened logical element ``logical_n * N_LANES + out_k`` with the
    row-dependent swizzle ``out_k = (lane + (logical_n & 255)) & 255``,
    which spreads neighbouring values across lanes.
    """
    block = tl.program_id(0)
    lanes = tl.arange(0, N_LANES)
    lane_index = block * N_LANES + lanes
    word = tl.zeros((N_LANES,), tl.int32)
    shift = tl.zeros((N_LANES,), tl.int32)
    word_value = tl.zeros((N_LANES,), tl.uint32)
    overflow = tl.zeros((N_LANES,), tl.int1)
    extra_start = tl.full((N_LANES,), N_STEPS, tl.int32)
    has_data = block * BLOCK + lanes < n_elements

    logical_n_base = block * N_STEPS
    # Hoisted swizzle: (block*N_STEPS + step) & 255 == (block_shift + step) & 255
    # with block_shift loop-invariant, so the hot loop only sees `step`.
    # (N_STEPS=256 -> block_shift=0; N_STEPS=128 -> 128*(block & 1).)
    block_shift = (block * N_STEPS) & 255
    block_base = block * BLOCK
    # Fully-valid blocks can skip all per-element validity/mask checks.
    full_block = (block + 1) * BLOCK <= LOGICAL_NUMEL
    if full_block:
        for step in tl.range(0, N_STEPS, 2, loop_unroll_factor=4):
            source_offset = block_base + step * N_LANES + lanes
            logical_n = logical_n_base + step
            shift0 = (block_shift + step) & 255
            shift1 = (block_shift + step + 1) & 255
            input_k0 = (lanes + shift0) & 255
            input_k1 = (lanes + shift1) & 255
            input_offset0 = logical_n * N_LANES + input_k0
            input_offset1 = (logical_n + 1) * N_LANES + input_k1
            value0 = tl.load(source_bits + input_offset0).to(tl.int32)
            value1 = tl.load(source_bits + input_offset1).to(tl.int32)
            if PRECOMPUTED:
                byte0 = value0 & 255
                byte1 = value1 & 255
            else:
                byte0 = (value0 >> 7) & 0xFF
                byte1 = (value1 >> 7) & 0xFF
                sm0 = (value0 & 0x7F) | ((value0 >> 8) & 0x80)
                sm1 = (value1 & 0x7F) | ((value1 >> 8) & 0x80)
                tl.store(sign_mantissa + source_offset, sm0.to(tl.uint8))
                tl.store(
                    sign_mantissa + source_offset + N_LANES,
                    sm1.to(tl.uint8),
                )
            packed0 = tl.load(encode_table + byte0).to(tl.uint32)
            packed1 = tl.load(encode_table + byte1).to(tl.uint32)
            length0 = (packed0 >> 20).to(tl.int32)
            length1 = (packed1 >> 20).to(tl.int32)
            length = length0 + length1
            code = (packed0 & 0xfffff) | ((packed1 & 0xfffff) << length0)

            new_word = word_value | (code << shift)
            crosses_word = shift + length >= 32
            pair_overflow = word * 32 + shift + length > FIXED_WORDS * 32
            first_overflow = (~overflow) & pair_overflow

            extra_start = tl.where(first_overflow, step, extra_start)

            store_value = tl.where(first_overflow, word_value, new_word)
            safe_word = tl.minimum(word, FIXED_WORDS - 1)
            tl.store(
                encoded + safe_word * n_streams + lane_index,
                store_value,
                mask=crosses_word & (word < FIXED_WORDS),
            )
            word_value = tl.where(crosses_word, code >> (32 - shift), new_word)
            word += tl.where(crosses_word, 1, 0)
            shift = tl.where(crosses_word, shift + length - 32, shift + length)
            overflow |= pair_overflow
    else:
        for step in tl.range(0, N_STEPS, 2, loop_unroll_factor=4):
            source_offset = block_base + step * N_LANES + lanes
            valid0 = source_offset < n_elements
            valid1 = source_offset + N_LANES < n_elements
            logical_n = logical_n_base + step
            shift0 = (block_shift + step) & 255
            shift1 = (block_shift + step + 1) & 255
            input_k0 = (lanes + shift0) & 255
            input_k1 = (lanes + shift1) & 255
            input_offset0 = logical_n * N_LANES + input_k0
            input_offset1 = (logical_n + 1) * N_LANES + input_k1
            input_valid0 = input_offset0 < LOGICAL_NUMEL
            input_valid1 = input_offset1 < LOGICAL_NUMEL
            value0 = tl.load(
                source_bits + input_offset0, mask=input_valid0, other=0,
            ).to(tl.int32)
            value1 = tl.load(
                source_bits + input_offset1, mask=input_valid1, other=0,
            ).to(tl.int32)
            if PRECOMPUTED:
                byte0 = value0 & 255
                byte1 = value1 & 255
            else:
                byte0 = (value0 >> 7) & 0xFF
                byte1 = (value1 >> 7) & 0xFF
                sm0 = (value0 & 0x7F) | ((value0 >> 8) & 0x80)
                sm1 = (value1 & 0x7F) | ((value1 >> 8) & 0x80)
                tl.store(
                    sign_mantissa + source_offset, sm0.to(tl.uint8),
                    mask=valid0 & input_valid0,
                )
                tl.store(
                    sign_mantissa + source_offset + N_LANES,
                    sm1.to(tl.uint8), mask=valid1 & input_valid1,
                )
            packed0 = tl.load(encode_table + byte0).to(tl.uint32)
            packed1 = tl.load(encode_table + byte1).to(tl.uint32)
            packed0 = tl.where(input_valid0, packed0, 0)
            packed1 = tl.where(input_valid1, packed1, 0)
            length0 = (packed0 >> 20).to(tl.int32)
            length1 = (packed1 >> 20).to(tl.int32)
            length = length0 + length1
            code = (packed0 & 0xfffff) | ((packed1 & 0xfffff) << length0)

            new_word = word_value | (code << shift)
            crosses_word = shift + length >= 32
            pair_overflow = (word * 32 + shift + length > FIXED_WORDS * 32) & valid0
            first_overflow = (~overflow) & pair_overflow

            extra_start = tl.where(first_overflow, step, extra_start)

            store_value = tl.where(first_overflow, word_value, new_word)
            safe_word = tl.minimum(word, FIXED_WORDS - 1)
            tl.store(
                encoded + safe_word * n_streams + lane_index,
                store_value,
                mask=crosses_word & (word < FIXED_WORDS),
            )
            word_value = tl.where(crosses_word, code >> (32 - shift), new_word)
            word += tl.where(crosses_word, 1, 0)
            shift = tl.where(crosses_word, shift + length - 32, shift + length)
            overflow |= pair_overflow

    safe_word = tl.minimum(word, FIXED_WORDS - 1)
    tl.store(
        encoded + safe_word * n_streams + lane_index,
        word_value,
        mask=has_data & (word < FIXED_WORDS) & (shift != 0),
    )
    tl.store(
        extra_starts + lane_index,
        tl.where(has_data & overflow, extra_start, 255),
    )


@triton.autotune(
    configs=ENCODE_AUTOTUNE_CONFIGS,
    key=["n_elements", "N_LANES", "N_STEPS", "FIXED_WORDS"],
)
@triton.jit
def _encode_components_kernel(
    source_bits, sign_mantissa, encoded, encode_table,
    extra_starts, n_elements, n_streams,
    LOGICAL_NUMEL: tl.constexpr,
    FIXED_WORDS: tl.constexpr,
    BLOCK: tl.constexpr, N_LANES: tl.constexpr, N_STEPS: tl.constexpr,
):
    """Encode flattened precomputed exponent planes into 1D storage."""
    _encode_impl(
        source_bits, sign_mantissa, encoded, encode_table, extra_starts,
        n_elements, n_streams, True,
        LOGICAL_NUMEL,
        FIXED_WORDS, BLOCK, N_LANES, N_STEPS,
    )


@triton.autotune(
    configs=ENCODE_AUTOTUNE_CONFIGS,
    key=["n_elements", "N_LANES", "N_STEPS", "FIXED_WORDS"],
)
@triton.jit
def _encode_kernel(
    source_bits, sign_mantissa, encoded, encode_table,
    extra_starts, n_elements, n_streams,
    LOGICAL_NUMEL: tl.constexpr,
    FIXED_WORDS: tl.constexpr,
    BLOCK: tl.constexpr, N_LANES: tl.constexpr, N_STEPS: tl.constexpr,
):
    """Encode a flattened tensor through the 1D codec mapping."""
    _encode_impl(
        source_bits, sign_mantissa, encoded, encode_table, extra_starts,
        n_elements, n_streams, False,
        LOGICAL_NUMEL,
        FIXED_WORDS, BLOCK, N_LANES, N_STEPS,
    )


def _estimate_center(source, size, *, precomputed, ignore_zero=False):
    """Estimate the exponent center from stratified jittered GPU samples."""
    sample_size = min(size, CENTER_SAMPLE_SIZE)
    center = torch.empty(1, dtype=torch.int32, device=source.device)
    _estimate_center_kernel[(1,)](
        source, center, size, SAMPLE_SIZE=sample_size,
        PRECOMPUTED=precomputed, IGNORE_ZERO=ignore_zero,
    )
    return center


def _launch_encode(
    source_values, sign_mantissa, encoded, encode_table, extra_starts,
    size, streams, *,
    precomputed, logical_numel,
    fixed_words, block_symbols, lanes, steps, blocks,
):
    """Launch the 1D encode kernel for raw BF16 or precomputed components."""
    if precomputed:
        _encode_components_kernel[(blocks,)](
            source_values, sign_mantissa, encoded, encode_table, extra_starts,
            size, streams, LOGICAL_NUMEL=logical_numel,
            FIXED_WORDS=fixed_words, BLOCK=block_symbols,
            N_LANES=lanes, N_STEPS=steps,
        )
    else:
        _encode_kernel[(blocks,)](
            source_values, sign_mantissa, encoded, encode_table, extra_starts,
            size, streams, LOGICAL_NUMEL=logical_numel,
            FIXED_WORDS=fixed_words, BLOCK=block_symbols,
            N_LANES=lanes, N_STEPS=steps,
        )


def _compact_bad_streams(
    extra_starts,
    bad_streams_out, bad_starts_out, fallback_offsets_out,
    metadata_buffer, allocation_descriptor, final_counts,
    bad_count, fallback_total, streams, steps,
    *,
    buffered,
):
    """Compact overflow-stream metadata for a buffered or private fallback path."""
    compact_grid = lambda meta: (triton.cdiv(streams, meta["BLOCK"]),)
    _compact_bad_streams_kernel[compact_grid](
        extra_starts, bad_streams_out, bad_starts_out, fallback_offsets_out,
        metadata_buffer, allocation_descriptor, final_counts,
        bad_count, fallback_total, streams, steps,
        BUFFERED=buffered,
    )


def _compact_extra(
    source_values,
    bad_streams, bad_starts, fallback_offsets, fallback_data,
    metadata_buffer, allocation_descriptor, final_counts, bad_count,
    size, streams, *,
    precomputed, buffered,
    logical_numel,
    block_symbols, lanes, steps,
):
    """Compact fallback tail values for a buffered or private fallback path."""
    compact_grid = lambda meta: (triton.cdiv(streams, meta["TILE"]),)
    if precomputed:
        _compact_components_extra_kernel[compact_grid](
            source_values, bad_streams, bad_starts, fallback_offsets,
            fallback_data, metadata_buffer, allocation_descriptor,
            final_counts, bad_count, size,
            BUFFERED=buffered, LOGICAL_NUMEL=logical_numel,
            BLOCK=block_symbols,
            N_LANES=lanes, N_STEPS=steps,
        )
    else:
        _compact_extra_kernel[compact_grid](
            source_values, bad_streams, bad_starts, fallback_offsets,
            fallback_data, metadata_buffer, allocation_descriptor,
            final_counts, bad_count, size,
            BUFFERED=buffered, LOGICAL_NUMEL=logical_numel,
            BLOCK=block_symbols,
            N_LANES=lanes, N_STEPS=steps,
        )


def encode_components(
    source_values: torch.Tensor,
    sign_mantissa: torch.Tensor,
    size: int,
    distribution: Distribution,
    buffer: TensorBuffer | None,
    shape: tuple[int, ...],
    *,
    precomputed: bool,
    center: torch.Tensor | None = None,
    logical_numel: int,
) -> CompressedTensor:
    """Encode BF16 bits or split components into a ``CompressedTensor``.

    Args:
        source_values: Int16 BF16 bits, or raw uint8 exponents if ``precomputed``.
        sign_mantissa: Output side-byte stream, or precomputed side bytes.
        size: Padded codec element count (storage element count).
        distribution: Codebook and stream geometry.
        buffer: Optional shared fallback arena.
        shape: Original tensor shape.
        precomputed: ``False`` extracts both fields from BF16 bits; ``True``
            consumes raw exponent bytes plus existing side bytes from a fused op.
        center: Precomputed exponent center, or ``None`` to estimate it.
        logical_numel: Flattened logical element count (1D storage mapping).
    """
    # Geometry fixes the independent stream count and per-stream bit budget.
    stream_geometry = geometry(distribution)
    block_symbols, lanes, steps, fixed_words = stream_geometry
    blocks = triton.cdiv(size, block_symbols)
    streams = blocks * lanes
    if center is None:
        center = _estimate_center(
            source_values, logical_numel, precomputed=precomputed,
            ignore_zero=distribution.zero_prob > 0,
        )
    encode_table, _, _ = get_distribution_tables(distribution)
    shifted_encode = torch.empty(256, dtype=torch.int32, device=source_values.device)
    _shift_encoding_table_kernel[(1,)](
        encode_table, center, shifted_encode, BLOCK=256,
    )
    # Encode and decode share this sampled center through the result metadata.
    encoded = torch.empty(
        streams * fixed_words + 4,
        dtype=torch.int32,
        device=source_values.device,
    )
    extra_starts = torch.empty(
        streams, dtype=torch.uint8, device=source_values.device
    )
    _launch_encode(
        source_values, sign_mantissa, encoded, shifted_encode, extra_starts,
        size, streams,
        precomputed=precomputed,
        logical_numel=logical_numel, fixed_words=fixed_words,
        block_symbols=block_symbols, lanes=lanes, steps=steps, blocks=blocks,
    )

    # Count overflow streams and bytes once.  For a shared buffer these counts
    # are used asynchronously by the allocator; for private fallback they tell
    # us how much exact-size storage to allocate.
    counts = torch.zeros(4, dtype=torch.int32, device=source_values.device)
    count_grid = lambda meta: (triton.cdiv(streams, meta["BLOCK"]),)
    _count_bad_streams_kernel[count_grid](
        extra_starts, counts[:1], counts[1:2], streams, steps,
    )

    buffered = buffer is not None
    if buffer is not None:
        if buffer.capacity_bytes % 4:
            raise ValueError("TensorBuffer capacity must be divisible by 4")
        allocation = buffer.allocate_with_items(counts[1:2], counts[:1], 9)
        metadata = buffer.data.view(torch.int32)
        fallback_buffer = buffer.data
        bad_streams_out = metadata
        bad_starts_out = buffer.data
        fallback_offsets_out = metadata
        metadata_buffer = metadata
        allocation_descriptor = allocation.descriptor
        descriptor = allocation.descriptor
        # counts[0] and counts[1] are the final counts used by the buffered
        # metadata layout; counts[2] and counts[3] are zeroed compaction
        # accumulators.
        final_counts = counts[:1]
        bad_count = counts[2:3]
        fallback_total = counts[3:]
    else:
        count, fallback_size = (int(value) for value in counts[:2].tolist())
        bad_streams_out = torch.empty(count, dtype=torch.int32, device=source_values.device)
        bad_starts_out = torch.empty(count, dtype=torch.uint8, device=source_values.device)
        fallback_offsets_out = torch.empty(
            count, dtype=torch.int32, device=source_values.device
        )
        fallback_buffer = torch.empty(
            fallback_size, dtype=torch.int8, device=source_values.device
        )
        metadata_buffer = bad_streams_out
        # Private compaction does not use the descriptor/final count path, so
        # the zeroed counts[2] and counts[3] act as the atomic accumulators.
        allocation_descriptor = counts[2:3]
        descriptor = None
        final_counts = counts[2:3]
        bad_count = counts[2:3]
        fallback_total = counts[3:]

    _compact_bad_streams(
        extra_starts, bad_streams_out, bad_starts_out, fallback_offsets_out,
        metadata_buffer, allocation_descriptor, final_counts,
        bad_count, fallback_total, streams, steps,
        buffered=buffered,
    )
    _compact_extra(
        source_values, bad_streams_out, bad_starts_out, fallback_offsets_out,
        fallback_buffer, metadata_buffer, allocation_descriptor,
        final_counts, bad_count, size, streams,
        precomputed=precomputed, buffered=buffered,
        logical_numel=logical_numel,
        block_symbols=block_symbols, lanes=lanes, steps=steps,
    )
    return CompressedTensor(
        encoded, size, sign_mantissa,
        offsets=None if buffered else bad_streams_out,
        fallback_starts=None if buffered else bad_starts_out,
        fallback_offsets=None if buffered else fallback_offsets_out,
        fallback_buffer=fallback_buffer,
        fallback_descriptor=descriptor,
        buffer=buffer,
        fallback_count=counts[:1] if buffered else bad_count,
        fallback_used=counts[1:2] if buffered else fallback_total,
        distribution=distribution, center=center, shape=shape,
        layout=StorageLayout.COMPRESSED,
        stream_geometry=stream_geometry,
    )


def encode(
    data: torch.Tensor, distribution: Distribution, buffer: TensorBuffer | None = None, *,
    allow_raw: bool = False,
) -> CompressedTensor:
    """Compress any tensor through the flattened 1D blocked storage mapping."""
    shape = tuple(data.shape)
    source = data.contiguous().view(-1)
    logical_numel = source.numel()
    block_symbols, lanes, _, fixed_words = geometry(distribution)
    # Always use the flattened 1D layout: the codec operates on the
    # contiguous row-major stream and the original shape is restored by
    # reshape on decode.
    blocks = triton.cdiv(logical_numel, block_symbols)
    storage_numel = blocks * block_symbols
    streams = blocks * lanes
    minimum_bytes = storage_numel + (streams * fixed_words + 4) * 4
    if allow_raw and minimum_bytes > logical_numel * data.element_size():
        return CompressedTensor(
            source, logical_numel, buffer=buffer,
            distribution=distribution, shape=shape,
            layout=StorageLayout.RAW,
        )
    sign_mantissa = torch.empty(
        storage_numel, dtype=torch.uint8, device=source.device
    )
    return encode_components(
        source.view(torch.int16), sign_mantissa, storage_numel,
        distribution, buffer, shape, precomputed=False,
        logical_numel=logical_numel,
    )
