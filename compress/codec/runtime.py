"""Shared host-side compression and decompression pipeline."""

from dataclasses import replace
import torch
import triton

from ..code_storage import (
    CompressedTensor, Distribution, DistType, NoiseLevel, StorageLayout,
)
from ..huffman_tables import FIRST_BITS, FIRST_MASK, get_distribution_tables
from ..tensor_buffer import TensorBuffer
from ..trition_kernels import (
    _compact_bad_streams_kernel,
    _shift_encoding_table_kernel,
    _shift_decoding_table_kernel,
    _compact_matrix_components_extra_kernel,
    _compact_matrix_extra_kernel,
    _count_bad_streams_kernel,
    _decode_matrix_kernel,
    _encode_matrix_components_kernel,
    _encode_matrix_kernel,
    _estimate_center_kernel,
    _scatter_blocked_fallback_kernel,
)
BLOCK_SIZE = 65536              # CLEAN block size; MEDIUM/HIGH use half.
LANES = 256                     # Parallel streams in every block.
LANE_BITS = 800                 # Storage size per stream.

CENTER_SAMPLE_SIZE = 4096       # Number of samples used to estimate mean


def geometry(distribution: Distribution):
    """Setup geometry used for compression.
        Each lane is stored in fixed_bits bit budget (returned as 32 bit words).
        Number of symbols in each lane is steps = block_size  / lanes.
        Average number of bits per symbol is (fixed_bits * lanes) / block_size
        Longer sequences use the fallback buffer.
    """
    clean_steps = BLOCK_SIZE // LANES
    lanes = LANES
    if distribution.noise_level == NoiseLevel.CLEAN:
        block_size = BLOCK_SIZE
        lane_bits = (
            LANE_BITS
            if distribution.family != DistType.GAUSSIAN
            else LANE_BITS - 32
        )
    elif distribution.noise_level == NoiseLevel.MEDIUM:
        block_size = BLOCK_SIZE // 2
        clean_steps //= 2
        lane_bits = LANE_BITS - 6 * 32
    elif distribution.noise_level == NoiseLevel.HIGH:
        block_size = BLOCK_SIZE // 2
        clean_steps //= 2
        lane_bits = LANE_BITS - 3 * 32
    else:
        raise ValueError(f"Unknown noise level: {distribution.noise_level}")
    return block_size, lanes, clean_steps, lane_bits // 32


def _estimate_center(source, size, *, precomputed, ignore_zero=False):
    """Estimate the exponent center from strided samples on the GPU."""
    sample_size = min(size, CENTER_SAMPLE_SIZE)
    stride = size // sample_size
    center = torch.empty(1, dtype=torch.int32, device=source.device)
    _estimate_center_kernel[(1,)](
        source, center, size, SAMPLE_SIZE=sample_size, STRIDE=stride,
        PRECOMPUTED=precomputed, IGNORE_ZERO=ignore_zero,
    )
    return center


def _launch_encode(
    source_values, sign_mantissa, encoded, encode_table, center, extra_starts,
    size, streams, *,
    precomputed, logical_numel,
    fixed_words, block_size, lanes, steps, blocks,
):
    """Launch the 1D encode kernel for raw BF16 or precomputed components."""
    if precomputed:
        _encode_matrix_components_kernel[(blocks,)](
            source_values, sign_mantissa, encoded, encode_table, center,
            extra_starts, size, streams, LOGICAL_NUMEL=logical_numel,
            FIXED_WORDS=fixed_words, BLOCK=block_size,
            N_LANES=lanes, N_STEPS=steps,
        )
    else:
        _encode_matrix_kernel[(blocks,)](
            source_values, sign_mantissa, encoded, encode_table, center,
            extra_starts, size, streams, LOGICAL_NUMEL=logical_numel,
            FIXED_WORDS=fixed_words, BLOCK=block_size,
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
    _compact_bad_streams_kernel[(triton.cdiv(streams, 1024),)](
        extra_starts, bad_streams_out, bad_starts_out, fallback_offsets_out,
        metadata_buffer, allocation_descriptor, final_counts,
        bad_count, fallback_total, streams, steps,
        BUFFERED=buffered, BLOCK=1024,
    )


def _compact_extra(
    source_values,
    bad_streams, bad_starts, fallback_offsets, fallback_data,
    metadata_buffer, allocation_descriptor, final_counts, bad_count,
    size, streams, *,
    precomputed, buffered,
    logical_numel,
    block_size, lanes, steps,
):
    """Compact fallback tail values for a buffered or private fallback path."""
    compact_grid = (triton.cdiv(streams, 32),)
    if precomputed:
        _compact_matrix_components_extra_kernel[compact_grid](
            source_values, bad_streams, bad_starts, fallback_offsets,
            fallback_data, metadata_buffer, allocation_descriptor,
            final_counts, bad_count, size,
            BUFFERED=buffered, LOGICAL_NUMEL=logical_numel,
            BLOCK=block_size,
            N_LANES=lanes, N_STEPS=steps, TILE=32,
        )
    else:
        _compact_matrix_extra_kernel[compact_grid](
            source_values, bad_streams, bad_starts, fallback_offsets,
            fallback_data, metadata_buffer, allocation_descriptor,
            final_counts, bad_count, size,
            BUFFERED=buffered, LOGICAL_NUMEL=logical_numel,
            BLOCK=block_size,
            N_LANES=lanes, N_STEPS=steps, TILE=32,
        )


def compress_components(
    source_values,
    sign_mantissa,
    size,
    distribution,
    buffer: TensorBuffer | None,
    shape,
    *,
    precomputed,
    center=None,
    logical_numel,
):
    """Encode BF16 bits or split components into a ``CompressedTensor``.

    Args:
        source_values: Int16 BF16 bits, or raw uint8 exponents if ``precomputed``.
        sign_mantissa: Output side-byte stream, or precomputed side bytes.
        size: Padded codec element count (storage element count).
        distribution: codebook and stream geometry.
        buffer: Optional shared fallback arena; shape: original tensor shape.
        precomputed: ``False`` extracts both fields from BF16 bits; ``True``
            consumes raw exponent bytes plus existing side bytes from a fused op.
        logical_numel: Flattened logical element count (1D storage mapping).
    """
    # Geometry fixes the independent stream count and per-stream bit budget.
    block_size, lanes, steps, fixed_words = geometry(distribution)
    blocks = triton.cdiv(size, block_size)
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
        source_values, sign_mantissa, encoded, shifted_encode, center,
        extra_starts, size, streams,
        precomputed=precomputed,
        logical_numel=logical_numel, fixed_words=fixed_words,
        block_size=block_size, lanes=lanes, steps=steps, blocks=blocks,
    )

    if buffer is not None:
        # Keep fallback metadata and bytes inside one descriptor-backed arena region.
        if buffer.capacity_bytes % 4:
            raise ValueError("TensorBuffer capacity must be divisible by 4")
        counts = torch.zeros(4, dtype=torch.int32, device=source_values.device)
        _count_bad_streams_kernel[(triton.cdiv(streams, 1024),)](
            extra_starts, counts[:1], counts[1:2], streams, steps, BLOCK=1024,
        )
        allocation = buffer.allocate_with_items(counts[1:2], counts[:1], 9)
        metadata = buffer.data.view(torch.int32)
        _compact_bad_streams(
            extra_starts, metadata, buffer.data, metadata, metadata,
            allocation.descriptor, counts[:1], counts[2:3], counts[3:],
            streams, steps, buffered=True,
        )
        _compact_extra(
            source_values, metadata, buffer.data, metadata, buffer.data,
            metadata, allocation.descriptor, counts[:1], counts[2:3],
            size, streams,
            precomputed=precomputed, buffered=True,
            logical_numel=logical_numel,
            block_size=block_size, lanes=lanes, steps=steps,
        )
        result = CompressedTensor(
            encoded, size, sign_mantissa,
            fallback_buffer=buffer.data,
            fallback_descriptor=allocation.descriptor,
            buffer=buffer,
            fallback_count=counts[:1], fallback_used=counts[1:2],
            distribution=distribution, center=center, shape=shape,
        )
        return result

    # Private fallback storage synchronizes once to allocate exact-size tensors.
    counts = torch.zeros(2, dtype=torch.int32, device=source_values.device)
    bad_count, fallback_total = counts[:1], counts[1:]
    _count_bad_streams_kernel[(triton.cdiv(streams, 1024),)](
        extra_starts, bad_count, fallback_total, streams, steps, BLOCK=1024,
    )
    count, fallback_size = (int(value) for value in counts.tolist())
    bad_streams = torch.empty(count, dtype=torch.int32, device=source_values.device)
    bad_starts = torch.empty(count, dtype=torch.uint8, device=source_values.device)
    fallback_offsets = torch.empty(
        count, dtype=torch.int32, device=source_values.device
    )
    fallback_data = torch.empty(
        fallback_size, dtype=torch.int8, device=source_values.device
    )
    bad_count.zero_()
    fallback_total.zero_()
    _compact_bad_streams(
        extra_starts, bad_streams, bad_starts, fallback_offsets,
        bad_streams, bad_count, bad_count, bad_count, fallback_total,
        streams, steps, buffered=False,
    )
    _compact_extra(
        source_values, bad_streams, bad_starts, fallback_offsets,
        fallback_data, bad_streams, bad_count, bad_count, bad_count,
        size, streams,
        precomputed=precomputed, buffered=False,
        logical_numel=logical_numel,
        block_size=block_size, lanes=lanes, steps=steps,
    )
    result = CompressedTensor(
        encoded, size, sign_mantissa,
        bad_streams, bad_starts, fallback_offsets,
        fallback_buffer=fallback_data, fallback_base=0,
        buffer=buffer,
        fallback_count=bad_count, fallback_used=fallback_total,
        distribution=distribution, center=center, shape=shape,
    )
    return result


def compress_dense(
    data, distribution, buffer: TensorBuffer | None = None, *,
    allow_raw=True,
):
    """Compress any tensor through the flattened 1D blocked storage mapping."""
    shape = tuple(data.shape)
    source = data.contiguous().view(-1)
    logical_numel = source.numel()
    block_size, lanes, steps, fixed_words = geometry(distribution)
    # Always use the flattened 1D layout: the codec operates on the
    # contiguous row-major stream and the original shape is restored by
    # reshape on decode. Flattening removes the old 2D tile padding and lets
    # every tensor use the single-tile fast path in the kernels.
    blocks = triton.cdiv(logical_numel, block_size)
    storage_numel = blocks * block_size
    streams = blocks * lanes
    minimum_bytes = storage_numel + (streams * fixed_words + 4) * 4
    if allow_raw and minimum_bytes > logical_numel * data.element_size():
        return CompressedTensor(
            source, logical_numel, buffer=buffer,
            distribution=distribution, shape=shape, layout=StorageLayout.RAW,
        )
    sign_mantissa = torch.empty(
        storage_numel, dtype=torch.uint8, device=source.device
    )
    result = compress_components(
        source.view(torch.int16), sign_mantissa, storage_numel,
        distribution, buffer, shape, precomputed=False,
        logical_numel=logical_numel,
    )
    return replace(result, shape=shape, layout=StorageLayout.COMPRESSED)


def decode_matrix_dense(data: CompressedTensor) -> torch.Tensor:
    """Decode blocked storage directly into its original logical tensor shape."""
    logical_numel = data.logical_numel
    _, decode_table, rare_length = get_distribution_tables(data.distribution)
    shifted_decode = torch.empty(
        1 << FIRST_BITS, dtype=torch.int32, device=data.data.device,
    )
    _shift_decoding_table_kernel[(1,)](
        decode_table, data.center, shifted_decode,
        TABLE_SIZE=1 << FIRST_BITS, BLOCK=1 << FIRST_BITS,
    )
    block_size, lanes, steps, fixed_words = geometry(data.distribution)
    blocks = triton.cdiv(data.size, block_size)
    streams = blocks * lanes
    output = torch.empty(logical_numel, dtype=torch.int16, device=data.data.device)
    # Use the optimized normal-Triton decode kernel (no Gluon needed).
    _decode_matrix_kernel[(blocks,)](
        data.data, data.sign_mantissa, output, shifted_decode,
        data.size, streams, data.center,
        LOGICAL_NUMEL=logical_numel,
        FIRST_MASK=FIRST_MASK, RARE_LENGTH=rare_length,
        BLOCK=block_size, N_LANES=lanes, N_STEPS=steps,
        FIXED_WORDS=fixed_words,
        ON_DEMAND=logical_numel > 100_000_000,
    )
    scatter_meta = dict(
        LOGICAL_NUMEL=logical_numel,
        TILE=64, BLOCK=block_size, N_LANES=lanes, N_STEPS=steps,
    )
    if data.fallback_descriptor is not None:
        metadata = data.fallback_buffer.view(torch.int32)
        _scatter_blocked_fallback_kernel[(triton.cdiv(streams, 64),)](
            metadata, data.fallback_buffer, metadata, data.fallback_buffer, 0,
            metadata, data.fallback_descriptor, data.fallback_count,
            data.sign_mantissa, output, data.size,
            BUFFERED=True, **scatter_meta,
        )
    elif data.offsets.numel():
        _scatter_blocked_fallback_kernel[(triton.cdiv(data.offsets.numel(), 64),)](
            data.offsets, data.fallback_starts, data.fallback_offsets,
            data.fallback_buffer, data.fallback_base, data.offsets,
            data.offsets, data.fallback_count, data.sign_mantissa,
            output, data.size, BUFFERED=False, **scatter_meta,
        )
    return output.view(torch.bfloat16).reshape(data.shape)
