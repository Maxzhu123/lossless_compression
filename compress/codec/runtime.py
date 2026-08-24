"""Shared host-side compression and decompression pipeline."""

import torch
import triton

from ..code_storage import CompressedTensor, Distribution, DistType, NoiseLevel
from ..huffman_tables import FIRST_MASK, get_distribution_tables
from ..tensor_buffer import TensorBuffer
from ..trition_kernels import (
    _compact_bad_streams_kernel,
    _compact_extra_kernel,
    _count_bad_streams_kernel,
    _decode_kernel,
    _encode_kernel,
    _estimate_center_kernel,
    _scatter_fallback_kernel,
)


BLOCK_SIZE = 65536
LANES = 256
LANE_BITS = 800
CENTER_SAMPLE_SIZE = 4096


def geometry(distribution: Distribution):
    """Return block size, lanes, steps and fixed words for a codec layout."""
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


def uses_raw_source(size: int, distribution: Distribution) -> bool:
    block_size, lanes, _, fixed_words = geometry(distribution)
    streams = triton.cdiv(size, block_size) * lanes
    return (streams * fixed_words + 4) * 4 > size


def _estimate_center(source, size, *, precomputed):
    sample_size = min(size, CENTER_SAMPLE_SIZE)
    stride = size // sample_size
    center = torch.empty(1, dtype=torch.int32, device=source.device)
    _estimate_center_kernel[(1,)](
        source, center, size, SAMPLE_SIZE=sample_size, STRIDE=stride,
        PRECOMPUTED=precomputed,
    )
    return center


def compress_components(
    source_values,
    sign_mantissa,
    size,
    distribution,
    buffer,
    shape,
    *,
    precomputed,
):
    """Encode BF16 bits or precomputed exponent/sign-mantissa components."""
    block_size, lanes, steps, fixed_words = geometry(distribution)
    blocks = triton.cdiv(size, block_size)
    streams = blocks * lanes
    encode_table, _, _ = get_distribution_tables(distribution)
    center = _estimate_center(source_values, size, precomputed=precomputed)
    encoded = torch.empty(
        streams * fixed_words + 4,
        dtype=torch.int32,
        device=source_values.device,
    )
    extra_starts = torch.empty(
        streams, dtype=torch.uint8, device=source_values.device
    )
    _encode_kernel[(blocks,)](
        source_values, sign_mantissa, encoded, encode_table, center,
        extra_starts, size, streams, PRECOMPUTED=precomputed,
        FIXED_WORDS=fixed_words, BLOCK=block_size,
        N_LANES=lanes, N_STEPS=steps,
    )

    if buffer is not None:
        if buffer.capacity_bytes % 4:
            raise ValueError("TensorBuffer capacity must be divisible by 4")
        counts = torch.zeros(4, dtype=torch.int32, device=source_values.device)
        _count_bad_streams_kernel[(triton.cdiv(streams, 1024),)](
            extra_starts, counts[:1], counts[1:2], streams, steps, BLOCK=1024,
        )
        allocation = buffer.allocate_with_items(counts[1:2], counts[:1], 9)
        metadata = buffer.data.view(torch.int32)
        _compact_bad_streams_kernel[(triton.cdiv(streams, 1024),)](
            extra_starts, metadata, buffer.data, metadata, metadata,
            allocation.descriptor, counts[:1], counts[2:3], counts[3:],
            streams, steps, BUFFERED=True, BLOCK=1024,
        )
        _compact_extra_kernel[(triton.cdiv(streams, 32),)](
            source_values, metadata, buffer.data, metadata, buffer.data,
            metadata, allocation.descriptor, counts[:1], counts[2:3], size,
            BUFFERED=True, PRECOMPUTED=precomputed, BLOCK=block_size,
            N_LANES=lanes, N_STEPS=steps, TILE=32,
        )
        return CompressedTensor(
            encoded, size, sign_mantissa,
            fallback_buffer=buffer.data,
            fallback_descriptor=allocation.descriptor,
            fallback_count=counts[:1], fallback_used=counts[1:2],
            distribution=distribution, center=center, shape=shape,
        )

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
    _compact_bad_streams_kernel[(triton.cdiv(streams, 1024),)](
        extra_starts, bad_streams, bad_starts, fallback_offsets,
        bad_streams, bad_count, bad_count, bad_count, fallback_total,
        streams, steps, BUFFERED=False, BLOCK=1024,
    )
    _compact_extra_kernel[(triton.cdiv(streams, 32),)](
        source_values, bad_streams, bad_starts, fallback_offsets, fallback_data,
        bad_streams, bad_count, bad_count, bad_count, size,
        BUFFERED=False, PRECOMPUTED=precomputed, BLOCK=block_size,
        N_LANES=lanes, N_STEPS=steps, TILE=32,
    )
    return CompressedTensor(
        encoded, size, sign_mantissa,
        bad_streams, bad_starts, fallback_offsets,
        fallback_buffer=fallback_data, fallback_base=0,
        fallback_count=bad_count, fallback_used=fallback_total,
        distribution=distribution, center=center, shape=shape,
    )


def compress_dense(data, distribution, buffer=None):
    """Compress a dense BF16 tensor through the shared codec runtime."""
    shape = tuple(data.shape)
    source = data.contiguous().view(-1)
    size = source.numel()
    if uses_raw_source(size, distribution):
        return CompressedTensor(
            source, size, distribution=distribution, shape=shape,
        )
    sign_mantissa = torch.empty(
        size, dtype=torch.uint8, device=source.device
    )
    return compress_components(
        source.view(torch.int16), sign_mantissa, size,
        distribution, buffer, shape, precomputed=False,
    )


def decode(data: CompressedTensor) -> torch.Tensor:
    """Decode a compressed tensor through the dedicated codec kernels."""
    if data.offsets is None and data.fallback_descriptor is None:
        return data.data.reshape(data.shape)

    _, table, rare_length = get_distribution_tables(data.distribution)
    block_size, lanes, steps, fixed_words = geometry(data.distribution)
    blocks = triton.cdiv(data.size, block_size)
    output = torch.empty(data.size, dtype=torch.int16, device=data.data.device)
    _decode_kernel[(blocks,)](
        data.data, data.sign_mantissa, output, table,
        data.size, blocks * lanes, data.center,
        SKIP_FALLBACK=True, FIRST_MASK=FIRST_MASK, RARE_LENGTH=rare_length,
        BLOCK=block_size, N_LANES=lanes, N_STEPS=steps,
        FIXED_WORDS=fixed_words,
    )
    if data.fallback_descriptor is not None:
        metadata = data.fallback_buffer.view(torch.int32)
        _scatter_fallback_kernel[(triton.cdiv(blocks * lanes, 64),)](
            metadata, data.fallback_buffer, metadata, data.fallback_buffer,
            0, metadata, data.fallback_descriptor, data.fallback_count,
            data.sign_mantissa, output, data.size,
            BUFFERED=True, TILE=64, BLOCK=block_size,
            N_LANES=lanes, N_STEPS=steps,
        )
    else:
        count = data.offsets.numel()
        if count:
            _scatter_fallback_kernel[(triton.cdiv(count, 64),)](
                data.offsets, data.fallback_starts, data.fallback_offsets,
                data.fallback_buffer, data.fallback_base, data.offsets,
                data.offsets, data.fallback_count, data.sign_mantissa,
                output, data.size, BUFFERED=False, TILE=64,
                BLOCK=block_size, N_LANES=lanes, N_STEPS=steps,
            )
    return output.view(torch.bfloat16).reshape(data.shape)
