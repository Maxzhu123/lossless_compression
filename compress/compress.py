import torch
import triton

from .trition_kernels import _estimate_center_kernel, _encode_kernel, _count_bad_streams_kernel, _compact_bad_streams_kernel, _compact_extra_kernel, \
    _scatter_fallback_kernel, _decode_kernel
from .code_storage import CompressedTensor, NoiseLevel, Distribution, DistType
from .huffman_tables import FIRST_MASK, get_distribution_tables
from .tensor_buffer import TensorBuffer


BLOCK_SIZE = 65536              # CLEAN block size; MEDIUM/HIGH use half.
LANES = 256                     # Parallel streams in every block.
LANE_BITS = 800                 # Storage size per stream.

CENTER_SAMPLE_SIZE = 4096       # Number of samples used to estimate mean


def _geometry(distribution: Distribution):
    """ Setup geometry used for compression.
        Each lane is stored in fixed_bits bit budget (returned as 32 bit words).
        Number of symbols in each lane is steps = block_size  / lanes.
        Average number of bits per symbol is (fixed_bits * lanes) / block_size
        Longer sequences use the fallback buffer.
    """
    noise_level = distribution.noise_level
    clean_steps = BLOCK_SIZE // LANES
    lanes = LANES
    if noise_level == NoiseLevel.CLEAN:       # Low noise case
        # Empirical and Laplace use 3.125 bits/symbol.
        block_size = BLOCK_SIZE
        # Gaussian uses one word fewer
        lane_bits = LANE_BITS if distribution.family != DistType.GAUSSIAN else LANE_BITS - 32
    elif noise_level == NoiseLevel.MEDIUM:      # Medium noise case
        # 19 words = 608 bits = 4.75 bits/symbol.
        block_size = BLOCK_SIZE // 2
        clean_steps = clean_steps // 2
        lane_bits = LANE_BITS - 6 * 32
    elif noise_level == NoiseLevel.HIGH:        # High noise case
        # 22 words = 704 bits = 5.5 bits/symbol.
        block_size = BLOCK_SIZE // 2
        clean_steps = clean_steps // 2
        lane_bits = LANE_BITS - 3 * 32
    else:
        raise ValueError(f"Unknown noise level: {noise_level}")

    return block_size, lanes, clean_steps, lane_bits // 32


def _estimate_center(source_bits, size):
    """Estimate the exponent center with one strided-sampling pass."""
    sample_size = min(size, CENTER_SAMPLE_SIZE)
    stride = size // sample_size
    center = torch.empty(1, dtype=torch.int32, device=source_bits.device)
    _estimate_center_kernel[(1,)](
        source_bits,
        center,
        size,
        SAMPLE_SIZE=sample_size,
        STRIDE=stride,
    )
    return center


def compress(
    data: torch.Tensor,
    distribution: Distribution = Distribution(),
    buffer: TensorBuffer | None = None,
) -> CompressedTensor:
    """Losslessly encode the exponent byte of a CUDA bfloat16 tensor.

    The sign bit and seven mantissa bits are retained as one raw byte per
    element. The remaining eight exponent bits are converted to their signed,
    unbiased int8 representation and Huffman-coded by the baseline codec.
    """

    shape = tuple(data.shape)
    source = data.contiguous().view(-1)
    size = source.numel()

    # The fixed exponent payload size depends only on the input size and
    # selected layout, so bypass the codec before allocating or launching it
    # when that payload alone exceeds the available one byte per element.
    block_size, lanes, steps, fixed_words = _geometry(distribution)
    blocks = triton.cdiv(size, block_size)
    streams = blocks * lanes
    payload_words = streams * fixed_words
    if (payload_words + 4) * 4 > size:
        return CompressedTensor(
            source, size,
            buffer=buffer, distribution=distribution, shape=shape,
        )

    # A bfloat16 layout is [sign:1 | exponent:8 | mantissa:7]. Keep the
    # sign/mantissa byte verbatim and use the pre-existing int8 codec for the
    # unbiased exponent. The encode kernel extracts the exponent directly from
    # the int16 bit pattern and emits the sign/mantissa side stream.
    source_bits = source.view(torch.int16)
    sign_mantissa = torch.empty(size, dtype=torch.uint8, device=source.device)

    # Select a distribution-specific codebook. The kernel shifts the table
    # around the sampled exponent center, preserving the copied codec's
    # support for shifted Gaussian and other benchmark cases.
    encode_table, _, _ = get_distribution_tables(distribution)
    center = _estimate_center(source_bits, size)

    # Encode independent streams into a fixed-size payload; the kernel records
    # streams whose remaining values do not fit in that budget.
    # Four padding words make both 64-bit lookaheads safe at the end.
    encoded = torch.empty(
        payload_words + 4, dtype=torch.int32, device=source.device
    )
    # Starts are emitted in pairs, so 254 is the largest possible overflow
    # start.  Use 255 as the no-overflow sentinel instead of spending four
    # bytes per stream on this transient array.
    extra_starts = torch.empty(streams, dtype=torch.uint8, device=source.device)
    _encode_kernel[(blocks,)](
        source_bits, sign_mantissa, encoded,
        encode_table, center, extra_starts,
        size, streams,
        FIXED_WORDS=fixed_words, BLOCK=block_size, N_LANES=lanes, N_STEPS=steps,
    )
    if buffer is not None:
        if buffer.capacity_bytes % 4:
            raise ValueError("TensorBuffer capacity must be divisible by 4")

        # Count overflow on the GPU, reserve one compact device-buffer region,
        # then write metadata and fallback bytes directly through its descriptor.
        # The first pair holds final counts; the second is compaction scratch,
        # avoiding another allocation and zero-fill.  allocate_with_items also
        # computes fallback_bytes + 9 * bad_count inside the allocator kernel,
        # avoiding separate device multiplication and addition launches.
        counts = torch.zeros(4, dtype=torch.int32, device=source.device)
        _count_bad_streams_kernel[(triton.cdiv(streams, 1024),)](
            extra_starts, counts[:1], counts[1:2], streams, steps, BLOCK=1024,
        )
        allocation = buffer.allocate_with_items(counts[1:2], counts[:1], 9)
        metadata_buffer = buffer.data.view(torch.int32)
        _compact_bad_streams_kernel[(triton.cdiv(streams, 1024),)](
            extra_starts, metadata_buffer, buffer.data, metadata_buffer,
            metadata_buffer, allocation.descriptor, counts[:1],
            counts[2:3], counts[3:], streams, steps,
            BUFFERED=True, BLOCK=1024,
        )
        _compact_extra_kernel[(triton.cdiv(streams, 32),)](
            source_bits, metadata_buffer, buffer.data, metadata_buffer,
            buffer.data, metadata_buffer, allocation.descriptor,
            counts[:1], counts[2:3], size,
            BUFFERED=True, BLOCK=block_size, N_LANES=lanes,
            N_STEPS=steps, TILE=32,
        )
        return CompressedTensor(
            encoded,
            size,
            sign_mantissa,
            fallback_buffer=buffer.data,
            fallback_descriptor=allocation.descriptor,
            buffer=buffer,
            fallback_count=counts[:1],
            fallback_used=counts[1:2],
            distribution=distribution,
            center=center,
            shape=shape,
        )

    # Count first, then synchronize once to allocate only the compact metadata
    # and fallback storage required for the non-buffer path.
    counts = torch.zeros(2, dtype=torch.int32, device=source.device)
    bad_count_tensor = counts[:1]
    fallback_total = counts[1:]
    _count_bad_streams_kernel[(triton.cdiv(streams, 1024),)](
        extra_starts, bad_count_tensor, fallback_total,
        streams, steps, BLOCK=1024,
    )
    bad_count, fallback_size = (int(value) for value in counts.tolist())
    bad_streams = torch.empty(bad_count, dtype=torch.int32, device=source.device)
    bad_starts = torch.empty(bad_count, dtype=torch.uint8, device=source.device)
    fallback_offsets = torch.empty(bad_count, dtype=torch.int32, device=source.device)
    fallback_data = torch.empty(fallback_size, dtype=torch.int8, device=source.device)
    fallback_base = 0
    bad_count_tensor.zero_()
    fallback_total.zero_()

    _compact_bad_streams_kernel[(triton.cdiv(streams, 1024),)](
        extra_starts, bad_streams, bad_starts, fallback_offsets,
        bad_streams, bad_count_tensor, bad_count_tensor, bad_count_tensor,
        fallback_total, streams,
        steps, BUFFERED=False, BLOCK=1024,
    )

    # TILE-based kernel; beyond the active bad_count range, programs exit on
    # the GPU before doing any fallback work.
    _compact_extra_kernel[(triton.cdiv(streams, 32),)](
        source_bits, bad_streams, bad_starts,
        fallback_offsets, fallback_data,
        bad_streams, bad_count_tensor, bad_count_tensor, bad_count_tensor,
        size,
        BUFFERED=False, BLOCK=block_size, N_LANES=lanes, N_STEPS=steps, TILE=32,
    )
    return CompressedTensor(
        encoded, size, sign_mantissa,
        bad_streams, bad_starts, fallback_offsets,
        fallback_buffer=fallback_data, fallback_base=fallback_base,
        buffer=buffer,
        fallback_count=bad_count_tensor, fallback_used=fallback_total,
        distribution=distribution, center=center, shape=shape,
    )


def decompress(data: CompressedTensor) -> torch.Tensor:
    """Decode a tensor produced by :func:`compress`."""
    if data.offsets is None and data.fallback_descriptor is None:
        return data.data.reshape(data.shape)

    _, decode_table, rare_length = get_distribution_tables(data.distribution)
    block_size, lanes, steps, fixed_words = _geometry(data.distribution)
    blocks = triton.cdiv(data.size, block_size)
    out_bits = torch.empty(data.size, dtype=torch.int16, device=data.data.device)
    _decode_kernel[(blocks,)](
        data.data, data.sign_mantissa,
        out_bits, decode_table,
        data.size, blocks * lanes, data.center,
        SKIP_FALLBACK=True, FIRST_MASK=FIRST_MASK, RARE_LENGTH=rare_length,
        BLOCK=block_size, N_LANES=lanes, N_STEPS=steps, FIXED_WORDS=fixed_words,
    )
    if data.fallback_descriptor is not None:
        metadata_buffer = data.fallback_buffer.view(torch.int32)
        _scatter_fallback_kernel[(triton.cdiv(blocks * lanes, 64),)](
            metadata_buffer, data.fallback_buffer, metadata_buffer, data.fallback_buffer,
            0, metadata_buffer, data.fallback_descriptor,
            data.fallback_count, data.sign_mantissa,
            out_bits, data.size,
            BUFFERED=True, TILE=64, BLOCK=block_size, N_LANES=lanes, N_STEPS=steps,
        )
    else:
        n_bad = data.offsets.numel()
        if n_bad > 0:
            _scatter_fallback_kernel[(triton.cdiv(n_bad, 64),)](
                data.offsets, data.fallback_starts, data.fallback_offsets, data.fallback_buffer,
                data.fallback_base, data.offsets, data.offsets, data.fallback_count, data.sign_mantissa,
                out_bits, data.size,
                BUFFERED=False, TILE=64, BLOCK=block_size, N_LANES=lanes, N_STEPS=steps,
            )
    return out_bits.view(torch.bfloat16).reshape(data.shape)
