import torch
import triton

from trition_kernels import _estimate_center_kernel, _encode_kernel, _compact_bad_streams_kernel, _compact_extra_kernel, \
    _scatter_fallback_kernel, _decode_kernel
from .code_storage import CompressedTensor, CompressionLayout, Distribution
from .huffman_tables import FIRST_MASK, get_distribution_tables
from .tensor_buffer import TensorBuffer


BLOCK_SIZE = 32768              # Number of elements encoded per block
LANES = 128                     # Number of streams processed in parallel
STEPS = BLOCK_SIZE // LANES     # Number of symbols per stream

FIXED_BITS = 832                # Size of stream. Excess bits go into fallback.
FIXED_WORDS = FIXED_BITS // 32  # Number of 32-bit words in stream.

CENTER_SAMPLE_SIZE = 4096       # Number of samples used to estimate mean


def _geometry(layout: CompressionLayout):
    """ Set the shape of the fixed size payload.
        Average number of bits per symbol is lanes / (32*fixed_words) = lanes / fixed_bits
    """
    if layout == CompressionLayout.MEDIUM:
        # 32768-element blocks, 256 lanes, 128 steps/stream, 19 words = 608
        # bits/stream. Allowed average is
        # 608 / 128 = 4.75 bits/symbol
        return BLOCK_SIZE, LANES * 2, STEPS // 2, (FIXED_WORDS + 12) // 2
    if layout == CompressionLayout.HIGH:
        # 32768-element blocks, 256 lanes, 128 steps/stream, 22 words = 704
        # bits/stream. Allowed average is 704 / 128 = 5.5 bits/symbol
        return BLOCK_SIZE, LANES * 2, STEPS // 2, FIXED_WORDS // 2 + 9
    # 65536-element blocks, 256 lanes, 256 steps/stream, 26 words = 832
    # bits/stream. Allowed average is 832 / 256 = 3.25 bits/symbol
    return BLOCK_SIZE * 2, LANES * 2, STEPS, FIXED_WORDS


def _estimate_center(source_bits, size):
    """Estimate the exponent center with one strided-sampling pass."""
    sample_size = min(size, CENTER_SAMPLE_SIZE)
    if sample_size == 0:
        return 0
    stride = max(size // sample_size, 1)
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
    layout: CompressionLayout = CompressionLayout.CLEAN,
    distribution: Distribution = Distribution.standard(),
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

    # RAW bypasses the codec while retaining the same compressed-tensor API.
    if layout == CompressionLayout.RAW or size == 0:
        return CompressedTensor(
            data.dtype,
            source,
            size,
            layout=layout,
            distribution=distribution,
            shape=shape,
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
    block_size, lanes, steps, fixed_words = _geometry(layout)
    blocks = triton.cdiv(size, block_size)
    streams = blocks * lanes
    payload_words = streams * fixed_words
    # Four padding words make both 64-bit lookaheads safe at the end.
    encoded = torch.empty(
        payload_words + 4, dtype=torch.int32, device=source.device
    )
    extra_starts = torch.empty(streams, dtype=torch.int32, device=source.device)
    _encode_kernel[(blocks,)](
        source_bits,
        sign_mantissa,
        encoded,
        encode_table,
        center,
        extra_starts,
        size,
        streams,
        FIXED_WORDS=fixed_words,
        BLOCK=block_size,
        N_LANES=lanes,
        N_STEPS=steps,
    )
    if encoded.nbytes > size:
        # For very small inputs the fixed-size payload is larger than the
        # source tensor, so return the original BF16 data instead.
        return CompressedTensor(
            data.dtype,
            source,
            size,
            layout=layout,
            distribution=distribution,
            shape=shape,
        )
    bad_count_tensor = torch.zeros(1, dtype=torch.int32, device=source.device)

    if buffer is not None:
        # Path backed by a shared TensorBuffer.  Metadata and fallback storage
        # are carved from the persistent buffer; no host scalar reads are
        # needed.
        bad_streams, _ = buffer.allocate(streams * 4, torch.int32)
        bad_starts, _ = buffer.allocate(streams * 4, torch.int32)
        fallback_offsets, _ = buffer.allocate(streams * 4, torch.int32)
        fallback_total, _ = buffer.allocate(4, torch.int32)
        fallback_data, fallback_base = buffer.allocate(
            streams * steps, dtype=torch.int8
        )
        fallback_total.zero_()
    else:
        # No-buffer fallback: allocate the same metadata and fallback arrays as
        # ordinary tensors, and point the kernels at them with base offset 0.
        # This preserves the old non-buffer behaviour while keeping the same
        # GPU kernels.  A host scalar read is only used here to size the
        # private fallback_data tensor.
        bad_streams = torch.empty(streams, dtype=torch.int32, device=source.device)
        bad_starts = torch.empty(streams, dtype=torch.int32, device=source.device)
        fallback_offsets = torch.empty(streams, dtype=torch.int32, device=source.device)
        fallback_total = torch.zeros(1, dtype=torch.int32, device=source.device)
        fallback_base = 0

    _compact_bad_streams_kernel[(triton.cdiv(streams, 1024),)](
        extra_starts,
        bad_streams,
        bad_starts,
        fallback_offsets,
        bad_count_tensor,
        fallback_total,
        streams,
        steps,
        BLOCK=1024,
    )

    if buffer is None:
        # Exact fallback allocation for the non-buffer path, matching the old
        # dynamic-allocation behaviour.  In the buffer path fallback_data was
        # already reserved at full capacity.
        fallback_size = int(fallback_total.item())
        fallback_data = torch.empty(
            fallback_size, dtype=torch.int8, device=source.device
        )

    # TILE-based kernel; beyond the active bad_count range, programs exit on
    # the GPU before doing any fallback work.
    _compact_extra_kernel[(triton.cdiv(streams, 32),)](
        source_bits,
        bad_streams,
        bad_starts,
        fallback_offsets,
        fallback_data,
        bad_count_tensor,
        size,
        BLOCK=block_size,
        N_LANES=lanes,
        N_STEPS=steps,
        TILE=32,
    )
    return CompressedTensor(
        data.dtype,
        encoded,
        size,
        sign_mantissa,
        bad_streams,
        bad_starts,
        fallback_offsets,
        None,
        fallback_buffer=buffer.data if buffer is not None else fallback_data,
        fallback_base=fallback_base,
        fallback_count=bad_count_tensor,
        fallback_used=fallback_total,
        layout=layout,
        distribution=distribution,
        center=center,
        shape=shape,
    )


def decompress(data: CompressedTensor) -> torch.Tensor:
    """Decode a tensor produced by :func:`compress`."""
    if data.offsets is None:
        return data.data.reshape(data.shape)

    _, decode_table, rare_length = get_distribution_tables(data.distribution)
    block_size, lanes, steps, fixed_words = _geometry(data.layout)
    blocks = triton.cdiv(data.size, block_size)
    out_bits = torch.empty(data.size, dtype=torch.int16, device=data.data.device)
    has_fallback = data.fallback_data is not None or data.fallback_buffer is not None
    _decode_kernel[(blocks,)](
        data.data,
        data.sign_mantissa,
        out_bits,
        decode_table,
        data.size,
        blocks * lanes,
        data.center,
        SKIP_FALLBACK=has_fallback,
        FIRST_MASK=FIRST_MASK,
        RARE_LENGTH=rare_length,
        BLOCK=block_size,
        N_LANES=lanes,
        N_STEPS=steps,
        FIXED_WORDS=fixed_words,
    )
    if has_fallback:
        if data.fallback_buffer is not None:
            fallback_buffer = data.fallback_buffer
            fallback_base = data.fallback_base
            fallback_count = data.fallback_count
            n_bad = data.offsets.numel()
        else:
            # Compatibility with non-buffer compressed tensors.
            fallback_buffer = data.fallback_data
            fallback_base = 0
            fallback_count = torch.full(
                (1,),
                data.offsets.numel(),
                dtype=torch.int32,
                device=data.data.device,
            )
            n_bad = data.offsets.numel()
        _scatter_fallback_kernel[(triton.cdiv(n_bad, 64),)](
            data.offsets,
            data.fallback_starts,
            data.fallback_offsets,
            fallback_buffer,
            fallback_base,
            fallback_count,
            data.sign_mantissa,
            out_bits,
            data.size,
            n_bad,
            TILE=64,
            BLOCK=block_size,
            N_LANES=lanes,
            N_STEPS=steps,
        )
    return out_bits.view(torch.bfloat16).reshape(data.shape)
