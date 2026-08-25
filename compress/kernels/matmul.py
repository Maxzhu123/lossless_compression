"""Triton decoding support for matrix-tiled compressed weights."""

import triton
from triton import language as tl

from ..codec.primitives import decode_symbol, pack_bf16


MATRIX_TILE = 256


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=8, num_stages=2),
    ],
    key=["N", "K", "N_STEPS"],
)
@triton.jit
def split_matrix_components_kernel(
    source_bits, exponents, sign_mantissa,
    N: tl.constexpr, K: tl.constexpr, K_TILE_BLOCKS: tl.constexpr,
    N_STEPS: tl.constexpr,
):
    """Map native matrix values into padded tiled exponent and side-byte planes."""
    block = tl.program_id(0)
    lane = tl.arange(0, 256)
    n_tile = block // K_TILE_BLOCKS
    k_tile = block % K_TILE_BLOCKS
    logical_k = k_tile * 256 + lane
    for step in tl.range(0, N_STEPS):
        logical_n = n_tile * N_STEPS + step
        valid = (logical_n < N) & (logical_k < K)
        bits = tl.load(
            source_bits + logical_n * K + logical_k, mask=valid, other=0,
        ).to(tl.int32)
        output_offset = block * N_STEPS * 256 + step * 256 + lane
        exponent = (bits >> 7) & 0xFF
        sm = (bits & 0x7F) | ((bits >> 8) & 0x80)
        tl.store(exponents + output_offset, exponent.to(tl.uint8))
        tl.store(sign_mantissa + output_offset, sm.to(tl.uint8))


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=8, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=3, maxnreg=64),
    ],
    key=["N", "K", "MATRIX_STEPS", "FIXED_WORDS"],
)
@triton.jit
def decode_matrix_kernel(
    encoded, sign_mantissa, decode_table,
    fallback_buffer, fallback_descriptor, fallback_count,
    stream_starts, stream_offsets, output, center,
    N: tl.constexpr, K: tl.constexpr, K_TILE_BLOCKS: tl.constexpr,
    MATRIX_STEPS: tl.constexpr, N_STREAMS: tl.constexpr,
    FIXED_WORDS: tl.constexpr, FALLBACK_BASE: tl.constexpr,
    BUFFERED: tl.constexpr, FIRST_MASK: tl.constexpr,
    RARE_LENGTH: tl.constexpr,
):
    """Decode one storage block directly into contiguous ``[N, K]`` matrix order."""
    storage_block = tl.program_id(0)
    lane = tl.arange(0, 256)
    stream = storage_block * 256 + lane
    n_tile = storage_block // K_TILE_BLOCKS
    k_tile = storage_block % K_TILE_BLOCKS
    logical_k = k_tile * 256 + lane
    valid_k = logical_k < K
    word = tl.zeros((256,), tl.int32)
    shift = tl.zeros((256,), tl.int32)
    word0 = tl.load(encoded + stream).to(tl.uint32).to(tl.uint64)
    word1 = tl.load(encoded + N_STREAMS + stream).to(tl.uint32).to(tl.uint64)
    window = word0 | (word1 << 32)
    center_value = tl.load(center).to(tl.int32)
    start = tl.load(stream_starts + stream).to(tl.int32)
    extra_offset = tl.load(stream_offsets + stream).to(tl.int32)
    if BUFFERED:
        allocation_base = tl.load(fallback_descriptor).to(tl.int32)
        bad_count = tl.load(fallback_count).to(tl.int32)
        fallback_base = allocation_base + 9 * bad_count
    else:
        fallback_base = FALLBACK_BASE

    for step in tl.range(0, MATRIX_STEPS, 2):
        value, length = decode_symbol(
            window >> shift, decode_table, center_value,
            FIRST_MASK, RARE_LENGTH,
        )
        use_fallback = (start != 255) & (step >= start)
        fallback_value = tl.load(
            fallback_buffer + fallback_base + extra_offset + step - start,
            mask=use_fallback, other=0,
        ).to(tl.int32)
        value = tl.where(use_fallback, fallback_value, value)
        storage_offset = storage_block * MATRIX_STEPS * 256 + step * 256 + lane
        sm = tl.load(sign_mantissa + storage_offset)
        bits = pack_bf16(value, sm).to(tl.int16).to(tl.bfloat16, bitcast=True)
        logical_n = n_tile * MATRIX_STEPS + step
        tl.store(
            output + logical_n * K + logical_k, bits,
            mask=valid_k & (logical_n < N),
        )

        shift1 = shift + length
        value1, length1 = decode_symbol(
            window >> shift1, decode_table, center_value,
            FIRST_MASK, RARE_LENGTH,
        )
        use_fallback1 = (start != 255) & (step + 1 >= start)
        fallback_value1 = tl.load(
            fallback_buffer + fallback_base + extra_offset + step + 1 - start,
            mask=use_fallback1, other=0,
        ).to(tl.int32)
        value1 = tl.where(use_fallback1, fallback_value1, value1)
        sm1 = tl.load(sign_mantissa + storage_offset + 256)
        bits1 = pack_bf16(value1, sm1).to(tl.int16).to(tl.bfloat16, bitcast=True)
        tl.store(
            output + (logical_n + 1) * K + logical_k, bits1,
            mask=valid_k & (logical_n + 1 < N),
        )

        next_shift = shift1 + length1
        crosses_word = next_shift >= 32
        word2 = tl.load(
            encoded + tl.minimum(word + 2, FIXED_WORDS - 1) * N_STREAMS + stream,
            mask=crosses_word, other=0,
        ).to(tl.uint32).to(tl.uint64)
        next_window = (window >> 32) | (word2 << 32)
        window = tl.where(crosses_word, next_window, window)
        word += crosses_word
        shift = tl.where(crosses_word, next_shift - 32, next_shift)
