"""One-pass fused pointwise kernel for two compressed BF16 operands.

Computes ``output = alpha * a + b`` without materialising either operand as a
dense tensor.  The matrix kernel handles private and buffered fallback storage
directly, so a single launch covers fixed streams, overflow tails, dense output,
and compressed component output.
"""

import triton
from triton import language as tl

from ..codec.autotune import DECODE_AUTOTUNE_CONFIGS
from ..codec.primitives import decode_symbol, pack_bf16
from .pointwise import (
    _pointwise_location,
    _store_result,
)


DUAL_DECODE_AUTOTUNE_CONFIGS = [
    triton.Config({}, num_warps=8, num_stages=3),
]

@triton.jit
def _scalar_mul_add_compressed_compressed_matrix_impl(
    a_encoded, a_sign_mantissa, a_decode_table, a_center,
    a_bad_streams, a_bad_starts, a_fallback_offsets, a_metadata,
    a_descriptor, a_fallback_count,
    a_fallback_buffer, a_fallback_base,
    b_encoded, b_sign_mantissa, b_decode_table, b_center,
    b_bad_streams, b_bad_starts, b_fallback_offsets, b_metadata,
    b_descriptor, b_fallback_count,
    b_fallback_buffer, b_fallback_base,
    output, auxiliary, alpha,
    n_elements, n_streams,
    BUFFERED: tl.constexpr,
    OUTPUT_POLICY: tl.constexpr,
    MATRIX_N: tl.constexpr, MATRIX_K: tl.constexpr,
    MATRIX_NUMEL: tl.constexpr, K_TILE_BLOCKS: tl.constexpr,
    FIRST_MASK: tl.constexpr, RARE_LENGTH: tl.constexpr,
    BLOCK: tl.constexpr, N_LANES: tl.constexpr,
    N_STEPS: tl.constexpr, FIXED_WORDS: tl.constexpr,
):
    """One-pass matrix fused kernel: alpha * a + b, including fallbacks."""
    block = tl.program_id(0)
    lanes = tl.arange(0, N_LANES)
    lane_index = block * N_LANES + lanes
    stream = lane_index

    a_count = tl.load(a_fallback_count).to(tl.int32)
    b_count = tl.load(b_fallback_count).to(tl.int32)

    a_start = tl.full((N_LANES,), N_STEPS, tl.int32)
    b_start = tl.full((N_LANES,), N_STEPS, tl.int32)
    a_offset = tl.zeros((N_LANES,), tl.int32)
    b_offset = tl.zeros((N_LANES,), tl.int32)

    if BUFFERED:
        a_base = tl.load(a_descriptor).to(tl.int32)
        a_base_words = a_base // 4
        a_real_base = a_base + 9 * a_count
        for i in tl.range(0, a_count):
            a_sid = tl.load(a_metadata + a_base_words + i).to(tl.int32)
            a_start_i = tl.load(a_bad_starts + a_base + 8 * a_count + i).to(tl.int32)
            a_off = tl.load(a_metadata + a_base_words + a_count + i).to(tl.int32)
            is_a = a_sid == stream
            a_start = tl.where(is_a, a_start_i, a_start)
            a_offset = tl.where(is_a, a_off, a_offset)
    else:
        a_real_base = a_fallback_base
        for i in tl.range(0, a_count):
            a_sid = tl.load(a_bad_streams + i).to(tl.int32)
            a_start_i = tl.load(a_bad_starts + i).to(tl.int32)
            a_off = tl.load(a_fallback_offsets + i).to(tl.int32)
            is_a = a_sid == stream
            a_start = tl.where(is_a, a_start_i, a_start)
            a_offset = tl.where(is_a, a_off, a_offset)

    if BUFFERED:
        b_base = tl.load(b_descriptor).to(tl.int32)
        b_base_words = b_base // 4
        b_real_base = b_base + 9 * b_count
        for i in tl.range(0, b_count):
            b_sid = tl.load(b_metadata + b_base_words + i).to(tl.int32)
            b_start_i = tl.load(b_bad_starts + b_base + 8 * b_count + i).to(tl.int32)
            b_off = tl.load(b_metadata + b_base_words + b_count + i).to(tl.int32)
            is_b = b_sid == stream
            b_start = tl.where(is_b, b_start_i, b_start)
            b_offset = tl.where(is_b, b_off, b_offset)
    else:
        b_real_base = b_fallback_base
        for i in tl.range(0, b_count):
            b_sid = tl.load(b_bad_streams + i).to(tl.int32)
            b_start_i = tl.load(b_bad_starts + i).to(tl.int32)
            b_off = tl.load(b_fallback_offsets + i).to(tl.int32)
            is_b = b_sid == stream
            b_start = tl.where(is_b, b_start_i, b_start)
            b_offset = tl.where(is_b, b_off, b_offset)

    a_word = tl.zeros((N_LANES,), tl.int32)
    a_shift = tl.zeros((N_LANES,), tl.int32)
    b_word = tl.zeros((N_LANES,), tl.int32)
    b_shift = tl.zeros((N_LANES,), tl.int32)

    a_word0 = tl.load(a_encoded + a_word * n_streams + stream)
    a_word1 = tl.load(a_encoded + (a_word + 1) * n_streams + stream)
    a_window = a_word0.to(tl.uint32).to(tl.uint64)
    a_window |= a_word1.to(tl.uint32).to(tl.uint64) << 32

    b_word0 = tl.load(b_encoded + b_word * n_streams + stream)
    b_word1 = tl.load(b_encoded + (b_word + 1) * n_streams + stream)
    b_window = b_word0.to(tl.uint32).to(tl.uint64)
    b_window |= b_word1.to(tl.uint32).to(tl.uint64) << 32

    a_center_value = tl.load(a_center).to(tl.int32)
    b_center_value = tl.load(b_center).to(tl.int32)
    alpha_value = tl.load(alpha).to(tl.float32)

    n_tile = block // K_TILE_BLOCKS
    k_tile = block % K_TILE_BLOCKS
    full_generic = (
        ((n_tile + 1) * N_STEPS <= MATRIX_N)
        & ((k_tile + 1) * N_LANES <= MATRIX_K)
    )
    if full_generic:
        for step in tl.range(0, N_STEPS, 2, loop_unroll_factor=6):
            a_word2_prefetch = tl.load(
                a_encoded + tl.minimum(a_word + 2, FIXED_WORDS - 1) * n_streams
                + stream,
            ).to(tl.uint32).to(tl.uint64)
            b_word2_prefetch = tl.load(
                b_encoded + tl.minimum(b_word + 2, FIXED_WORDS - 1) * n_streams
                + stream,
            ).to(tl.uint32).to(tl.uint64)
            a0_is_fb = step >= a_start
            b0_is_fb = step >= b_start

            a0, a0_len = decode_symbol(
                a_window >> a_shift, a_decode_table, a_center_value,
                FIRST_MASK, RARE_LENGTH,
            )
            b0, b0_len = decode_symbol(
                b_window >> b_shift, b_decode_table, b_center_value,
                FIRST_MASK, RARE_LENGTH,
            )
            a0_fb = tl.load(
                a_fallback_buffer + a_real_base + a_offset + (step - a_start),
                mask=a0_is_fb, other=0,
            ).to(tl.int32)
            b0_fb = tl.load(
                b_fallback_buffer + b_real_base + b_offset + (step - b_start),
                mask=b0_is_fb, other=0,
            ).to(tl.int32)
            a0 = tl.where(a0_is_fb, a0_fb, a0)
            b0 = tl.where(b0_is_fb, b0_fb, b0)
            a0_len = tl.where(a0_is_fb, 0, a0_len)
            b0_len = tl.where(b0_is_fb, 0, b0_len)

            offset0 = block * BLOCK + step * N_LANES + lanes
            logical_n0 = n_tile * N_STEPS + step
            # Keep the logical column swizzle.  Do NOT simplify `logical_n & 255`
            # to `step` even when N_STEPS == 256: the row-dependent swizzle
            # matches the encode/decode layout and prevents correlations across
            # rows of the input, so future optimisations must preserve it.
            logical_k0 = k_tile * N_LANES + ((lanes + (logical_n0 & 255)) & 255)
            logical_offset0 = logical_n0 * MATRIX_K + logical_k0
            sm_a0 = tl.load(a_sign_mantissa + offset0, cache_modifier='.cg')
            sm_b0 = tl.load(b_sign_mantissa + offset0, cache_modifier='.cg')
            a_left0 = pack_bf16(a0, sm_a0).to(tl.int16).to(tl.bfloat16, bitcast=True)
            b_left0 = pack_bf16(b0, sm_b0).to(tl.int16).to(tl.bfloat16, bitcast=True)
            _store_result(
                tl.math.fma(a_left0, alpha_value, b_left0), output, auxiliary,
                offset0, logical_offset0, True, True, OUTPUT_POLICY,
            )

            a_shift1 = a_shift + a0_len
            b_shift1 = b_shift + b0_len
            a1_is_fb = step + 1 >= a_start
            b1_is_fb = step + 1 >= b_start

            a1, a1_len = decode_symbol(
                a_window >> a_shift1, a_decode_table, a_center_value,
                FIRST_MASK, RARE_LENGTH,
            )
            b1, b1_len = decode_symbol(
                b_window >> b_shift1, b_decode_table, b_center_value,
                FIRST_MASK, RARE_LENGTH,
            )
            a1_fb = tl.load(
                a_fallback_buffer + a_real_base + a_offset + (step + 1 - a_start),
                mask=a1_is_fb, other=0,
            ).to(tl.int32)
            b1_fb = tl.load(
                b_fallback_buffer + b_real_base + b_offset + (step + 1 - b_start),
                mask=b1_is_fb, other=0,
            ).to(tl.int32)
            a1 = tl.where(a1_is_fb, a1_fb, a1)
            b1 = tl.where(b1_is_fb, b1_fb, b1)
            a1_len = tl.where(a1_is_fb, 0, a1_len)
            b1_len = tl.where(b1_is_fb, 0, b1_len)

            offset1 = offset0 + N_LANES
            logical_n1 = logical_n0 + 1
            logical_k1 = k_tile * N_LANES + ((lanes + (logical_n1 & 255)) & 255)
            logical_offset1 = logical_n1 * MATRIX_K + logical_k1
            sm_a1 = tl.load(a_sign_mantissa + offset1, cache_modifier='.cg')
            sm_b1 = tl.load(b_sign_mantissa + offset1, cache_modifier='.cg')
            a_left1 = pack_bf16(a1, sm_a1).to(tl.int16).to(tl.bfloat16, bitcast=True)
            b_left1 = pack_bf16(b1, sm_b1).to(tl.int16).to(tl.bfloat16, bitcast=True)
            _store_result(
                tl.math.fma(a_left1, alpha_value, b_left1), output, auxiliary,
                offset1, logical_offset1, True, True, OUTPUT_POLICY,
            )

            a_next_shift = a_shift1 + a1_len
            a_crosses = a_next_shift >= 32
            a_next_window = (a_window >> 32) | (a_word2_prefetch << 32)
            a_window = tl.where(a_crosses, a_next_window, a_window)
            a_word += a_crosses
            a_shift = tl.where(a_crosses, a_next_shift - 32, a_next_shift)

            b_next_shift = b_shift1 + b1_len
            b_crosses = b_next_shift >= 32
            b_next_window = (b_window >> 32) | (b_word2_prefetch << 32)
            b_window = tl.where(b_crosses, b_next_window, b_window)
            b_word += b_crosses
            b_shift = tl.where(b_crosses, b_next_shift - 32, b_next_shift)

    else:
        for step in tl.range(0, N_STEPS):
            a_is_fb = step >= a_start
            b_is_fb = step >= b_start

            a_value, a_length = decode_symbol(
                a_window >> a_shift, a_decode_table, a_center_value,
                FIRST_MASK, RARE_LENGTH,
            )
            b_value, b_length = decode_symbol(
                b_window >> b_shift, b_decode_table, b_center_value,
                FIRST_MASK, RARE_LENGTH,
            )
            a_fb_value = tl.load(
                a_fallback_buffer + a_real_base + a_offset + (step - a_start),
                mask=a_is_fb, other=0,
            ).to(tl.int32)
            b_fb_value = tl.load(
                b_fallback_buffer + b_real_base + b_offset + (step - b_start),
                mask=b_is_fb, other=0,
            ).to(tl.int32)
            a_value = tl.where(a_is_fb, a_fb_value, a_value)
            b_value = tl.where(b_is_fb, b_fb_value, b_value)
            a_length = tl.where(a_is_fb, 0, a_length)
            b_length = tl.where(b_is_fb, 0, b_length)

            offset, logical_offset, storage_valid, logical_valid = _pointwise_location(
                block, step, lanes, n_elements, MATRIX_N, MATRIX_K,
                MATRIX_NUMEL, K_TILE_BLOCKS,
                BLOCK, N_LANES, N_STEPS,
            )
            valid = storage_valid
            logical_active = valid & logical_valid
            sm_a = tl.load(a_sign_mantissa + offset, mask=valid, other=0, cache_modifier='.cg')
            sm_b = tl.load(b_sign_mantissa + offset, mask=valid, other=0, cache_modifier='.cg')
            a_left = pack_bf16(a_value, sm_a).to(tl.int16).to(tl.bfloat16, bitcast=True)
            b_left = pack_bf16(b_value, sm_b).to(tl.int16).to(tl.bfloat16, bitcast=True)
            _store_result(
                tl.math.fma(a_left, alpha_value, b_left), output, auxiliary,
                offset, logical_offset, logical_active, valid, OUTPUT_POLICY,
            )

            a_shift1 = a_shift + a_length
            a_crosses = a_shift1 >= 32
            a_word2 = tl.load(
                a_encoded + tl.minimum(a_word + 2, FIXED_WORDS - 1) * n_streams
                + stream,
                mask=a_crosses, other=0,
            ).to(tl.uint32).to(tl.uint64)
            a_next_window = (a_window >> 32) | (a_word2 << 32)
            a_window = tl.where(a_crosses, a_next_window, a_window)
            a_word += a_crosses
            a_shift = tl.where(a_crosses, a_shift1 - 32, a_shift1)

            b_shift1 = b_shift + b_length
            b_crosses = b_shift1 >= 32
            b_word2 = tl.load(
                b_encoded + tl.minimum(b_word + 2, FIXED_WORDS - 1) * n_streams
                + stream,
                mask=b_crosses, other=0,
            ).to(tl.uint32).to(tl.uint64)
            b_next_window = (b_window >> 32) | (b_word2 << 32)
            b_window = tl.where(b_crosses, b_next_window, b_window)
            b_word += b_crosses
            b_shift = tl.where(b_crosses, b_shift1 - 32, b_shift1)



@triton.autotune(
    configs=DUAL_DECODE_AUTOTUNE_CONFIGS,
    key=["n_elements", "N_LANES", "N_STEPS", "FIXED_WORDS", "OUTPUT_POLICY"],
)
@triton.jit
def pointwise_scalar_mul_add_compressed_compressed_matrix_kernel(
    a_encoded, a_sign_mantissa, a_decode_table, a_center,
    a_bad_streams, a_bad_starts, a_fallback_offsets, a_metadata,
    a_descriptor, a_fallback_count,
    a_fallback_buffer, a_fallback_base,
    b_encoded, b_sign_mantissa, b_decode_table, b_center,
    b_bad_streams, b_bad_starts, b_fallback_offsets, b_metadata,
    b_descriptor, b_fallback_count,
    b_fallback_buffer, b_fallback_base,
    output, auxiliary, alpha,
    n_elements, n_streams,
    BUFFERED: tl.constexpr,
    OUTPUT_POLICY: tl.constexpr,
    MATRIX_N: tl.constexpr, MATRIX_K: tl.constexpr,
    MATRIX_NUMEL: tl.constexpr, K_TILE_BLOCKS: tl.constexpr,
    FIRST_MASK: tl.constexpr, RARE_LENGTH: tl.constexpr,
    BLOCK: tl.constexpr, N_LANES: tl.constexpr,
    N_STEPS: tl.constexpr, FIXED_WORDS: tl.constexpr,
):
    _scalar_mul_add_compressed_compressed_matrix_impl(
        a_encoded, a_sign_mantissa, a_decode_table, a_center,
        a_bad_streams, a_bad_starts, a_fallback_offsets, a_metadata,
        a_descriptor, a_fallback_count,
        a_fallback_buffer, a_fallback_base,
        b_encoded, b_sign_mantissa, b_decode_table, b_center,
        b_bad_streams, b_bad_starts, b_fallback_offsets, b_metadata,
        b_descriptor, b_fallback_count,
        b_fallback_buffer, b_fallback_base,
        output, auxiliary, alpha,
        n_elements, n_streams,
        BUFFERED, OUTPUT_POLICY,
        MATRIX_N, MATRIX_K, MATRIX_NUMEL, K_TILE_BLOCKS,
        FIRST_MASK, RARE_LENGTH, BLOCK, N_LANES, N_STEPS, FIXED_WORDS,
    )
