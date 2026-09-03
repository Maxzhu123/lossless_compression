"""Fused pointwise kernel for two compressed BF16 operands.

Computes ``output = alpha * a + b`` without materialising either operand as a
dense tensor.  This kernel currently handles the private-fallback, no-overflow
case.  Buffered fallback and overflowed streams are handled by the host-side
``compressed_scale_add`` fallback path.
"""

import triton
from triton import language as tl

from ..codec.autotune import DECODE_AUTOTUNE_CONFIGS, SCATTER_FALLBACK_AUTOTUNE_CONFIGS
from ..codec.primitives import decode_symbol, pack_bf16
from .pointwise import (
    _pointwise_location,
    _store_result,
)


@triton.jit
def _scalar_mul_add_compressed_compressed_impl(
    a_encoded, a_sign_mantissa,
    b_encoded, b_sign_mantissa,
    output, auxiliary, a_decode_table, b_decode_table,
    n_elements, n_streams, a_center, b_center, alpha,
    OUTPUT_POLICY: tl.constexpr,
    MATRIX_N: tl.constexpr, MATRIX_K: tl.constexpr,
    MATRIX_NUMEL: tl.constexpr,
    K_TILE_BLOCKS: tl.constexpr,
    FIRST_MASK: tl.constexpr, RARE_LENGTH: tl.constexpr,
    BLOCK: tl.constexpr, N_LANES: tl.constexpr,
    N_STEPS: tl.constexpr, FIXED_WORDS: tl.constexpr,
):
    """Apply ``output = alpha * a + b`` with both inputs compressed."""
    block = tl.program_id(0)
    lanes = tl.arange(0, N_LANES)
    lane_index = block * N_LANES + lanes

    a_word = tl.zeros((N_LANES,), tl.int32)
    a_shift = tl.zeros((N_LANES,), tl.int32)
    b_word = tl.zeros((N_LANES,), tl.int32)
    b_shift = tl.zeros((N_LANES,), tl.int32)

    a_word0 = tl.load(a_encoded + a_word * n_streams + lane_index)
    a_word1 = tl.load(a_encoded + (a_word + 1) * n_streams + lane_index)
    a_window = a_word0.to(tl.uint32).to(tl.uint64)
    a_window |= a_word1.to(tl.uint32).to(tl.uint64) << 32

    b_word0 = tl.load(b_encoded + b_word * n_streams + lane_index)
    b_word1 = tl.load(b_encoded + (b_word + 1) * n_streams + lane_index)
    b_window = b_word0.to(tl.uint32).to(tl.uint64)
    b_window |= b_word1.to(tl.uint32).to(tl.uint64) << 32

    a_center_value = tl.load(a_center).to(tl.int32)
    b_center_value = tl.load(b_center).to(tl.int32)
    alpha_value = tl.load(alpha).to(tl.float32)

    if K_TILE_BLOCKS == 1 and MATRIX_K == N_LANES and (block + 1) * BLOCK <= MATRIX_NUMEL:
        storage_offset = block * BLOCK + lanes
        true_mask = tl.full((N_LANES,), True, tl.int1)
        for step in tl.range(0, N_STEPS, 2, flatten=True, warp_specialize=True):
            logical_n0 = block * N_STEPS + step
            logical_n1 = logical_n0 + 1
            logical_k0 = (lanes + (logical_n0 & 255)) & 255
            logical_k1 = (lanes + (logical_n1 & 255)) & 255
            logical_offset0 = logical_n0 * MATRIX_K + logical_k0
            logical_offset1 = logical_n1 * MATRIX_K + logical_k1

            a0, a_len0 = decode_symbol(
                a_window >> a_shift, a_decode_table, a_center_value,
                FIRST_MASK, RARE_LENGTH,
            )
            b0, b_len0 = decode_symbol(
                b_window >> b_shift, b_decode_table, b_center_value,
                FIRST_MASK, RARE_LENGTH,
            )
            sm_a0 = tl.load(a_sign_mantissa + storage_offset, cache_modifier='.cg')
            sm_b0 = tl.load(b_sign_mantissa + storage_offset, cache_modifier='.cg')
            a_left0 = pack_bf16(a0, sm_a0).to(tl.int16).to(tl.bfloat16, bitcast=True)
            b_left0 = pack_bf16(b0, sm_b0).to(tl.int16).to(tl.bfloat16, bitcast=True)
            _store_result(
                tl.math.fma(a_left0, alpha_value, b_left0), output, auxiliary,
                storage_offset, logical_offset0, true_mask, true_mask,
                OUTPUT_POLICY,
            )

            a_shift1 = a_shift + a_len0
            b_shift1 = b_shift + b_len0
            a1, a_len1 = decode_symbol(
                a_window >> a_shift1, a_decode_table, a_center_value,
                FIRST_MASK, RARE_LENGTH,
            )
            b1, b_len1 = decode_symbol(
                b_window >> b_shift1, b_decode_table, b_center_value,
                FIRST_MASK, RARE_LENGTH,
            )
            sm_a1 = tl.load(a_sign_mantissa + storage_offset + N_LANES, cache_modifier='.cg')
            sm_b1 = tl.load(b_sign_mantissa + storage_offset + N_LANES, cache_modifier='.cg')
            a_left1 = pack_bf16(a1, sm_a1).to(tl.int16).to(tl.bfloat16, bitcast=True)
            b_left1 = pack_bf16(b1, sm_b1).to(tl.int16).to(tl.bfloat16, bitcast=True)
            _store_result(
                tl.math.fma(a_left1, alpha_value, b_left1), output, auxiliary,
                storage_offset + N_LANES, logical_offset1, true_mask, true_mask,
                OUTPUT_POLICY,
            )

            a_next_shift = a_shift1 + a_len1
            a_crosses = a_next_shift >= 32
            a_word2 = tl.load(
                a_encoded + tl.minimum(a_word + 2, FIXED_WORDS - 1) * n_streams
                + lane_index,
                mask=a_crosses, other=0,
            ).to(tl.uint32).to(tl.uint64)
            a_next_window = (a_window >> 32) | (a_word2 << 32)
            a_window = tl.where(a_crosses, a_next_window, a_window)
            a_word += a_crosses
            a_shift = tl.where(a_crosses, a_next_shift - 32, a_next_shift)

            b_next_shift = b_shift1 + b_len1
            b_crosses = b_next_shift >= 32
            b_word2 = tl.load(
                b_encoded + tl.minimum(b_word + 2, FIXED_WORDS - 1) * n_streams
                + lane_index,
                mask=b_crosses, other=0,
            ).to(tl.uint32).to(tl.uint64)
            b_next_window = (b_window >> 32) | (b_word2 << 32)
            b_window = tl.where(b_crosses, b_next_window, b_window)
            b_word += b_crosses
            b_shift = tl.where(b_crosses, b_next_shift - 32, b_next_shift)

            storage_offset += 2 * N_LANES
    else:
        storage_offset = block * BLOCK + lanes
        logical_n_base = 0
        logical_k = lanes
        k_tile = 0
        if K_TILE_BLOCKS > 1:
            n_tile = block // K_TILE_BLOCKS
            k_tile = block % K_TILE_BLOCKS
            logical_k = k_tile * N_LANES + lanes
            logical_n_base = n_tile * N_STEPS
        if K_TILE_BLOCKS == 1 and MATRIX_K == N_LANES:
            full_generic = False
        else:
            full_generic = (
                ((n_tile + 1) * N_STEPS <= MATRIX_N)
                & ((k_tile + 1) * N_LANES <= MATRIX_K)
            )
        if full_generic:
            for step in tl.range(0, N_STEPS, 2, loop_unroll_factor=4):
                a_word2_prefetch = tl.load(
                    a_encoded + tl.minimum(a_word + 2, FIXED_WORDS - 1) * n_streams
                    + lane_index,
                ).to(tl.uint32).to(tl.uint64)
                b_word2_prefetch = tl.load(
                    b_encoded + tl.minimum(b_word + 2, FIXED_WORDS - 1) * n_streams
                    + lane_index,
                ).to(tl.uint32).to(tl.uint64)
                logical_n = logical_n_base + step
                logical_k = k_tile * N_LANES + ((lanes + (logical_n & 255)) & 255)
                logical_offset = logical_n * MATRIX_K + logical_k
                a0, a_len0 = decode_symbol(
                    a_window >> a_shift, a_decode_table, a_center_value,
                    FIRST_MASK, RARE_LENGTH,
                )
                b0, b_len0 = decode_symbol(
                    b_window >> b_shift, b_decode_table, b_center_value,
                    FIRST_MASK, RARE_LENGTH,
                )
                sm_a = tl.load(a_sign_mantissa + storage_offset, cache_modifier='.cg')
                sm_b = tl.load(b_sign_mantissa + storage_offset, cache_modifier='.cg')
                a_left = pack_bf16(a0, sm_a).to(tl.int16).to(tl.bfloat16, bitcast=True)
                b_left = pack_bf16(b0, sm_b).to(tl.int16).to(tl.bfloat16, bitcast=True)
                _store_result(
                    tl.math.fma(a_left, alpha_value, b_left), output, auxiliary,
                    storage_offset, logical_offset, True, True, OUTPUT_POLICY,
                )

                a_shift1 = a_shift + a_len0
                b_shift1 = b_shift + b_len0
                a1, a_len1 = decode_symbol(
                    a_window >> a_shift1, a_decode_table, a_center_value,
                    FIRST_MASK, RARE_LENGTH,
                )
                b1, b_len1 = decode_symbol(
                    b_window >> b_shift1, b_decode_table, b_center_value,
                    FIRST_MASK, RARE_LENGTH,
                )
                logical_n1 = logical_n + 1
                logical_k1 = k_tile * N_LANES + ((lanes + (logical_n1 & 255)) & 255)
                logical_offset1 = logical_n1 * MATRIX_K + logical_k1
                sm_a1 = tl.load(a_sign_mantissa + storage_offset + N_LANES, cache_modifier='.cg')
                sm_b1 = tl.load(b_sign_mantissa + storage_offset + N_LANES, cache_modifier='.cg')
                a_left1 = pack_bf16(a1, sm_a1).to(tl.int16).to(tl.bfloat16, bitcast=True)
                b_left1 = pack_bf16(b1, sm_b1).to(tl.int16).to(tl.bfloat16, bitcast=True)
                _store_result(
                    tl.math.fma(a_left1, alpha_value, b_left1), output, auxiliary,
                    storage_offset + N_LANES, logical_offset1, True, True,
                    OUTPUT_POLICY,
                )

                a_next_shift = a_shift1 + a_len1
                a_crosses = a_next_shift >= 32
                a_next_window = (a_window >> 32) | (a_word2_prefetch << 32)
                a_window = tl.where(a_crosses, a_next_window, a_window)
                a_word += a_crosses
                a_shift = tl.where(a_crosses, a_next_shift - 32, a_next_shift)

                b_next_shift = b_shift1 + b_len1
                b_crosses = b_next_shift >= 32
                b_next_window = (b_window >> 32) | (b_word2_prefetch << 32)
                b_window = tl.where(b_crosses, b_next_window, b_window)
                b_word += b_crosses
                b_shift = tl.where(b_crosses, b_next_shift - 32, b_next_shift)

                storage_offset += 2 * N_LANES
        else:
            for step in tl.range(0, N_STEPS, 2):
                offset, logical_offset, storage_valid, valid = _pointwise_location(
                    block, step, lanes, n_elements, MATRIX_N, MATRIX_K,
                    MATRIX_NUMEL, K_TILE_BLOCKS,
                    BLOCK, N_LANES, N_STEPS,
                )
                a0, a_len0 = decode_symbol(
                    a_window >> a_shift, a_decode_table, a_center_value,
                    FIRST_MASK, RARE_LENGTH,
                )
                b0, b_len0 = decode_symbol(
                    b_window >> b_shift, b_decode_table, b_center_value,
                    FIRST_MASK, RARE_LENGTH,
                )
                sm_a = tl.load(a_sign_mantissa + offset, mask=storage_valid, other=0, cache_modifier='.cg')
                sm_b = tl.load(b_sign_mantissa + offset, mask=storage_valid, other=0, cache_modifier='.cg')
                a_left = pack_bf16(a0, sm_a).to(tl.int16).to(tl.bfloat16, bitcast=True)
                b_left = pack_bf16(b0, sm_b).to(tl.int16).to(tl.bfloat16, bitcast=True)
                _store_result(
                    tl.math.fma(a_left, alpha_value, b_left), output, auxiliary,
                    offset, logical_offset, valid, storage_valid, OUTPUT_POLICY,
                )

                a_shift1 = a_shift + tl.where(storage_valid, a_len0, 0)
                b_shift1 = b_shift + tl.where(storage_valid, b_len0, 0)
                offset1, logical_offset1, storage_valid1, valid1 = _pointwise_location(
                    block, step + 1, lanes, n_elements, MATRIX_N, MATRIX_K,
                    MATRIX_NUMEL, K_TILE_BLOCKS,
                    BLOCK, N_LANES, N_STEPS,
                )
                a1, a_len1 = decode_symbol(
                    a_window >> a_shift1, a_decode_table, a_center_value,
                    FIRST_MASK, RARE_LENGTH,
                )
                b1, b_len1 = decode_symbol(
                    b_window >> b_shift1, b_decode_table, b_center_value,
                    FIRST_MASK, RARE_LENGTH,
                )
                sm_a1 = tl.load(a_sign_mantissa + offset1, mask=storage_valid1, other=0, cache_modifier='.cg')
                sm_b1 = tl.load(b_sign_mantissa + offset1, mask=storage_valid1, other=0, cache_modifier='.cg')
                a_left1 = pack_bf16(a1, sm_a1).to(tl.int16).to(tl.bfloat16, bitcast=True)
                b_left1 = pack_bf16(b1, sm_b1).to(tl.int16).to(tl.bfloat16, bitcast=True)
                _store_result(
                    tl.math.fma(a_left1, alpha_value, b_left1), output, auxiliary,
                    offset1, logical_offset1, valid1, storage_valid1, OUTPUT_POLICY,
                )

                a_next_shift = a_shift1 + tl.where(storage_valid1, a_len1, 0)
                a_crosses = a_next_shift >= 32
                a_word2 = tl.load(
                    a_encoded + tl.minimum(a_word + 2, FIXED_WORDS - 1) * n_streams
                    + lane_index,
                    mask=a_crosses, other=0,
                ).to(tl.uint32).to(tl.uint64)
                a_next_window = (a_window >> 32) | (a_word2 << 32)
                a_window = tl.where(a_crosses, a_next_window, a_window)
                a_word += a_crosses
                a_shift = tl.where(a_crosses, a_next_shift - 32, a_next_shift)

                b_next_shift = b_shift1 + tl.where(storage_valid1, b_len1, 0)
                b_crosses = b_next_shift >= 32
                b_word2 = tl.load(
                    b_encoded + tl.minimum(b_word + 2, FIXED_WORDS - 1) * n_streams
                    + lane_index,
                    mask=b_crosses, other=0,
                ).to(tl.uint32).to(tl.uint64)
                b_next_window = (b_window >> 32) | (b_word2 << 32)
                b_window = tl.where(b_crosses, b_next_window, b_window)
                b_word += b_crosses
                b_shift = tl.where(b_crosses, b_next_shift - 32, b_next_shift)


@triton.autotune(
    configs=DECODE_AUTOTUNE_CONFIGS,
    key=["n_elements", "N_LANES", "N_STEPS", "FIXED_WORDS", "OUTPUT_POLICY"],
)
@triton.jit
def pointwise_scalar_mul_add_compressed_compressed_kernel(
    a_encoded, a_sign_mantissa, b_encoded, b_sign_mantissa,
    output, auxiliary, a_decode_table, b_decode_table,
    n_elements, n_streams, a_center, b_center, alpha,
    OUTPUT_POLICY: tl.constexpr,
    MATRIX_N: tl.constexpr, MATRIX_K: tl.constexpr,
    MATRIX_NUMEL: tl.constexpr, K_TILE_BLOCKS: tl.constexpr,
    FIRST_MASK: tl.constexpr, RARE_LENGTH: tl.constexpr,
    BLOCK: tl.constexpr, N_LANES: tl.constexpr,
    N_STEPS: tl.constexpr, FIXED_WORDS: tl.constexpr,
):
    _scalar_mul_add_compressed_compressed_impl(
        a_encoded, a_sign_mantissa, b_encoded, b_sign_mantissa,
        output, auxiliary, a_decode_table, b_decode_table,
        n_elements, n_streams, a_center, b_center, alpha, OUTPUT_POLICY,
        MATRIX_N, MATRIX_K, MATRIX_NUMEL, K_TILE_BLOCKS,
        FIRST_MASK, RARE_LENGTH, BLOCK, N_LANES, N_STEPS, FIXED_WORDS,
    )




@triton.jit
def _scalar_mul_add_compressed_compressed_fallback_impl(
    jobs, job_sides,
    a_encoded, a_sign_mantissa, a_decode_table, a_center,
    b_encoded, b_sign_mantissa, b_decode_table, b_center,
    a_fallback_buffer, a_fallback_base,
    a_fallback_starts, a_fallback_offsets,
    b_fallback_buffer, b_fallback_base,
    b_fallback_starts, b_fallback_offsets,
    output, auxiliary, alpha,
    n_elements, n_streams,
    job_count,
    OUTPUT_POLICY: tl.constexpr,
    MATRIX_N: tl.constexpr, MATRIX_K: tl.constexpr,
    MATRIX_NUMEL: tl.constexpr, K_TILE_BLOCKS: tl.constexpr,
    FIRST_MASK: tl.constexpr, RARE_LENGTH: tl.constexpr,
    BLOCK: tl.constexpr, N_LANES: tl.constexpr,
    N_STEPS: tl.constexpr, FIXED_WORDS: tl.constexpr,
    TILE: tl.constexpr,
):
    """Correct output tails for streams stored in either fallback buffer."""
    pid = tl.program_id(0)
    tile = pid * TILE + tl.arange(0, TILE)
    valid = tile < job_count
    if pid * TILE >= job_count:
        return

    stream = tl.load(jobs + tile, mask=valid, other=0).to(tl.int32)
    side = tl.load(job_sides + tile, mask=valid, other=0).to(tl.int32)
    block = stream // N_LANES
    lane = stream - block * N_LANES

    a_start = tl.load(a_fallback_starts + stream, mask=valid, other=N_STEPS).to(tl.int32)
    a_offset = tl.load(a_fallback_offsets + stream, mask=valid, other=0).to(tl.int32)
    b_start = tl.load(b_fallback_starts + stream, mask=valid, other=N_STEPS).to(tl.int32)
    b_offset = tl.load(b_fallback_offsets + stream, mask=valid, other=0).to(tl.int32)
    correction_start = tl.where(side == 0, a_start, b_start)

    a_word = tl.zeros((TILE,), tl.int32)
    a_shift = tl.zeros((TILE,), tl.int32)
    b_word = tl.zeros((TILE,), tl.int32)
    b_shift = tl.zeros((TILE,), tl.int32)

    a_word0 = tl.load(a_encoded + a_word * n_streams + stream, mask=valid, other=0)
    a_word1 = tl.load(a_encoded + (a_word + 1) * n_streams + stream, mask=valid, other=0)
    a_window = a_word0.to(tl.uint32).to(tl.uint64)
    a_window |= a_word1.to(tl.uint32).to(tl.uint64) << 32
    b_word0 = tl.load(b_encoded + b_word * n_streams + stream, mask=valid, other=0)
    b_word1 = tl.load(b_encoded + (b_word + 1) * n_streams + stream, mask=valid, other=0)
    b_window = b_word0.to(tl.uint32).to(tl.uint64)
    b_window |= b_word1.to(tl.uint32).to(tl.uint64) << 32

    a_center_value = tl.load(a_center).to(tl.int32)
    b_center_value = tl.load(b_center).to(tl.int32)
    alpha_value = tl.load(alpha).to(tl.float32)

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
            a_fallback_buffer + a_fallback_base + a_offset + (step - a_start),
            mask=valid & a_is_fb, other=0,
        ).to(tl.int32)
        b_fb_value = tl.load(
            b_fallback_buffer + b_fallback_base + b_offset + (step - b_start),
            mask=valid & b_is_fb, other=0,
        ).to(tl.int32)

        a_value = tl.where(a_is_fb, a_fb_value, a_value)
        b_value = tl.where(b_is_fb, b_fb_value, b_value)
        a_length = tl.where(a_is_fb, 0, a_length)
        b_length = tl.where(b_is_fb, 0, b_length)

        active = valid & (step >= correction_start)
        offset, logical_offset, storage_valid, logical_valid = _pointwise_location(
            block, step, lane, n_elements, MATRIX_N, MATRIX_K,
            MATRIX_NUMEL, K_TILE_BLOCKS,
            BLOCK, N_LANES, N_STEPS,
        )
        active = active & storage_valid
        logical_active = active & logical_valid
        sm_a = tl.load(a_sign_mantissa + offset, mask=active, other=0, cache_modifier='.cg')
        sm_b = tl.load(b_sign_mantissa + offset, mask=active, other=0, cache_modifier='.cg')
        a_left = pack_bf16(a_value, sm_a).to(tl.int16).to(tl.bfloat16, bitcast=True)
        b_left = pack_bf16(b_value, sm_b).to(tl.int16).to(tl.bfloat16, bitcast=True)
        _store_result(
            tl.math.fma(a_left, alpha_value, b_left), output, auxiliary,
            offset, logical_offset, logical_active, active, OUTPUT_POLICY,
        )

        a_next_shift = a_shift + a_length
        a_crosses = a_next_shift >= 32
        a_word2 = tl.load(
            a_encoded + tl.minimum(a_word + 2, FIXED_WORDS - 1) * n_streams
            + stream,
            mask=a_crosses & valid, other=0,
        ).to(tl.uint32).to(tl.uint64)
        a_next_window = (a_window >> 32) | (a_word2 << 32)
        a_window = tl.where(a_crosses, a_next_window, a_window)
        a_word += a_crosses
        a_shift = tl.where(a_crosses, a_next_shift - 32, a_next_shift)

        b_next_shift = b_shift + b_length
        b_crosses = b_next_shift >= 32
        b_word2 = tl.load(
            b_encoded + tl.minimum(b_word + 2, FIXED_WORDS - 1) * n_streams
            + stream,
            mask=b_crosses & valid, other=0,
        ).to(tl.uint32).to(tl.uint64)
        b_next_window = (b_window >> 32) | (b_word2 << 32)
        b_window = tl.where(b_crosses, b_next_window, b_window)
        b_word += b_crosses
        b_shift = tl.where(b_crosses, b_next_shift - 32, b_next_shift)


@triton.autotune(
    configs=SCATTER_FALLBACK_AUTOTUNE_CONFIGS,
    key=["n_elements", "N_LANES", "N_STEPS", "BLOCK"],
)
@triton.jit
def pointwise_scalar_mul_add_compressed_compressed_fallback_kernel(
    jobs, job_sides,
    a_encoded, a_sign_mantissa, a_decode_table, a_center,
    b_encoded, b_sign_mantissa, b_decode_table, b_center,
    a_fallback_buffer, a_fallback_base,
    a_fallback_starts, a_fallback_offsets,
    b_fallback_buffer, b_fallback_base,
    b_fallback_starts, b_fallback_offsets,
    output, auxiliary, alpha,
    n_elements, n_streams,
    job_count,
    OUTPUT_POLICY: tl.constexpr,
    MATRIX_N: tl.constexpr, MATRIX_K: tl.constexpr,
    MATRIX_NUMEL: tl.constexpr, K_TILE_BLOCKS: tl.constexpr,
    FIRST_MASK: tl.constexpr, RARE_LENGTH: tl.constexpr,
    BLOCK: tl.constexpr, N_LANES: tl.constexpr,
    N_STEPS: tl.constexpr, FIXED_WORDS: tl.constexpr,
    TILE: tl.constexpr,
):
    _scalar_mul_add_compressed_compressed_fallback_impl(
        jobs, job_sides,
        a_encoded, a_sign_mantissa, a_decode_table, a_center,
        b_encoded, b_sign_mantissa, b_decode_table, b_center,
        a_fallback_buffer, a_fallback_base,
        a_fallback_starts, a_fallback_offsets,
        b_fallback_buffer, b_fallback_base,
        b_fallback_starts, b_fallback_offsets,
        output, auxiliary, alpha,
        n_elements, n_streams,
        job_count, OUTPUT_POLICY,
        MATRIX_N, MATRIX_K, MATRIX_NUMEL, K_TILE_BLOCKS,
        FIRST_MASK, RARE_LENGTH, BLOCK, N_LANES, N_STEPS, FIXED_WORDS,
        TILE,
    )
