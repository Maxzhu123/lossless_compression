"""Dedicated fused scalar multiply-add pointwise kernels.

Kept separate from the binary pointwise kernels so the existing add/multiply
hot path does not pay any scalar/alpha overhead.
"""

import triton
from triton import language as tl

from ..codec.autotune import (
    DECODE_AUTOTUNE_CONFIGS,
    SCATTER_FALLBACK_AUTOTUNE_CONFIGS,
)
from ..codec.primitives import decode_symbol, pack_bf16
from .pointwise import (
    _pointwise_location,
    _store_result,
)


@triton.jit
def _scalar_mul_add_compressed_dense_impl(
    encoded, sign_mantissa, other, output, auxiliary, decode_table,
    n_elements, n_streams, center, alpha,
    OUTPUT_POLICY: tl.constexpr,
    MATRIX_N: tl.constexpr, MATRIX_K: tl.constexpr,
    MATRIX_NUMEL: tl.constexpr,
    K_TILE_BLOCKS: tl.constexpr,
    FIRST_MASK: tl.constexpr, RARE_LENGTH: tl.constexpr,
    BLOCK: tl.constexpr, N_LANES: tl.constexpr,
    N_STEPS: tl.constexpr, FIXED_WORDS: tl.constexpr,
):
    """Apply ``output = alpha * decoded + other``."""
    # One program handles one codec block; each lane decodes one fixed stream.
    block = tl.program_id(0)
    lanes = tl.arange(0, N_LANES)
    lane_index = block * N_LANES + lanes
    word = tl.zeros((N_LANES,), tl.int32)
    shift = tl.zeros((N_LANES,), tl.int32)
    word0 = tl.load(encoded + word * n_streams + lane_index)
    word1 = tl.load(encoded + (word + 1) * n_streams + lane_index)
    window = word0.to(tl.uint32).to(tl.uint64)
    window |= word1.to(tl.uint32).to(tl.uint64) << 32
    center_value = tl.load(center).to(tl.int32)
    alpha_value = tl.load(alpha).to(tl.float32)

    if K_TILE_BLOCKS == 1 and MATRIX_K == N_LANES and (block + 1) * BLOCK <= MATRIX_NUMEL:
        storage_offset = block * BLOCK + lanes
        true_mask = tl.full((N_LANES,), True, tl.int1)
        k_tile = 0
        for step in tl.range(0, N_STEPS, 2, flatten=True, warp_specialize=True):
            logical_n0 = block * N_STEPS + step
            logical_n1 = logical_n0 + 1
            logical_k0 = (lanes + (logical_n0 & 255)) & 255
            logical_k1 = (lanes + (logical_n1 & 255)) & 255
            logical_offset0 = logical_n0 * MATRIX_K + logical_k0
            logical_offset1 = logical_n1 * MATRIX_K + logical_k1
            current = window >> shift
            value, length = decode_symbol(
                current, decode_table, center_value,
                FIRST_MASK, RARE_LENGTH,
            )
            sm = tl.load(sign_mantissa + storage_offset, cache_modifier='.cg')
            left = pack_bf16(value, sm).to(tl.int16).to(
                tl.bfloat16, bitcast=True
            )
            right = tl.load(other + logical_offset0, cache_modifier='.cg')
            _store_result(
                tl.math.fma(left, alpha_value, right), output, auxiliary, storage_offset,
                logical_offset0, true_mask, true_mask, OUTPUT_POLICY,
            )
            shift1 = shift + length
            current1 = window >> shift1
            value1, length1 = decode_symbol(
                current1, decode_table, center_value,
                FIRST_MASK, RARE_LENGTH,
            )
            sm1 = tl.load(sign_mantissa + storage_offset + N_LANES, cache_modifier='.cg')
            left1 = pack_bf16(value1, sm1).to(tl.int16).to(
                tl.bfloat16, bitcast=True
            )
            right1 = tl.load(other + logical_offset1, cache_modifier='.cg')
            _store_result(
                tl.math.fma(left1, alpha_value, right1), output, auxiliary,
                storage_offset + N_LANES, logical_offset1,
                true_mask, true_mask, OUTPUT_POLICY,
            )
            next_shift = shift1 + length1
            crosses_word = next_shift >= 32
            word2 = tl.load(
                encoded + tl.minimum(word + 2, FIXED_WORDS - 1) * n_streams
                + lane_index,
                mask=crosses_word, other=0,
            ).to(tl.uint32).to(tl.uint64)
            next_window = (window >> 32) | (word2 << 32)
            window = tl.where(crosses_word, next_window, window)
            word += crosses_word
            shift = tl.where(crosses_word, next_shift - 32, next_shift)
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
                word2_prefetch = tl.load(
                    encoded + tl.minimum(word + 2, FIXED_WORDS - 1) * n_streams
                    + lane_index,
                ).to(tl.uint32).to(tl.uint64)
                logical_n = logical_n_base + step
                logical_k = k_tile * N_LANES + ((lanes + (logical_n & 255)) & 255)
                logical_offset = logical_n * MATRIX_K + logical_k
                value, length = decode_symbol(
                    window >> shift, decode_table, center_value,
                    FIRST_MASK, RARE_LENGTH,
                )
                sm = tl.load(sign_mantissa + storage_offset, cache_modifier='.cg')
                left = pack_bf16(value, sm).to(tl.int16).to(
                    tl.bfloat16, bitcast=True
                )
                right = tl.load(other + logical_offset, cache_modifier='.cg')
                _store_result(
                    tl.math.fma(left, alpha_value, right), output, auxiliary, storage_offset,
                    logical_offset, True, True, OUTPUT_POLICY,
                )
                shift1 = shift + length
                current1 = window >> shift1
                value1, length1 = decode_symbol(
                    current1, decode_table, center_value,
                    FIRST_MASK, RARE_LENGTH,
                )
                logical_n1 = logical_n + 1
                logical_k1 = k_tile * N_LANES + ((lanes + (logical_n1 & 255)) & 255)
                logical_offset1 = logical_n1 * MATRIX_K + logical_k1
                sm1 = tl.load(sign_mantissa + storage_offset + N_LANES, cache_modifier='.cg')
                left1 = pack_bf16(value1, sm1).to(tl.int16).to(
                    tl.bfloat16, bitcast=True
                )
                right1 = tl.load(other + logical_offset1, cache_modifier='.cg')
                _store_result(
                    tl.math.fma(left1, alpha_value, right1), output, auxiliary,
                    storage_offset + N_LANES, logical_offset1,
                    True, True, OUTPUT_POLICY,
                )
                next_shift = shift1 + length1
                crosses_word = next_shift >= 32
                next_window = (window >> 32) | (word2_prefetch << 32)
                window = tl.where(crosses_word, next_window, window)
                word += crosses_word
                shift = tl.where(crosses_word, next_shift - 32, next_shift)
                storage_offset += 2 * N_LANES
        else:
            for step in tl.range(0, N_STEPS, 2):
                offset, logical_offset, storage_valid, valid = _pointwise_location(
                    block, step, lanes, n_elements, MATRIX_N, MATRIX_K,
                    MATRIX_NUMEL, K_TILE_BLOCKS,
                    BLOCK, N_LANES, N_STEPS,
                )
                value, length = decode_symbol(
                    window >> shift, decode_table, center_value,
                    FIRST_MASK, RARE_LENGTH,
                )
                sm = tl.load(sign_mantissa + offset, mask=storage_valid, other=0, cache_modifier='.cg')
                left = pack_bf16(value, sm).to(tl.int16).to(
                    tl.bfloat16, bitcast=True
                )
                right = tl.load(other + logical_offset, mask=valid, other=0.0, cache_modifier='.cg')
                _store_result(
                    tl.math.fma(left, alpha_value, right), output, auxiliary, offset, logical_offset,
                    valid, storage_valid, OUTPUT_POLICY,
                )
                shift1 = shift + tl.where(storage_valid, length, 0)
                offset1, logical_offset1, storage_valid1, valid1 = _pointwise_location(
                    block, step + 1, lanes, n_elements, MATRIX_N, MATRIX_K,
                    MATRIX_NUMEL, K_TILE_BLOCKS,
                    BLOCK, N_LANES, N_STEPS,
                )
                value1, length1 = decode_symbol(
                    window >> shift1, decode_table, center_value,
                    FIRST_MASK, RARE_LENGTH,
                )
                sm1 = tl.load(sign_mantissa + offset1, mask=storage_valid1, other=0, cache_modifier='.cg')
                left1 = pack_bf16(value1, sm1).to(tl.int16).to(
                    tl.bfloat16, bitcast=True
                )
                right1 = tl.load(other + logical_offset1, mask=valid1, other=0.0, cache_modifier='.cg')
                _store_result(
                    tl.math.fma(left1, alpha_value, right1), output, auxiliary, offset1, logical_offset1,
                    valid1, storage_valid1, OUTPUT_POLICY,
                )
                next_shift = shift1 + tl.where(storage_valid1, length1, 0)
                crosses_word = next_shift >= 32
                word2 = tl.load(
                    encoded + tl.minimum(word + 2, FIXED_WORDS - 1) * n_streams
                    + lane_index,
                    mask=crosses_word, other=0,
                ).to(tl.uint32).to(tl.uint64)
                next_window = (window >> 32) | (word2 << 32)
                window = tl.where(crosses_word, next_window, window)
                word += crosses_word
                shift = tl.where(crosses_word, next_shift - 32, next_shift)


@triton.autotune(
    configs=DECODE_AUTOTUNE_CONFIGS,
    key=["n_elements", "N_LANES", "N_STEPS", "FIXED_WORDS", "OUTPUT_POLICY"],
)
@triton.jit
def pointwise_scalar_mul_add_dense_matrix_kernel(
    encoded, sign_mantissa, other, output, auxiliary, decode_table,
    n_elements, n_streams, center, alpha,
    OUTPUT_POLICY: tl.constexpr,
    MATRIX_N: tl.constexpr, MATRIX_K: tl.constexpr,
    MATRIX_NUMEL: tl.constexpr, K_TILE_BLOCKS: tl.constexpr,
    FIRST_MASK: tl.constexpr, RARE_LENGTH: tl.constexpr,
    BLOCK: tl.constexpr, N_LANES: tl.constexpr,
    N_STEPS: tl.constexpr, FIXED_WORDS: tl.constexpr,
):
    _scalar_mul_add_compressed_dense_impl(
        encoded, sign_mantissa, other, output, auxiliary, decode_table,
        n_elements, n_streams, center, alpha, OUTPUT_POLICY,
        MATRIX_N, MATRIX_K, MATRIX_NUMEL, K_TILE_BLOCKS,
        FIRST_MASK, RARE_LENGTH, BLOCK, N_LANES, N_STEPS, FIXED_WORDS,
    )


@triton.jit
def _scalar_mul_add_compressed_dense_fallback_impl(
    bad_streams, bad_starts, fallback_offsets,
    fallback_buffer, fallback_base, metadata, descriptor, fallback_count,
    sign_mantissa, other, output, auxiliary, n_elements, alpha,
    OUTPUT_POLICY: tl.constexpr, BUFFERED: tl.constexpr,
    MATRIX_N: tl.constexpr, MATRIX_K: tl.constexpr,
    MATRIX_NUMEL: tl.constexpr,
    K_TILE_BLOCKS: tl.constexpr,
    TILE: tl.constexpr, BLOCK: tl.constexpr,
    N_LANES: tl.constexpr, N_STEPS: tl.constexpr,
):
    pid = tl.program_id(0)
    tile = pid * TILE + tl.arange(0, TILE)
    alpha_value = tl.load(alpha).to(tl.float32)
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
    lane = stream - block * N_LANES
    tail_steps = N_STEPS - start
    max_tail = tl.max(tl.where(valid, tail_steps, 0), axis=0)
    for tail_step in tl.range(0, max_tail):
        step = start + tail_step
        offset, logical_offset, storage_valid, logical_valid = _pointwise_location(
            block, step, lane, n_elements, MATRIX_N, MATRIX_K,
            MATRIX_NUMEL, K_TILE_BLOCKS,
            BLOCK, N_LANES, N_STEPS,
        )
        active = valid & (tail_step < tail_steps) & storage_valid
        logical_active = active & logical_valid
        exponent = tl.load(
            fallback_buffer + fallback_base + fallback_offset + tail_step,
            mask=active, other=0,
        ).to(tl.int32)
        sm = tl.load(sign_mantissa + offset, mask=active, other=0, cache_modifier='.cg')
        left = pack_bf16(exponent, sm).to(tl.int16).to(
            tl.bfloat16, bitcast=True
        )
        right = tl.load(other + logical_offset, mask=logical_active, other=0.0, cache_modifier='.cg')
        _store_result(
            tl.math.fma(left, alpha_value, right), output, auxiliary, offset, logical_offset,
            logical_active, active, OUTPUT_POLICY,
        )


@triton.autotune(
    configs=SCATTER_FALLBACK_AUTOTUNE_CONFIGS,
    key=["n_elements", "N_LANES", "N_STEPS", "BLOCK"],
)
@triton.jit
def pointwise_scalar_mul_add_dense_matrix_fallback_kernel(
    bad_streams, bad_starts, fallback_offsets,
    fallback_buffer, fallback_base, metadata, descriptor, fallback_count,
    sign_mantissa, other, output, auxiliary, n_elements, alpha,
    OUTPUT_POLICY: tl.constexpr, BUFFERED: tl.constexpr,
    MATRIX_N: tl.constexpr, MATRIX_K: tl.constexpr,
    MATRIX_NUMEL: tl.constexpr, K_TILE_BLOCKS: tl.constexpr,
    TILE: tl.constexpr, BLOCK: tl.constexpr,
    N_LANES: tl.constexpr, N_STEPS: tl.constexpr,
):
    _scalar_mul_add_compressed_dense_fallback_impl(
        bad_streams, bad_starts, fallback_offsets, fallback_buffer,
        fallback_base, metadata, descriptor, fallback_count,
        sign_mantissa, other, output, auxiliary, n_elements, alpha,
        OUTPUT_POLICY, BUFFERED,
        MATRIX_N, MATRIX_K, MATRIX_NUMEL, K_TILE_BLOCKS,
        TILE, BLOCK, N_LANES, N_STEPS,
    )
