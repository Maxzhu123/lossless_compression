"""Fused pointwise kernels for two compressed BF16 operands.

Computes ``output = alpha * a + b`` without materialising either operand as a
dense tensor.  The implementation uses a fixed-stream main kernel followed by
parallel fallback-tile kernels, handling private and buffered fallback storage.
"""

import triton
from triton import language as tl

from ..codec.autotune import DUAL_DECODE_AUTOTUNE_CONFIGS
from ..codec.primitives import decode_symbol, pack_bf16
from .pointwise import _pointwise_location, _store_result



@triton.jit
def _setup_fallback_map_kernel(
    bad_streams, bad_starts, fallback_offsets, metadata,
    descriptor, fallback_count,
    stream_starts, stream_offsets,
    BUFFERED: tl.constexpr,
):
    """Build per-stream fallback maps without host-side count synchronisation."""
    pid = tl.program_id(0)
    offs = pid * 1024 + tl.arange(0, 1024)
    count = tl.load(fallback_count).to(tl.int32)
    valid = offs < count
    if pid * 1024 >= count:
        return
    if BUFFERED:
        base = tl.load(descriptor).to(tl.int32)
        base_words = base // 4
        stream = tl.load(metadata + base_words + offs, mask=valid, other=0).to(tl.int32)
        start = tl.load(bad_starts + base + 8 * count + offs, mask=valid, other=0).to(tl.int32)
        offset = tl.load(metadata + base_words + count + offs, mask=valid, other=0).to(tl.int32)
    else:
        stream = tl.load(bad_streams + offs, mask=valid, other=0).to(tl.int32)
        start = tl.load(bad_starts + offs, mask=valid, other=0).to(tl.int32)
        offset = tl.load(fallback_offsets + offs, mask=valid, other=0).to(tl.int32)
    tl.store(stream_starts + stream, start, mask=valid)
    tl.store(stream_offsets + stream, offset, mask=valid)


@triton.jit
def _advance_decode(word, shift, window, bit_length, encoded, n_streams, stream,
                    FIXED_WORDS: tl.constexpr):
    """Advance one Huffman stream after decoding ``bit_length`` bits."""
    shift1 = shift + bit_length
    crosses = shift1 >= 32
    word2 = tl.load(
        encoded + tl.minimum(word + 2, FIXED_WORDS - 1) * n_streams + stream,
        mask=crosses, other=0,
    ).to(tl.uint32).to(tl.uint64)
    next_window = (window >> 32) | (word2 << 32)
    window = tl.where(crosses, next_window, window)
    word += crosses
    shift = tl.where(crosses, shift1 - 32, shift1)
    return word, shift, window


@triton.autotune(
    configs=DUAL_DECODE_AUTOTUNE_CONFIGS,
    key=["n_elements", "N_LANES", "N_STEPS", "FIXED_WORDS", "OUTPUT_POLICY"],
)
@triton.jit
def pointwise_scalar_mul_add_compressed_compressed_fixed_matrix_kernel(
    a_encoded, a_sign_mantissa, a_decode_table, a_center,
    b_encoded, b_sign_mantissa, b_decode_table, b_center,
    output, auxiliary, alpha,
    n_elements, n_streams,
    OUTPUT_POLICY: tl.constexpr,
    LOGICAL_NUMEL: tl.constexpr,
    FIRST_MASK: tl.constexpr, RARE_LENGTH: tl.constexpr,
    BLOCK: tl.constexpr, N_LANES: tl.constexpr,
    N_STEPS: tl.constexpr, FIXED_WORDS: tl.constexpr,
):
    """Decode and fuse only the fixed Huffman streams.

    Overflow tails are overwritten later by the fallback tile kernel.  This
    kernel never scans the fallback lists, so its cost is O(elements).
    """
    block = tl.program_id(0)
    lanes = tl.arange(0, N_LANES)
    lane_index = block * N_LANES + lanes
    stream = lane_index

    a_word = tl.zeros((N_LANES,), tl.int32)
    a_shift = tl.zeros((N_LANES,), tl.int32)
    b_word = tl.zeros((N_LANES,), tl.int32)
    b_shift = tl.zeros((N_LANES,), tl.int32)

    a_window = tl.load(a_encoded + stream).to(tl.uint32).to(tl.uint64)
    a_window |= tl.load(a_encoded + n_streams + stream).to(tl.uint32).to(tl.uint64) << 32
    b_window = tl.load(b_encoded + stream).to(tl.uint32).to(tl.uint64)
    b_window |= tl.load(b_encoded + n_streams + stream).to(tl.uint32).to(tl.uint64) << 32

    a_center_value = tl.load(a_center).to(tl.int32)
    b_center_value = tl.load(b_center).to(tl.int32)
    alpha_value = tl.load(alpha).to(tl.float32)

    block_shift = (block * N_STEPS) & 255
    if (block + 1) * BLOCK <= LOGICAL_NUMEL:
        storage_offset = block * BLOCK + lanes
        for step in tl.range(0, N_STEPS, 2, loop_unroll_factor=6):
            a_word2 = tl.load(
                a_encoded + tl.minimum(a_word + 2, FIXED_WORDS - 1) * n_streams + stream,
            ).to(tl.uint32).to(tl.uint64)
            b_word2 = tl.load(
                b_encoded + tl.minimum(b_word + 2, FIXED_WORDS - 1) * n_streams + stream,
            ).to(tl.uint32).to(tl.uint64)

            a0, a0_len = decode_symbol(
                a_window >> a_shift, a_decode_table, a_center_value,
                FIRST_MASK, RARE_LENGTH,
            )
            b0, b0_len = decode_symbol(
                b_window >> b_shift, b_decode_table, b_center_value,
                FIRST_MASK, RARE_LENGTH,
            )
            logical_n0 = block * N_STEPS + step
            logical_k0 = (lanes + ((block_shift + step) & 255)) & 255
            logical_offset0 = logical_n0 * N_LANES + logical_k0
            sm_a0 = tl.load(a_sign_mantissa + storage_offset, cache_modifier='.cg')
            sm_b0 = tl.load(b_sign_mantissa + storage_offset, cache_modifier='.cg')
            a_left0 = pack_bf16(a0, sm_a0).to(tl.int16).to(tl.bfloat16, bitcast=True)
            b_left0 = pack_bf16(b0, sm_b0).to(tl.int16).to(tl.bfloat16, bitcast=True)
            _store_result(
                tl.math.fma(a_left0, alpha_value, b_left0), output, auxiliary,
                storage_offset, logical_offset0, True, True, OUTPUT_POLICY,
            )

            a_shift1 = a_shift + a0_len
            b_shift1 = b_shift + b0_len
            a1, a1_len = decode_symbol(
                a_window >> a_shift1, a_decode_table, a_center_value,
                FIRST_MASK, RARE_LENGTH,
            )
            b1, b1_len = decode_symbol(
                b_window >> b_shift1, b_decode_table, b_center_value,
                FIRST_MASK, RARE_LENGTH,
            )
            logical_n1 = logical_n0 + 1
            logical_k1 = (lanes + ((block_shift + step + 1) & 255)) & 255
            logical_offset1 = logical_n1 * N_LANES + logical_k1
            sm_a1 = tl.load(a_sign_mantissa + storage_offset + N_LANES, cache_modifier='.cg')
            sm_b1 = tl.load(b_sign_mantissa + storage_offset + N_LANES, cache_modifier='.cg')
            a_left1 = pack_bf16(a1, sm_a1).to(tl.int16).to(tl.bfloat16, bitcast=True)
            b_left1 = pack_bf16(b1, sm_b1).to(tl.int16).to(tl.bfloat16, bitcast=True)
            _store_result(
                tl.math.fma(a_left1, alpha_value, b_left1), output, auxiliary,
                storage_offset + N_LANES, logical_offset1, True, True, OUTPUT_POLICY,
            )

            a_word, a_shift, a_window = _advance_decode(
                a_word, a_shift, a_window, a0_len + a1_len,
                a_encoded, n_streams, stream, FIXED_WORDS,
            )

            b_word, b_shift, b_window = _advance_decode(
                b_word, b_shift, b_window, b0_len + b1_len,
                b_encoded, n_streams, stream, FIXED_WORDS,
            )
            storage_offset += 2 * N_LANES
    else:
        for step in tl.range(0, N_STEPS):
            a_value, a_length = decode_symbol(
                a_window >> a_shift, a_decode_table, a_center_value,
                FIRST_MASK, RARE_LENGTH,
            )
            b_value, b_length = decode_symbol(
                b_window >> b_shift, b_decode_table, b_center_value,
                FIRST_MASK, RARE_LENGTH,
            )
            offset, logical_offset, storage_valid, valid = _pointwise_location(
                block, step, lanes, n_elements, LOGICAL_NUMEL,
                BLOCK, N_LANES, N_STEPS,
            )
            sm_a = tl.load(a_sign_mantissa + offset, mask=storage_valid, other=0, cache_modifier='.cg')
            sm_b = tl.load(b_sign_mantissa + offset, mask=storage_valid, other=0, cache_modifier='.cg')
            a_left = pack_bf16(a_value, sm_a).to(tl.int16).to(tl.bfloat16, bitcast=True)
            b_left = pack_bf16(b_value, sm_b).to(tl.int16).to(tl.bfloat16, bitcast=True)
            _store_result(
                tl.math.fma(a_left, alpha_value, b_left), output, auxiliary,
                offset, logical_offset, valid & storage_valid, storage_valid, OUTPUT_POLICY,
            )

            a_word, a_shift, a_window = _advance_decode(
                a_word, a_shift, a_window, a_length,
                a_encoded, n_streams, stream, FIXED_WORDS,
            )

            b_word, b_shift, b_window = _advance_decode(
                b_word, b_shift, b_window, b_length,
                b_encoded, n_streams, stream, FIXED_WORDS,
            )


@triton.jit
def pointwise_scalar_mul_add_compressed_compressed_fallback_matrix_kernel(
    a_encoded, a_sign_mantissa, a_decode_table, a_center,
    a_stream_starts, a_stream_offsets,
    b_encoded, b_sign_mantissa, b_decode_table, b_center,
    b_stream_starts, b_stream_offsets,
    a_fallback_buffer, a_fallback_base,
    b_fallback_buffer, b_fallback_base,
    a_descriptor, a_fallback_count,
    b_descriptor, b_fallback_count,
    output, auxiliary, alpha,
    n_elements, n_streams,
    BUFFERED: tl.constexpr,
    OUTPUT_POLICY: tl.constexpr,
    LOGICAL_NUMEL: tl.constexpr,
    FIRST_MASK: tl.constexpr, RARE_LENGTH: tl.constexpr,
    BLOCK: tl.constexpr, N_LANES: tl.constexpr,
    N_STEPS: tl.constexpr, FIXED_WORDS: tl.constexpr,
    TILE: tl.constexpr,
):
    """Correct fused output for fallback streams in one stream-driven pass.

    Every stream is examined through its per-stream start/offset maps.  A tile
    returns early when it contains no fallback streams; otherwise those streams
    are rewritten with the fully correct fused value.
    """
    pid = tl.program_id(0)
    tile = pid * TILE + tl.arange(0, TILE)
    valid = tile < n_streams
    stream = tile

    a_start = tl.load(a_stream_starts + stream, mask=valid, other=N_STEPS).to(tl.int32)
    a_offset = tl.load(a_stream_offsets + stream, mask=valid, other=0).to(tl.int32)
    b_start = tl.load(b_stream_starts + stream, mask=valid, other=N_STEPS).to(tl.int32)
    b_offset = tl.load(b_stream_offsets + stream, mask=valid, other=0).to(tl.int32)

    needs_fix = valid & ((a_start < N_STEPS) | (b_start < N_STEPS))
    if tl.sum(needs_fix.to(tl.int32)) == 0:
        return

    if BUFFERED:
        a_real_base = tl.load(a_descriptor).to(tl.int32) + 9 * tl.load(a_fallback_count).to(tl.int32)
        b_real_base = tl.load(b_descriptor).to(tl.int32) + 9 * tl.load(b_fallback_count).to(tl.int32)
    else:
        a_real_base = a_fallback_base
        b_real_base = b_fallback_base

    a_word = tl.zeros((TILE,), tl.int32)
    a_shift = tl.zeros((TILE,), tl.int32)
    b_word = tl.zeros((TILE,), tl.int32)
    b_shift = tl.zeros((TILE,), tl.int32)

    a_word0 = tl.load(a_encoded + stream, mask=valid, other=0)
    a_word1 = tl.load(a_encoded + n_streams + stream, mask=valid, other=0)
    a_window = a_word0.to(tl.uint32).to(tl.uint64)
    a_window |= a_word1.to(tl.uint32).to(tl.uint64) << 32
    b_word0 = tl.load(b_encoded + stream, mask=valid, other=0)
    b_word1 = tl.load(b_encoded + n_streams + stream, mask=valid, other=0)
    b_window = b_word0.to(tl.uint32).to(tl.uint64)
    b_window |= b_word1.to(tl.uint32).to(tl.uint64) << 32

    a_center_value = tl.load(a_center).to(tl.int32)
    b_center_value = tl.load(b_center).to(tl.int32)
    alpha_value = tl.load(alpha).to(tl.float32)

    block = stream // N_LANES
    lane = stream - block * N_LANES
    block_shift = (block * N_STEPS) & 255
    storage_base = block * BLOCK + lane
    logical_base_n = block * N_STEPS

    for step in tl.range(0, N_STEPS):
        a_value, a_length = decode_symbol(
            a_window >> a_shift, a_decode_table, a_center_value,
            FIRST_MASK, RARE_LENGTH,
        )
        b_value, b_length = decode_symbol(
            b_window >> b_shift, b_decode_table, b_center_value,
            FIRST_MASK, RARE_LENGTH,
        )
        a_is_fb = needs_fix & (step >= a_start)
        b_is_fb = needs_fix & (step >= b_start)
        a_fb = tl.load(
            a_fallback_buffer + a_real_base + a_offset + (step - a_start),
            mask=a_is_fb, other=0,
        ).to(tl.int32)
        b_fb = tl.load(
            b_fallback_buffer + b_real_base + b_offset + (step - b_start),
            mask=b_is_fb, other=0,
        ).to(tl.int32)
        a_value = tl.where(a_is_fb, a_fb, a_value)
        b_value = tl.where(b_is_fb, b_fb, b_value)
        a_length = tl.where(a_is_fb, 0, a_length)
        b_length = tl.where(b_is_fb, 0, b_length)

        storage_offset = storage_base + step * N_LANES
        logical_n = logical_base_n + step
        logical_k = (lane + ((block_shift + step) & 255)) & 255
        logical_offset = logical_n * N_LANES + logical_k
        storage_valid = storage_offset < n_elements
        logical_valid = logical_offset < LOGICAL_NUMEL
        active = needs_fix & storage_valid & logical_valid
        storage_active = needs_fix & storage_valid
        sm_a = tl.load(a_sign_mantissa + storage_offset, mask=storage_active, other=0)
        sm_b = tl.load(b_sign_mantissa + storage_offset, mask=storage_active, other=0)
        a_left = pack_bf16(a_value, sm_a).to(tl.int16).to(tl.bfloat16, bitcast=True)
        b_left = pack_bf16(b_value, sm_b).to(tl.int16).to(tl.bfloat16, bitcast=True)
        _store_result(
            tl.math.fma(a_left, alpha_value, b_left), output, auxiliary,
            storage_offset, logical_offset, active, storage_active, OUTPUT_POLICY,
        )

        a_word, a_shift, a_window = _advance_decode(
            a_word, a_shift, a_window, a_length,
            a_encoded, n_streams, stream, FIXED_WORDS,
        )
        b_word, b_shift, b_window = _advance_decode(
            b_word, b_shift, b_window, b_length,
            b_encoded, n_streams, stream, FIXED_WORDS,
        )
