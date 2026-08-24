"""Generic Triton templates for pointwise compressed-tensor operations."""

import triton
from triton import language as tl

from ..codec.autotune import (
    DECODE_AUTOTUNE_CONFIGS,
    SCATTER_FALLBACK_AUTOTUNE_CONFIGS,
)
from ..codec.primitives import decode_symbol, pack_bf16


DENSE_OUTPUT = tl.constexpr(0)
COMPRESSED_OUTPUT = tl.constexpr(1)


@triton.jit
def _store_result(
    result, output, auxiliary, offset, mask,
    OUTPUT_POLICY: tl.constexpr,
):
    result = result.to(tl.bfloat16)
    if OUTPUT_POLICY == DENSE_OUTPUT:
        tl.store(output + offset, result, mask=mask)
    else:
        bits = result.to(tl.int16, bitcast=True).to(tl.int32)
        sign_mantissa = (bits & 0x7F) | ((bits >> 8) & 0x80)
        exponent = (bits >> 7) & 0xFF
        tl.store(output + offset, sign_mantissa.to(tl.uint8), mask=mask)
        tl.store(auxiliary + offset, exponent.to(tl.uint8), mask=mask)


@triton.autotune(
    configs=DECODE_AUTOTUNE_CONFIGS,
    key=["n_elements", "N_LANES", "N_STEPS", "FIXED_WORDS"],
)
@triton.jit
def binary_compressed_dense_kernel(
    encoded, sign_mantissa, other, output, auxiliary, decode_table,
    n_elements, n_streams, center,
    OP: tl.constexpr, OUTPUT_POLICY: tl.constexpr,
    FIRST_MASK: tl.constexpr, RARE_LENGTH: tl.constexpr,
    BLOCK: tl.constexpr, N_LANES: tl.constexpr,
    N_STEPS: tl.constexpr, FIXED_WORDS: tl.constexpr,
):
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

    for step in tl.range(0, N_STEPS, 2):
        offset = block * BLOCK + step * N_LANES + lanes
        valid = offset < n_elements
        value, length = decode_symbol(
            window >> shift, decode_table, center_value,
            FIRST_MASK, RARE_LENGTH,
        )
        sm = tl.load(sign_mantissa + offset, mask=valid, other=0)
        left = pack_bf16(value, sm).to(tl.int16).to(
            tl.bfloat16, bitcast=True
        )
        right = tl.load(other + offset, mask=valid, other=0.0)
        _store_result(
            OP(left, right), output, auxiliary, offset, valid, OUTPUT_POLICY,
        )

        shift1 = shift + tl.where(valid, length, 0)
        offset1 = offset + N_LANES
        valid1 = offset1 < n_elements
        value1, length1 = decode_symbol(
            window >> shift1, decode_table, center_value,
            FIRST_MASK, RARE_LENGTH,
        )
        sm1 = tl.load(sign_mantissa + offset1, mask=valid1, other=0)
        left1 = pack_bf16(value1, sm1).to(tl.int16).to(
            tl.bfloat16, bitcast=True
        )
        right1 = tl.load(other + offset1, mask=valid1, other=0.0)
        _store_result(
            OP(left1, right1), output, auxiliary,
            offset1, valid1, OUTPUT_POLICY,
        )

        next_shift = shift1 + tl.where(valid1, length1, 0)
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
    configs=SCATTER_FALLBACK_AUTOTUNE_CONFIGS,
    key=["n_elements", "N_LANES", "N_STEPS", "BLOCK"],
)
@triton.jit
def binary_fallback_kernel(
    bad_streams, bad_starts, fallback_offsets,
    fallback_buffer, fallback_base, metadata, descriptor, fallback_count,
    sign_mantissa, other, output, auxiliary, n_elements,
    OP: tl.constexpr, OUTPUT_POLICY: tl.constexpr, BUFFERED: tl.constexpr,
    TILE: tl.constexpr, BLOCK: tl.constexpr,
    N_LANES: tl.constexpr, N_STEPS: tl.constexpr,
):
    pid = tl.program_id(0)
    tile = pid * TILE + tl.arange(0, TILE)
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
        fallback_offset = tl.load(
            fallback_offsets + tile, mask=valid, other=0,
        )
    stream = stream.to(tl.int32)
    start = start.to(tl.int32)
    fallback_offset = fallback_offset.to(tl.int32)
    block = stream // N_LANES
    lane = stream - block * N_LANES
    for step in tl.range(0, N_STEPS):
        offset = block * BLOCK + step * N_LANES + lane
        active = valid & (step >= start) & (offset < n_elements)
        exponent = tl.load(
            fallback_buffer + fallback_base + fallback_offset + step - start,
            mask=active, other=0,
        ).to(tl.int32)
        sm = tl.load(sign_mantissa + offset, mask=active, other=0)
        left = pack_bf16(exponent, sm).to(tl.int16).to(
            tl.bfloat16, bitcast=True
        )
        right = tl.load(other + offset, mask=active, other=0.0)
        _store_result(
            OP(left, right), output, auxiliary,
            offset, active, OUTPUT_POLICY,
        )
