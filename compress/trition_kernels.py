import triton
from triton import language as tl

from .codec.autotune import (
    COMPACT_BAD_STREAMS_AUTOTUNE_CONFIGS,
    COMPACT_EXTRA_AUTOTUNE_CONFIGS,
    DECODE_AUTOTUNE_CONFIGS,
    ENCODE_AUTOTUNE_CONFIGS,
    ESTIMATE_CENTER_AUTOTUNE_CONFIGS,
    SCATTER_FALLBACK_AUTOTUNE_CONFIGS,
)


@triton.autotune(
    configs=ESTIMATE_CENTER_AUTOTUNE_CONFIGS,
    key=["SAMPLE_SIZE"],
)
@triton.jit
def _estimate_center_kernel(
    source_bits, center_out, size,
    SAMPLE_SIZE: tl.constexpr, STRIDE, PRECOMPUTED: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK)
    total = tl.zeros((BLOCK,), tl.int32)
    for i in range(0, SAMPLE_SIZE, BLOCK):
        idx = i + offsets
        mask = idx < SAMPLE_SIZE
        pos = tl.minimum(idx * STRIDE, size - 1)
        value = tl.load(source_bits + pos, mask=mask, other=0).to(tl.int32)
        if PRECOMPUTED:
            exp = value - 127
        else:
            exp = ((value >> 7) & 0xFF) - 127
        total += tl.where(mask, exp, 0)
    s = tl.sum(total, axis=0)
    # Round half away from zero, clamp, and suppress tiny sampling noise.
    center = tl.where(
        s >= 0,
        (s + SAMPLE_SIZE // 2) // SAMPLE_SIZE,
        -((-s + SAMPLE_SIZE // 2) // SAMPLE_SIZE),
    )
    center = tl.minimum(tl.maximum(center, -128), 127)
    center = tl.where(tl.abs(center) <= 1, 0, center)
    tl.store(center_out, center)


@triton.autotune(
    configs=ENCODE_AUTOTUNE_CONFIGS,
    key=["n_elements", "N_LANES", "N_STEPS", "FIXED_WORDS"],
)
@triton.jit
def _encode_kernel(
    source_bits, sign_mantissa, encoded, encode_table,
    center, extra_starts,
    n_elements, n_streams,
    PRECOMPUTED: tl.constexpr, FIXED_WORDS: tl.constexpr,
    BLOCK: tl.constexpr, N_LANES: tl.constexpr, N_STEPS: tl.constexpr,
):
    block = tl.program_id(0)
    lanes = tl.arange(0, N_LANES)
    lane_index = block * N_LANES + lanes
    word = tl.zeros((N_LANES,), tl.int32)
    shift = tl.zeros((N_LANES,), tl.int32)
    word_value = tl.zeros((N_LANES,), tl.uint32)
    overflow = tl.zeros((N_LANES,), tl.int1)
    extra_start = tl.full((N_LANES,), N_STEPS, tl.int32)
    has_data = block * BLOCK + lanes < n_elements
    center_value = tl.load(center).to(tl.int32)

    for step in tl.range(0, N_STEPS, 2, loop_unroll_factor=4):
        source_offset = block * BLOCK + step * N_LANES + lanes
        valid0 = source_offset < n_elements
        valid1 = source_offset + N_LANES < n_elements
        value0 = tl.load(
            source_bits + tl.minimum(source_offset, n_elements - 1),
            mask=valid0, other=0,
        ).to(tl.int32)
        value1 = tl.load(
            source_bits + tl.minimum(source_offset + N_LANES, n_elements - 1),
            mask=valid1, other=0,
        ).to(tl.int32)
        if PRECOMPUTED:
            exp0 = value0 - 127
            exp1 = value1 - 127
        else:
            exp0 = ((value0 >> 7) & 0xFF) - 127
            exp1 = ((value1 >> 7) & 0xFF) - 127
            sm0 = (value0 & 0x7F) | ((value0 >> 8) & 0x80)
            sm1 = (value1 & 0x7F) | ((value1 >> 8) & 0x80)
            tl.store(sign_mantissa + source_offset, sm0.to(tl.uint8), mask=valid0)
            tl.store(
                sign_mantissa + source_offset + N_LANES,
                sm1.to(tl.uint8), mask=valid1,
            )
        packed0 = tl.load(
            encode_table + ((exp0 - center_value) & 255)
        ).to(tl.uint32)
        packed1 = tl.load(
            encode_table + ((exp1 - center_value) & 255)
        ).to(tl.uint32)
        length0 = (packed0 >> 20).to(tl.int32)
        length1 = (packed1 >> 20).to(tl.int32)
        length = length0 + length1
        code = (packed0 & 0xfffff) | ((packed1 & 0xfffff) << length0)

        new_word = word_value | (code << shift)
        crosses_word = shift + length >= 32
        pair_overflow = (word * 32 + shift + length > FIXED_WORDS * 32) & valid0
        first_overflow = (~overflow) & pair_overflow

        extra_start = tl.where(first_overflow, step, extra_start)

        store_value = tl.where(first_overflow, word_value, new_word)
        safe_word = tl.minimum(word, FIXED_WORDS - 1)
        tl.store(
            encoded + safe_word * n_streams + lane_index,
            store_value,
            mask=crosses_word & (word < FIXED_WORDS),
        )
        word_value = tl.where(crosses_word, code >> (32 - shift), new_word)
        word += tl.where(crosses_word, 1, 0)
        shift = tl.where(crosses_word, shift + length - 32, shift + length)
        overflow |= pair_overflow

    safe_word = tl.minimum(word, FIXED_WORDS - 1)
    tl.store(
        encoded + safe_word * n_streams + lane_index,
        word_value,
        mask=has_data & (word < FIXED_WORDS) & (shift != 0),
    )
    tl.store(
        extra_starts + lane_index,
        tl.where(has_data & overflow, extra_start, 255),
    )


@triton.autotune(
    configs=COMPACT_BAD_STREAMS_AUTOTUNE_CONFIGS,
    key=["n_streams", "steps"],
    restore_value=["bad_count", "fallback_total"],
)
@triton.jit
def _count_bad_streams_kernel(
    extra_starts,
    bad_count, fallback_total, n_streams, steps,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_streams
    start = tl.load(extra_starts + offs, mask=mask, other=255)
    bad = mask & (start != 255)
    lengths = tl.where(bad, steps - start, 0)
    cnt = tl.sum(bad.to(tl.int32), axis=0)
    total_len = tl.sum(lengths, axis=0)
    tl.atomic_add(bad_count, cnt, mask=cnt != 0)
    tl.atomic_add(fallback_total, total_len, mask=total_len != 0)


@triton.autotune(
    configs=COMPACT_BAD_STREAMS_AUTOTUNE_CONFIGS,
    key=["n_streams", "steps"],
    restore_value=["bad_count", "fallback_total"],
)
@triton.jit
def _compact_bad_streams_kernel(
    extra_starts,
    bad_streams_out, bad_starts_out, fallback_offsets_out,
    metadata_buffer, allocation_descriptor, final_counts,
    bad_count, fallback_total, n_streams, steps,
    BUFFERED: tl.constexpr, BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_streams
    start = tl.load(extra_starts + offs, mask=mask, other=255)
    bad = mask & (start != 255)
    lengths = tl.where(bad, steps - start, 0)
    cnt = tl.sum(bad.to(tl.int32), axis=0)
    total_len = tl.sum(lengths, axis=0)
    block_start = tl.atomic_add(bad_count, cnt, mask=cnt != 0)
    fallback_block_start = tl.atomic_add(
        fallback_total, total_len, mask=total_len != 0
    )
    prefix = tl.cumsum(bad.to(tl.int32), axis=0) - bad.to(tl.int32)
    offset_prefix = tl.cumsum(lengths, axis=0) - lengths
    pos = block_start + prefix
    if BUFFERED:
        count = tl.load(final_counts).to(tl.int32)
        base = tl.load(allocation_descriptor).to(tl.int32)
        base_words = base // 4
        tl.store(metadata_buffer + base_words + pos, offs.to(tl.int32), mask=bad)
        tl.store(
            metadata_buffer + base_words + count + pos,
            fallback_block_start + offset_prefix,
            mask=bad,
        )
        tl.store(bad_starts_out + base + 8 * count + pos, start, mask=bad)
    else:
        tl.store(bad_streams_out + pos, offs.to(tl.int32), mask=bad)
        tl.store(bad_starts_out + pos, start, mask=bad)
        tl.store(
            fallback_offsets_out + pos,
            fallback_block_start + offset_prefix,
            mask=bad,
        )


@triton.autotune(
    configs=COMPACT_EXTRA_AUTOTUNE_CONFIGS,
    key=["n_elements", "N_LANES", "N_STEPS", "BLOCK"],
)
@triton.jit
def _compact_extra_kernel(
    source_bits,
    extra_streams, extra_starts,
    fallback_offsets, fallback_data,
    metadata_buffer, allocation_descriptor, final_counts, bad_count,
    n_elements,
    BUFFERED: tl.constexpr, PRECOMPUTED: tl.constexpr,
    BLOCK: tl.constexpr, N_LANES: tl.constexpr, N_STEPS: tl.constexpr, TILE: tl.constexpr,
):
    pid = tl.program_id(0)
    tile = pid * TILE + tl.arange(0, TILE)
    if BUFFERED:
        count = tl.load(final_counts).to(tl.int32)
    else:
        count = tl.load(bad_count).to(tl.int32)
    if pid * TILE >= count:
        return
    valid = tile < count
    if BUFFERED:
        base = tl.load(allocation_descriptor).to(tl.int32)
        base_words = base // 4
        stream = tl.load(metadata_buffer + base_words + tile, mask=valid, other=0).to(tl.int32)
        start = tl.load(
            extra_starts + base + 8 * count + tile,
            mask=valid,
            other=N_STEPS,
        ).to(tl.int32)
        fallback_offset = tl.load(
            metadata_buffer + base_words + count + tile,
            mask=valid,
            other=0,
        ).to(tl.int32)
        fallback_base = base + 9 * count
    else:
        stream = tl.load(extra_streams + tile, mask=valid, other=0).to(tl.int32)
        start = tl.load(extra_starts + tile, mask=valid, other=N_STEPS).to(tl.int32)
        fallback_offset = tl.load(fallback_offsets + tile, mask=valid, other=0).to(tl.int32)
    block = stream // N_LANES
    lane = stream - block * N_LANES
    for step in tl.range(0, N_STEPS):
        source_offset = block * BLOCK + (step + start) * N_LANES + lane
        active = valid & (step < (N_STEPS - start)) & (source_offset < n_elements)
        value = tl.load(source_bits + source_offset, mask=active, other=0).to(tl.int32)
        if PRECOMPUTED:
            values = (value - 127).to(tl.int8)
        else:
            values = (((value >> 7) & 0xFF) - 127).to(tl.int8)
        if BUFFERED:
            tl.store(
                fallback_data + fallback_base + fallback_offset + step,
                values,
                mask=active,
            )
        else:
            tl.store(fallback_data + fallback_offset + step, values, mask=active)


@triton.autotune(
    configs=SCATTER_FALLBACK_AUTOTUNE_CONFIGS,
    key=["n_elements", "N_LANES", "N_STEPS", "BLOCK"],
)
@triton.jit
def _scatter_fallback_kernel(
    bad_streams, bad_starts,
    fallback_offsets, fallback_buffer, fallback_base,
    metadata_buffer, allocation_descriptor, fallback_count,
    sign_mantissa,
    output, n_elements,
    BUFFERED: tl.constexpr,
    TILE: tl.constexpr, BLOCK: tl.constexpr, N_LANES: tl.constexpr, N_STEPS: tl.constexpr,
):
    pid = tl.program_id(0)
    tile = pid * TILE + tl.arange(0, TILE)
    count = tl.load(fallback_count).to(tl.int32)
    if pid * TILE >= count:
        return
    valid = tile < count
    if BUFFERED:
        base = tl.load(allocation_descriptor).to(tl.int32)
        base_words = base // 4
        stream = tl.load(metadata_buffer + base_words + tile, mask=valid, other=0).to(tl.int32)
        start = tl.load(
            bad_starts + base + 8 * count + tile,
            mask=valid,
            other=N_STEPS,
        ).to(tl.int32)
        fallback_offset = tl.load(
            metadata_buffer + base_words + count + tile,
            mask=valid,
            other=0,
        ).to(tl.int32)
        fallback_base = base + 9 * count
    else:
        stream = tl.load(bad_streams + tile, mask=valid, other=0).to(tl.int32)
        start = tl.load(bad_starts + tile, mask=valid, other=N_STEPS).to(tl.int32)
        fallback_offset = tl.load(fallback_offsets + tile, mask=valid, other=0).to(tl.int32)
    block = stream // N_LANES
    lane = stream - block * N_LANES
    for step in tl.range(0, N_STEPS):
        output_offset = block * BLOCK + step * N_LANES + lane
        active = valid & (step >= start) & (output_offset < n_elements)
        values = tl.load(
            fallback_buffer + fallback_base + fallback_offset + step - start,
            mask=active,
            other=0,
        ).to(tl.int32)
        sm = tl.load(
            sign_mantissa + output_offset, mask=active, other=0
        ).to(tl.int32)
        packed = (
            (((values + 127) & 255) << 7)
            | (sm & 0x7F)
            | ((sm & 0x80) << 8)
        )
        tl.store(output + output_offset, packed.to(tl.int16), mask=active)


@triton.jit
def _pack_bf16(value, sm):
    return (
        (((value.to(tl.int32) + 127) & 255) << 7)
        | (sm.to(tl.int32) & 0x7F)
        | ((sm.to(tl.int32) & 0x80) << 8)
    )


@triton.autotune(
    configs=DECODE_AUTOTUNE_CONFIGS,
    key=["n_elements", "N_LANES", "N_STEPS", "FIXED_WORDS"],
)
@triton.jit
def _decode_kernel(
    encoded, sign_mantissa, output,
    decode_table, n_elements, n_streams,
    center,
    SKIP_FALLBACK: tl.constexpr,
    FIRST_MASK: tl.constexpr, RARE_LENGTH: tl.constexpr,
    BLOCK: tl.constexpr,N_LANES: tl.constexpr, N_STEPS: tl.constexpr, FIXED_WORDS: tl.constexpr,
):
    block = tl.program_id(0)
    lanes = tl.arange(0, N_LANES)
    lane_index = block * N_LANES + lanes
    word = tl.zeros((N_LANES,), tl.int32)
    shift = tl.zeros((N_LANES,), tl.int32)
    word0 = tl.load(
        encoded + word * n_streams + lane_index
    ).to(tl.uint32).to(tl.uint64)
    word1 = tl.load(
        encoded + (word + 1) * n_streams + lane_index
    ).to(tl.uint32).to(tl.uint64)
    window = word0 | (word1 << 32)
    center_value = tl.load(center).to(tl.int32)

    for step in tl.range(0, N_STEPS, 2):
        output_offset = block * BLOCK + step * N_LANES + lanes
        valid = output_offset < n_elements

        current = window >> shift
        first = tl.load(decode_table + (current & FIRST_MASK).to(tl.int32))
        first_length = first & 255
        continuation = first_length == 0
        length = tl.where(continuation, RARE_LENGTH + 8, first_length)
        tail = ((current >> RARE_LENGTH) & 255).to(tl.int32)
        tail = tl.where(tail >= 128, tail - 256, tail)
        value = tl.where(continuation, tail, first >> 8) + center_value
        sm = tl.load(sign_mantissa + output_offset, mask=valid, other=0)
        packed = _pack_bf16(value, sm)
        tl.store(output + output_offset, packed.to(tl.int16), mask=valid)

        shift1 = shift + tl.where(valid, length, 0)
        current1 = window >> shift1
        first1 = tl.load(decode_table + (current1 & FIRST_MASK).to(tl.int32))
        first_length1 = first1 & 255
        continuation1 = first_length1 == 0
        length1 = tl.where(continuation1, RARE_LENGTH + 8, first_length1)
        tail1 = ((current1 >> RARE_LENGTH) & 255).to(tl.int32)
        tail1 = tl.where(tail1 >= 128, tail1 - 256, tail1)
        value1 = tl.where(continuation1, tail1, first1 >> 8) + center_value
        valid1 = output_offset + N_LANES < n_elements
        sm1 = tl.load(
            sign_mantissa + output_offset + N_LANES, mask=valid1, other=0
        )
        packed1 = _pack_bf16(value1, sm1)
        tl.store(
            output + output_offset + N_LANES,
            packed1.to(tl.int16), mask=valid1,
        )

        next_shift = shift1 + tl.where(valid1, length1, 0)
        crosses_word = next_shift >= 32
        if SKIP_FALLBACK:
            word2 = tl.load(
                encoded + tl.minimum(word + 2, FIXED_WORDS - 1) * n_streams
                + lane_index,
                mask=crosses_word,
                other=0,
            ).to(tl.uint32).to(tl.uint64)
        else:
            word2 = tl.load(
                encoded + (word + 2) * n_streams + lane_index,
                mask=crosses_word,
                other=0,
            ).to(tl.uint32).to(tl.uint64)
        next_window = (window >> 32) | (word2 << 32)
        window = tl.where(crosses_word, next_window, window)
        word += crosses_word
        shift = tl.where(crosses_word, next_shift - 32, next_shift)
