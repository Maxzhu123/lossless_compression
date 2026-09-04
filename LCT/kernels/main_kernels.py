import triton
from triton import language as tl

from codec.autotune import (
    COMPACT_BAD_STREAMS_AUTOTUNE_CONFIGS,
    COMPACT_EXTRA_AUTOTUNE_CONFIGS,
    DECODE_AUTOTUNE_CONFIGS,
    ENCODE_AUTOTUNE_CONFIGS,
    ESTIMATE_CENTER_AUTOTUNE_CONFIGS,
    SCATTER_FALLBACK_AUTOTUNE_CONFIGS,
)
from .primitives import pack_bf16


@triton.autotune(
    configs=ESTIMATE_CENTER_AUTOTUNE_CONFIGS,
    key=["SAMPLE_SIZE"],
)
@triton.jit
def _estimate_center_kernel(
    source_bits, center_out, size,
    SAMPLE_SIZE: tl.constexpr, STRIDE, PRECOMPUTED: tl.constexpr,
    IGNORE_ZERO: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK)
    total = tl.zeros((BLOCK,), tl.int32)
    s = tl.zeros((BLOCK,), tl.int32)
    n = tl.zeros((BLOCK,), tl.int32)
    for i in range(0, SAMPLE_SIZE, BLOCK):
        idx = i + offsets
        mask = idx < SAMPLE_SIZE
        pos = tl.minimum(idx * STRIDE, size - 1)
        value = tl.load(source_bits + pos, mask=mask, other=0).to(tl.int32)
        if PRECOMPUTED:
            exp = value - 127
        else:
            exp = ((value >> 7) & 0xFF) - 127
        if IGNORE_ZERO:
            nonzero = mask & (exp != -127)
            total += tl.where(nonzero, exp, 0)
            n += tl.where(nonzero, 1, 0)
        else:
            total += tl.where(mask, exp, 0)
            n += tl.where(mask, 1, 0)
    s = tl.sum(total, axis=0)
    n = tl.sum(n, axis=0)
    # If requested, ignore exact zeros when estimating the center.  This keeps
    # the center aligned with the nonzero component of zero-inflated
    # distributions.
    safe_n = tl.maximum(n, 1)
    center = tl.where(
        s >= 0,
        (s + safe_n // 2) // safe_n,
        -((-s + safe_n // 2) // safe_n),
    )
    center = tl.where(n > 0, center, 0)
    center = tl.minimum(tl.maximum(center, -128), 127)
    tl.store(center_out, center)


@triton.jit
def _shift_encoding_table_kernel(
    base_encode, center, shifted_encode,
    BLOCK: tl.constexpr,
):
    """Create a raw-exponent-byte-indexed encode table for one center."""
    idx = tl.arange(0, BLOCK)
    center_value = tl.load(center).to(tl.int32)
    zero_delta = (-127 - center_value) & 255
    raw_byte = idx
    exp = raw_byte - 127
    delta = (exp - center_value) & 255
    table_index = tl.where(
        exp == -127,
        0,
        delta + (delta < zero_delta).to(tl.int32),
    )
    packed = tl.load(base_encode + table_index).to(tl.uint32)
    tl.store(shifted_encode + raw_byte, packed)


@triton.jit
def _shift_decoding_table_kernel(
    base_decode, center, shifted_decode,
    BLOCK: tl.constexpr,
):
    """Create a decode table that stores unbiased exponents directly."""
    idx = tl.arange(0, BLOCK)
    center_value = tl.load(center).to(tl.int32)
    zero_delta = (-127 - center_value) & 255
    packed = tl.load(base_decode + idx).to(tl.int32)
    length = packed & 255
    symbol = (packed >> 8) & 255
    is_zero = symbol == 0
    delta = tl.where(symbol == 0, 0, symbol - (symbol <= zero_delta).to(tl.int32))
    delta = tl.where(delta >= 128, delta - 256, delta)
    exponent = tl.where(is_zero, -127, delta + center_value)
    shifted = tl.where(length == 0, 0, length | (exponent << 8))
    tl.store(shifted_decode + idx, shifted)


@triton.jit
def _encode_impl(
    source_bits, sign_mantissa, encoded, encode_table,
    extra_starts,
    n_elements, n_streams,
    PRECOMPUTED: tl.constexpr,
    LOGICAL_NUMEL: tl.constexpr,
    FIXED_WORDS: tl.constexpr,
    BLOCK: tl.constexpr, N_LANES: tl.constexpr, N_STEPS: tl.constexpr,
):
    """Encode one flattened contiguous block per program (1D storage).

    Storage offset ``block * BLOCK + step * N_LANES + lane`` maps to the
    flattened logical element ``logical_n * N_LANES + out_k`` with the
    row-dependent swizzle ``out_k = (lane + (logical_n & 255)) & 255``,
    which spreads neighbouring values across lanes.
    """
    block = tl.program_id(0)
    lanes = tl.arange(0, N_LANES)
    lane_index = block * N_LANES + lanes
    word = tl.zeros((N_LANES,), tl.int32)
    shift = tl.zeros((N_LANES,), tl.int32)
    word_value = tl.zeros((N_LANES,), tl.uint32)
    overflow = tl.zeros((N_LANES,), tl.int1)
    extra_start = tl.full((N_LANES,), N_STEPS, tl.int32)
    has_data = block * BLOCK + lanes < n_elements

    logical_n_base = block * N_STEPS
    # Hoisted swizzle: (block*N_STEPS + step) & 255 == (block_shift + step) & 255
    # with block_shift loop-invariant, so the hot loop only sees `step`.
    # (N_STEPS=256 -> block_shift=0; N_STEPS=128 -> 128*(block & 1).)
    block_shift = (block * N_STEPS) & 255
    block_base = block * BLOCK
    # Fully-valid blocks can skip all per-element validity/mask checks.
    full_block = (block + 1) * BLOCK <= LOGICAL_NUMEL
    if full_block:
        for step in tl.range(0, N_STEPS, 2, loop_unroll_factor=4):
            source_offset = block_base + step * N_LANES + lanes
            logical_n = logical_n_base + step
            shift0 = (block_shift + step) & 255
            shift1 = (block_shift + step + 1) & 255
            input_k0 = (lanes + shift0) & 255
            input_k1 = (lanes + shift1) & 255
            input_offset0 = logical_n * N_LANES + input_k0
            input_offset1 = (logical_n + 1) * N_LANES + input_k1
            value0 = tl.load(source_bits + input_offset0).to(tl.int32)
            value1 = tl.load(source_bits + input_offset1).to(tl.int32)
            if PRECOMPUTED:
                byte0 = value0 & 255
                byte1 = value1 & 255
            else:
                byte0 = (value0 >> 7) & 0xFF
                byte1 = (value1 >> 7) & 0xFF
                sm0 = (value0 & 0x7F) | ((value0 >> 8) & 0x80)
                sm1 = (value1 & 0x7F) | ((value1 >> 8) & 0x80)
                tl.store(sign_mantissa + source_offset, sm0.to(tl.uint8))
                tl.store(
                    sign_mantissa + source_offset + N_LANES,
                    sm1.to(tl.uint8),
                )
            packed0 = tl.load(encode_table + byte0).to(tl.uint32)
            packed1 = tl.load(encode_table + byte1).to(tl.uint32)
            length0 = (packed0 >> 20).to(tl.int32)
            length1 = (packed1 >> 20).to(tl.int32)
            length = length0 + length1
            code = (packed0 & 0xfffff) | ((packed1 & 0xfffff) << length0)

            new_word = word_value | (code << shift)
            crosses_word = shift + length >= 32
            pair_overflow = word * 32 + shift + length > FIXED_WORDS * 32
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
    else:
        for step in tl.range(0, N_STEPS, 2, loop_unroll_factor=4):
            source_offset = block_base + step * N_LANES + lanes
            valid0 = source_offset < n_elements
            valid1 = source_offset + N_LANES < n_elements
            logical_n = logical_n_base + step
            shift0 = (block_shift + step) & 255
            shift1 = (block_shift + step + 1) & 255
            input_k0 = (lanes + shift0) & 255
            input_k1 = (lanes + shift1) & 255
            input_offset0 = logical_n * N_LANES + input_k0
            input_offset1 = (logical_n + 1) * N_LANES + input_k1
            input_valid0 = input_offset0 < LOGICAL_NUMEL
            input_valid1 = input_offset1 < LOGICAL_NUMEL
            value0 = tl.load(
                source_bits + input_offset0, mask=input_valid0, other=0,
            ).to(tl.int32)
            value1 = tl.load(
                source_bits + input_offset1, mask=input_valid1, other=0,
            ).to(tl.int32)
            if PRECOMPUTED:
                byte0 = value0 & 255
                byte1 = value1 & 255
            else:
                byte0 = (value0 >> 7) & 0xFF
                byte1 = (value1 >> 7) & 0xFF
                sm0 = (value0 & 0x7F) | ((value0 >> 8) & 0x80)
                sm1 = (value1 & 0x7F) | ((value1 >> 8) & 0x80)
                tl.store(
                    sign_mantissa + source_offset, sm0.to(tl.uint8),
                    mask=valid0 & input_valid0,
                )
                tl.store(
                    sign_mantissa + source_offset + N_LANES,
                    sm1.to(tl.uint8), mask=valid1 & input_valid1,
                )
            packed0 = tl.load(encode_table + byte0).to(tl.uint32)
            packed1 = tl.load(encode_table + byte1).to(tl.uint32)
            packed0 = tl.where(input_valid0, packed0, 0)
            packed1 = tl.where(input_valid1, packed1, 0)
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
    configs=ENCODE_AUTOTUNE_CONFIGS,
    key=["n_elements", "N_LANES", "N_STEPS", "FIXED_WORDS"],
)
@triton.jit
def _encode_components_kernel(
    source_bits, sign_mantissa, encoded, encode_table,
    extra_starts, n_elements, n_streams,
    LOGICAL_NUMEL: tl.constexpr,
    FIXED_WORDS: tl.constexpr,
    BLOCK: tl.constexpr, N_LANES: tl.constexpr, N_STEPS: tl.constexpr,
):
    """Encode flattened precomputed exponent planes into 1D storage."""
    _encode_impl(
        source_bits, sign_mantissa, encoded, encode_table, extra_starts,
        n_elements, n_streams, True,
        LOGICAL_NUMEL,
        FIXED_WORDS, BLOCK, N_LANES, N_STEPS,
    )


@triton.autotune(
    configs=ENCODE_AUTOTUNE_CONFIGS,
    key=["n_elements", "N_LANES", "N_STEPS", "FIXED_WORDS"],
)
@triton.jit
def _encode_kernel(
    source_bits, sign_mantissa, encoded, encode_table,
    extra_starts, n_elements, n_streams,
    LOGICAL_NUMEL: tl.constexpr,
    FIXED_WORDS: tl.constexpr,
    BLOCK: tl.constexpr, N_LANES: tl.constexpr, N_STEPS: tl.constexpr,
):
    """Encode a flattened tensor through the 1D codec mapping."""
    _encode_impl(
        source_bits, sign_mantissa, encoded, encode_table, extra_starts,
        n_elements, n_streams, False,
        LOGICAL_NUMEL,
        FIXED_WORDS, BLOCK, N_LANES, N_STEPS,
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
    dense_offset = fallback_block_start + offset_prefix
    pos = block_start + prefix
    if BUFFERED:
        count = tl.load(final_counts).to(tl.int32)
        base = tl.load(allocation_descriptor).to(tl.int32)
        base_words = base // 4
        tl.store(metadata_buffer + base_words + pos, offs.to(tl.int32), mask=bad)
        tl.store(
            metadata_buffer + base_words + count + pos,
            dense_offset,
            mask=bad,
        )
        tl.store(bad_starts_out + base + 8 * count + pos, start, mask=bad)
    else:
        tl.store(bad_streams_out + pos, offs.to(tl.int32), mask=bad)
        tl.store(bad_starts_out + pos, start, mask=bad)
        tl.store(
            fallback_offsets_out + pos,
            dense_offset,
            mask=bad,
        )


@triton.jit
def _compact_extra_impl(
    source_bits,
    extra_streams, extra_starts,
    fallback_offsets, fallback_data,
    metadata_buffer, allocation_descriptor, final_counts, bad_count,
    n_elements,
    BUFFERED: tl.constexpr, PRECOMPUTED: tl.constexpr,
    LOGICAL_NUMEL: tl.constexpr,
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
    tail_steps = N_STEPS - start
    max_tail = tl.max(tl.where(valid, tail_steps, 0), axis=0)
    # Hoisted swizzle (see _encode_impl): block_shift is loop-invariant.
    block_shift = (block * N_STEPS) & 255
    block_base = block * BLOCK
    for step in tl.range(0, max_tail):
        source_offset = block_base + (step + start) * N_LANES + lane
        active = valid & (step < tail_steps) & (source_offset < n_elements)
        logical_n = block * N_STEPS + step + start
        logical_k = (lane + ((block_shift + step + start) & 255)) & 255
        input_offset = logical_n * N_LANES + logical_k
        input_active = active & (input_offset < LOGICAL_NUMEL)
        value = tl.load(
            source_bits + input_offset, mask=input_active, other=0,
        ).to(tl.int32)
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
    configs=COMPACT_EXTRA_AUTOTUNE_CONFIGS,
    key=["n_elements", "N_LANES", "N_STEPS", "BLOCK"],
)
@triton.jit
def _compact_components_extra_kernel(
    source_bits, extra_streams, extra_starts,
    fallback_offsets, fallback_data, metadata_buffer,
    allocation_descriptor, final_counts, bad_count, n_elements,
    BUFFERED: tl.constexpr, LOGICAL_NUMEL: tl.constexpr,
    BLOCK: tl.constexpr, N_LANES: tl.constexpr,
    N_STEPS: tl.constexpr, TILE: tl.constexpr,
):
    """Compact overflow exponents from flattened precomputed planes."""
    _compact_extra_impl(
        source_bits, extra_streams, extra_starts, fallback_offsets,
        fallback_data, metadata_buffer, allocation_descriptor,
        final_counts, bad_count, n_elements, BUFFERED, True,
        LOGICAL_NUMEL,
        BLOCK, N_LANES, N_STEPS, TILE,
    )


@triton.autotune(
    configs=COMPACT_EXTRA_AUTOTUNE_CONFIGS,
    key=["n_elements", "N_LANES", "N_STEPS", "BLOCK"],
)
@triton.jit
def _compact_extra_kernel(
    source_bits, extra_streams, extra_starts,
    fallback_offsets, fallback_data, metadata_buffer,
    allocation_descriptor, final_counts, bad_count, n_elements,
    BUFFERED: tl.constexpr, LOGICAL_NUMEL: tl.constexpr,
    BLOCK: tl.constexpr, N_LANES: tl.constexpr,
    N_STEPS: tl.constexpr, TILE: tl.constexpr,
):
    """Compact flattened overflow values through the 1D source mapping."""
    _compact_extra_impl(
        source_bits, extra_streams, extra_starts, fallback_offsets,
        fallback_data, metadata_buffer, allocation_descriptor,
        final_counts, bad_count, n_elements, BUFFERED, False,
        LOGICAL_NUMEL,
        BLOCK, N_LANES, N_STEPS, TILE,
    )


@triton.autotune(
    configs=DECODE_AUTOTUNE_CONFIGS,
    key=["n_elements", "N_STEPS", "FIXED_WORDS", "ON_DEMAND"],
)
@triton.jit
def _decode_kernel(
    encoded, sign_mantissa, output,
    decode_table, n_elements, n_streams, center,
    LOGICAL_NUMEL: tl.constexpr,
    FIRST_MASK: tl.constexpr, RARE_LENGTH: tl.constexpr,
    BLOCK: tl.constexpr,N_LANES: tl.constexpr, N_STEPS: tl.constexpr, FIXED_WORDS: tl.constexpr,
    ON_DEMAND: tl.constexpr,
):
    # One program decodes one codec block; each lane owns one fixed Huffman stream.
    block = tl.program_id(0)
    lanes = tl.arange(0, N_LANES)
    lane_index = block * N_LANES + lanes
    # word/shift form the current 64-bit decoding window into the fixed payload.
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
    zero_delta = (-127 - center_value) & 255
    # Fast path: fully-contained blocks skip all per-element validity checks.
    # The swizzled logical mapping (storage -> flattened output) still applies.
    # Hoisted swizzle: shift depends only on step + loop-invariant block_shift.
    block_shift = (block * N_STEPS) & 255
    block_base = block * BLOCK
    if (block + 1) * BLOCK <= LOGICAL_NUMEL:
        storage_offset = block_base + lanes
        for step in tl.range(0, N_STEPS, 2, flatten=True, warp_specialize=True):
            logical_n0 = block * N_STEPS + step
            logical_n1 = logical_n0 + 1
            out_k0 = (lanes + ((block_shift + step) & 255)) & 255
            out_k1 = (lanes + ((block_shift + step + 1) & 255)) & 255
            output_offset0 = logical_n0 * N_LANES + out_k0
            output_offset1 = logical_n1 * N_LANES + out_k1
            if not ON_DEMAND:
                word2_prefetch = tl.load(
                    encoded + tl.minimum(word + 2, FIXED_WORDS - 1) * n_streams + lane_index
                ).to(tl.uint32).to(tl.uint64)
            # Decode symbol 0 from the current prefix bits.
            current = window >> shift
            first = tl.load(
                decode_table + (current & FIRST_MASK).to(tl.int32),
                cache_modifier='.ca',
            )
            first_length = first & 255
            continuation = first_length == 0
            length = tl.where(continuation, RARE_LENGTH + 8, first_length)
            symbol = ((current >> RARE_LENGTH) & 255).to(tl.int32)
            is_zero = symbol == 0
            delta = tl.where(symbol == 0, 0, symbol - (symbol <= zero_delta).to(tl.int32))
            delta = tl.where(delta >= 128, delta - 256, delta)
            escaped_value = tl.where(is_zero, -127, delta + center_value)
            value = tl.where(continuation, escaped_value, first >> 8)
            sm = tl.load(sign_mantissa + storage_offset, cache_modifier='.cg')
            packed = (
                (((value.to(tl.int32) + 127) & 255) << 7)
                | (sm.to(tl.int32) & 0x7F)
                | ((sm.to(tl.int32) & 0x80) << 8)
            )
            tl.store(output + output_offset0, packed.to(tl.int16), cache_modifier='.cs')
            # Decode symbol 1 at the bit offset after symbol 0.
            shift1 = shift + length
            current1 = window >> shift1
            first1 = tl.load(
                decode_table + (current1 & FIRST_MASK).to(tl.int32),
                cache_modifier='.ca',
            )
            first_length1 = first1 & 255
            continuation1 = first_length1 == 0
            length1 = tl.where(continuation1, RARE_LENGTH + 8, first_length1)
            symbol1 = ((current1 >> RARE_LENGTH) & 255).to(tl.int32)
            is_zero1 = symbol1 == 0
            delta1 = tl.where(symbol1 == 0, 0, symbol1 - (symbol1 <= zero_delta).to(tl.int32))
            delta1 = tl.where(delta1 >= 128, delta1 - 256, delta1)
            escaped_value1 = tl.where(is_zero1, -127, delta1 + center_value)
            value1 = tl.where(continuation1, escaped_value1, first1 >> 8)
            sm1 = tl.load(sign_mantissa + storage_offset + N_LANES, cache_modifier='.cg')
            packed1 = (
                (((value1.to(tl.int32) + 127) & 255) << 7)
                | (sm1.to(tl.int32) & 0x7F)
                | ((sm1.to(tl.int32) & 0x80) << 8)
            )
            tl.store(
                output + output_offset1,
                packed1.to(tl.int16),
                cache_modifier='.cs',
            )
            # Advance the 64-bit window when both symbols cross a 32-bit word boundary.
            next_shift = shift1 + length1
            crosses_word = next_shift >= 32
            if ON_DEMAND:
                next_word = tl.load(
                    encoded + tl.minimum(word + 2, FIXED_WORDS - 1) * n_streams + lane_index,
                    mask=crosses_word, other=0,
                ).to(tl.uint32).to(tl.uint64)
                next_window = (window >> 32) | (next_word << 32)
            else:
                next_window = (window >> 32) | (word2_prefetch << 32)
            window = tl.where(crosses_word, next_window, window)
            word += crosses_word
            shift = tl.where(crosses_word, next_shift - 32, next_shift)
            storage_offset += 2 * N_LANES
    else:
        # Tail path: map the codec storage block back to flattened coordinates
        # with bounds masks.
        storage_offset = block_base + lanes
        logical_n = block * N_STEPS
        for step in tl.range(0, N_STEPS, 2):
            word2_prefetch = tl.load(
                encoded + tl.minimum(word + 2, FIXED_WORDS - 1) * n_streams + lane_index
            ).to(tl.uint32).to(tl.uint64)
            logical_n0 = logical_n + step
            logical_n1 = logical_n0 + 1
            out_k0 = (lanes + ((block_shift + step) & 255)) & 255
            out_k1 = (lanes + ((block_shift + step + 1) & 255)) & 255
            output_offset0 = logical_n0 * N_LANES + out_k0
            output_offset1 = logical_n1 * N_LANES + out_k1
            storage_valid = storage_offset < n_elements
            valid = output_offset0 < LOGICAL_NUMEL
            # Decode symbol 0 from the current prefix bits.
            current = window >> shift
            first = tl.load(
                decode_table + (current & FIRST_MASK).to(tl.int32),
                cache_modifier='.ca',
            )
            first_length = first & 255
            continuation = first_length == 0
            length = tl.where(continuation, RARE_LENGTH + 8, first_length)
            symbol = ((current >> RARE_LENGTH) & 255).to(tl.int32)
            is_zero = symbol == 0
            delta = tl.where(symbol == 0, 0, symbol - (symbol <= zero_delta).to(tl.int32))
            delta = tl.where(delta >= 128, delta - 256, delta)
            escaped_value = tl.where(is_zero, -127, delta + center_value)
            value = tl.where(continuation, escaped_value, first >> 8)
            sm = tl.load(
                sign_mantissa + storage_offset,
                mask=storage_valid, other=0, cache_modifier='.cg',
            )
            packed = (
                (((value.to(tl.int32) + 127) & 255) << 7)
                | (sm.to(tl.int32) & 0x7F)
                | ((sm.to(tl.int32) & 0x80) << 8)
            )
            tl.store(output + output_offset0, packed.to(tl.int16), mask=valid, cache_modifier='.cs')
            shift1 = shift + tl.where(storage_valid, length, 0)
            current1 = window >> shift1
            first1 = tl.load(
                decode_table + (current1 & FIRST_MASK).to(tl.int32),
                cache_modifier='.ca',
            )
            first_length1 = first1 & 255
            continuation1 = first_length1 == 0
            length1 = tl.where(continuation1, RARE_LENGTH + 8, first_length1)
            symbol1 = ((current1 >> RARE_LENGTH) & 255).to(tl.int32)
            is_zero1 = symbol1 == 0
            delta1 = tl.where(symbol1 == 0, 0, symbol1 - (symbol1 <= zero_delta).to(tl.int32))
            delta1 = tl.where(delta1 >= 128, delta1 - 256, delta1)
            escaped_value1 = tl.where(is_zero1, -127, delta1 + center_value)
            value1 = tl.where(continuation1, escaped_value1, first1 >> 8)
            storage_offset1 = storage_offset + N_LANES
            storage_valid1 = storage_offset1 < n_elements
            valid1 = output_offset1 < LOGICAL_NUMEL
            sm1 = tl.load(
                sign_mantissa + storage_offset1,
                mask=storage_valid1, other=0, cache_modifier='.cg',
            )
            packed1 = (
                (((value1.to(tl.int32) + 127) & 255) << 7)
                | (sm1.to(tl.int32) & 0x7F)
                | ((sm1.to(tl.int32) & 0x80) << 8)
            )
            tl.store(
                output + output_offset1,
                packed1.to(tl.int16), mask=valid1, cache_modifier='.cs',
            )
            next_shift = shift1 + tl.where(storage_valid1, length1, 0)
            crosses_word = next_shift >= 32
            next_window = (window >> 32) | (word2_prefetch << 32)
            window = tl.where(crosses_word, next_window, window)
            word += crosses_word
            shift = tl.where(crosses_word, next_shift - 32, next_shift)
            storage_offset += 2 * N_LANES


@triton.autotune(
    configs=SCATTER_FALLBACK_AUTOTUNE_CONFIGS,
    key=["n_elements", "N_LANES", "N_STEPS", "BLOCK"],
)
@triton.jit
def _scatter_blocked_fallback_kernel(
    bad_streams, bad_starts, fallback_offsets,
    fallback_buffer, fallback_base, metadata, descriptor, fallback_count,
    sign_mantissa, output, n_elements,
    BUFFERED: tl.constexpr,
    LOGICAL_NUMEL: tl.constexpr,
    TILE: tl.constexpr, BLOCK: tl.constexpr,
    N_LANES: tl.constexpr, N_STEPS: tl.constexpr,
):
    """Overwrite mapped decode output with compact fallback stream tails."""
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
        fallback_offset = tl.load(fallback_offsets + tile, mask=valid, other=0)
    stream = stream.to(tl.int32)
    start = start.to(tl.int32)
    fallback_offset = fallback_offset.to(tl.int32)
    block = stream // N_LANES
    lane = stream % N_LANES
    # Hoisted swizzle (see _encode_impl): block_shift is loop-invariant.
    block_shift = (block * N_STEPS) & 255
    block_base = block * BLOCK
    for step in tl.range(0, N_STEPS):
        storage_offset = block_base + step * N_LANES + lane
        logical_n = block * N_STEPS + step
        logical_k = (lane + ((block_shift + step) & 255)) & 255
        logical_offset = logical_n * N_LANES + logical_k
        active = (
            valid & (step >= start) & (storage_offset < n_elements)
            & (logical_offset < LOGICAL_NUMEL)
        )
        exponent = tl.load(
            fallback_buffer + fallback_base + fallback_offset + step - start,
            mask=active, other=0,
        ).to(tl.int32)
        sm = tl.load(sign_mantissa + storage_offset, mask=active, other=0)
        packed = pack_bf16(exponent, sm)
        tl.store(output + logical_offset, packed.to(tl.int16), mask=active)
