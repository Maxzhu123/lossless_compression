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
    result, output, auxiliary, storage_offset, logical_offset,
    logical_mask, storage_mask,
    OUTPUT_POLICY: tl.constexpr,
):
    """Round an operation result to BF16 and emit dense or component output."""
    result = result.to(tl.bfloat16)
    if OUTPUT_POLICY == DENSE_OUTPUT:
        tl.store(output + logical_offset, result, mask=logical_mask, cache_modifier='.cs')
    else:
        result = tl.where(logical_mask, result, 0.0).to(tl.bfloat16)
        bits = result.to(tl.int16, bitcast=True).to(tl.int32)
        sign_mantissa = (bits & 0x7F) | ((bits >> 8) & 0x80)
        exponent = (bits >> 7) & 0xFF
        tl.store(
            output + storage_offset, sign_mantissa.to(tl.uint8),
            mask=storage_mask,
        )
        tl.store(
            auxiliary + logical_offset, exponent.to(tl.uint8),
            mask=logical_mask,
        )


@triton.jit
def _pointwise_location(
    block, step, lane, n_elements,
    MATRIX_N: tl.constexpr, MATRIX_K: tl.constexpr, MATRIX_NUMEL: tl.constexpr,
    K_TILE_BLOCKS: tl.constexpr,
    BLOCK: tl.constexpr, N_LANES: tl.constexpr, N_STEPS: tl.constexpr,
):
    """Map one storage-stream position to its dense logical offset."""
    storage_offset = block * BLOCK + step * N_LANES + lane
    storage_valid = storage_offset < n_elements
    n_tile = block // K_TILE_BLOCKS
    k_tile = block % K_TILE_BLOCKS
    logical_n = n_tile * N_STEPS + step
    logical_k = k_tile * N_LANES + lane
    logical_offset = logical_n * MATRIX_K + logical_k
    logical_valid = (
        (logical_n < MATRIX_N) & (logical_k < MATRIX_K)
        & (logical_offset < MATRIX_NUMEL)
    )
    return storage_offset, logical_offset, storage_valid, logical_valid


@triton.jit
def _pointwise_compressed_dense_impl(
    encoded, sign_mantissa, other, output, auxiliary, decode_table,
    n_elements, n_streams, center,
    OP: tl.constexpr, OUTPUT_POLICY: tl.constexpr,
    MATRIX_N: tl.constexpr, MATRIX_K: tl.constexpr,
    MATRIX_NUMEL: tl.constexpr,
    K_TILE_BLOCKS: tl.constexpr,
    FIRST_MASK: tl.constexpr, RARE_LENGTH: tl.constexpr,
    BLOCK: tl.constexpr, N_LANES: tl.constexpr,
    N_STEPS: tl.constexpr, FIXED_WORDS: tl.constexpr,
):
    """ Main fused pointwise kernel for compressed-tensor operations.
            Compressed tensor + dense tensor → dense result or compression components.

    Args:
        encoded: Word-major fixed Huffman payload for the compressed operand.
        sign_mantissa: Raw side-byte stream paired with ``encoded``.
        other: Dense BF16 operand with one value per logical input element.
        output: Dense BF16 result, or sign/mantissa bytes for compressed output.
        auxiliary: Raw result exponent bytes for compressed output.
        decode_table: Distribution-specific lookup table; center: exponent shift.
        n_elements: Logical element count; n_streams: fixed-stream count.
        OP: Inlined binary operation; OUTPUT_POLICY: dense or component output.
    """
    # One program handles one codec block; each lane decodes one fixed stream.
    block = tl.program_id(0)
    lanes = tl.arange(0, N_LANES)
    lane_index = block * N_LANES + lanes
    # word/shift track the current position inside the 64-bit fixed-payload window.
    word = tl.zeros((N_LANES,), tl.int32)
    shift = tl.zeros((N_LANES,), tl.int32)
    word0 = tl.load(encoded + word * n_streams + lane_index)
    word1 = tl.load(encoded + (word + 1) * n_streams + lane_index)
    window = word0.to(tl.uint32).to(tl.uint64)
    window |= word1.to(tl.uint32).to(tl.uint64) << 32
    center_value = tl.load(center).to(tl.int32)

    # Fast 1D path: storage offsets are also logical offsets for fully-contained blocks.
    if K_TILE_BLOCKS == 1 and MATRIX_K == N_LANES and (block + 1) * BLOCK <= MATRIX_NUMEL:
        storage_offset = block * BLOCK + lanes
        true_mask = tl.full((N_LANES,), True, tl.int1)
        for step in tl.range(0, N_STEPS, 2, flatten=True, warp_specialize=True):
            # Decode symbol 0, reconstruct its BF16 value, then apply the op.
            current = window >> shift
            value, length = decode_symbol(
                current, decode_table, center_value,
                FIRST_MASK, RARE_LENGTH,
            )
            sm = tl.load(sign_mantissa + storage_offset, cache_modifier='.cg')
            left = pack_bf16(value, sm).to(tl.int16).to(
                tl.bfloat16, bitcast=True
            )
            right = tl.load(other + storage_offset, cache_modifier='.cg')
            _store_result(
                OP(left, right), output, auxiliary, storage_offset,
                storage_offset, true_mask, true_mask, OUTPUT_POLICY,
            )

            # Decode symbol 1 at the next bit offset and process it the same way.
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
            right1 = tl.load(other + storage_offset + N_LANES, cache_modifier='.cg')
            _store_result(
                OP(left1, right1), output, auxiliary,
                storage_offset + N_LANES, storage_offset + N_LANES,
                true_mask, true_mask, OUTPUT_POLICY,
            )

            # Advance the 64-bit window and storage offset after processing two symbols.
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
        # Matrix/general path: convert codec blocks to logical (n, k) coordinates.
        if K_TILE_BLOCKS > 1:
            n_tile = block // K_TILE_BLOCKS
            k_tile = block % K_TILE_BLOCKS
            logical_k = k_tile * N_LANES + lanes
            storage_offset = block * BLOCK + lanes
            logical_n_base = n_tile * N_STEPS
            # Fully-valid matrix blocks skip the runtime masks in the hot loop.
            full_generic = (
                ((n_tile + 1) * N_STEPS <= MATRIX_N)
                & ((k_tile + 1) * N_LANES <= MATRIX_K)
            )
            if full_generic:
                for step in tl.range(0, N_STEPS, 2, loop_unroll_factor=2):
                    # Prefetch the next 32-bit Huffman word while decoding the pair.
                    word2_prefetch = tl.load(
                        encoded + tl.minimum(word + 2, FIXED_WORDS - 1) * n_streams
                        + lane_index,
                    ).to(tl.uint32).to(tl.uint64)
                    logical_n = logical_n_base + step
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
                        OP(left, right), output, auxiliary, storage_offset,
                        logical_offset, True, True, OUTPUT_POLICY,
                    )

                    shift1 = shift + length
                    current1 = window >> shift1
                    value1, length1 = decode_symbol(
                        current1, decode_table, center_value,
                        FIRST_MASK, RARE_LENGTH,
                    )
                    logical_offset1 = logical_offset + MATRIX_K
                    sm1 = tl.load(sign_mantissa + storage_offset + N_LANES, cache_modifier='.cg')
                    left1 = pack_bf16(value1, sm1).to(tl.int16).to(
                        tl.bfloat16, bitcast=True
                    )
                    right1 = tl.load(other + logical_offset1, cache_modifier='.cg')
                    _store_result(
                        OP(left1, right1), output, auxiliary,
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
                # Masked general path for partial/irregular matrix blocks.
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
                        OP(left, right), output, auxiliary, offset, logical_offset,
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
                        OP(left1, right1), output, auxiliary, offset1, logical_offset1,
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
                    OP(left, right), output, auxiliary, offset, logical_offset,
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
                    OP(left1, right1), output, auxiliary, offset1, logical_offset1,
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
def pointwise_compressed_dense_matrix_kernel(
    encoded, sign_mantissa, other, output, auxiliary, decode_table,
    n_elements, n_streams, center,
    OP: tl.constexpr, OUTPUT_POLICY: tl.constexpr,
    MATRIX_N: tl.constexpr, MATRIX_K: tl.constexpr,
    MATRIX_NUMEL: tl.constexpr, K_TILE_BLOCKS: tl.constexpr,
    FIRST_MASK: tl.constexpr, RARE_LENGTH: tl.constexpr,
    BLOCK: tl.constexpr, N_LANES: tl.constexpr,
    N_STEPS: tl.constexpr, FIXED_WORDS: tl.constexpr,
):
    """Apply a pointwise policy to native matrix-tiled storage."""
    _pointwise_compressed_dense_impl(
        encoded, sign_mantissa, other, output, auxiliary, decode_table,
        n_elements, n_streams, center, OP, OUTPUT_POLICY,
        MATRIX_N, MATRIX_K, MATRIX_NUMEL, K_TILE_BLOCKS,
        FIRST_MASK, RARE_LENGTH, BLOCK, N_LANES, N_STEPS, FIXED_WORDS,
    )


@triton.jit
def _pointwise_compressed_dense_fallback_impl(
    bad_streams, bad_starts, fallback_offsets,
    fallback_buffer, fallback_base, metadata, descriptor, fallback_count,
    sign_mantissa, other, output, auxiliary, n_elements,
    OP: tl.constexpr, OUTPUT_POLICY: tl.constexpr, BUFFERED: tl.constexpr,
    MATRIX_N: tl.constexpr, MATRIX_K: tl.constexpr,
    MATRIX_NUMEL: tl.constexpr,
    K_TILE_BLOCKS: tl.constexpr,
    TILE: tl.constexpr, BLOCK: tl.constexpr,
    N_LANES: tl.constexpr, N_STEPS: tl.constexpr,
):
    """Recompute pointwise results for stream tails stored in fallback storage."""
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
            OP(left, right), output, auxiliary, offset, logical_offset,
            logical_active, active, OUTPUT_POLICY,
        )


@triton.autotune(
    configs=SCATTER_FALLBACK_AUTOTUNE_CONFIGS,
    key=["n_elements", "N_LANES", "N_STEPS", "BLOCK"],
)
@triton.jit
def pointwise_compressed_dense_matrix_fallback_kernel(
    bad_streams, bad_starts, fallback_offsets,
    fallback_buffer, fallback_base, metadata, descriptor, fallback_count,
    sign_mantissa, other, output, auxiliary, n_elements,
    OP: tl.constexpr, OUTPUT_POLICY: tl.constexpr, BUFFERED: tl.constexpr,
    MATRIX_N: tl.constexpr, MATRIX_K: tl.constexpr,
    MATRIX_NUMEL: tl.constexpr, K_TILE_BLOCKS: tl.constexpr,
    TILE: tl.constexpr, BLOCK: tl.constexpr,
    N_LANES: tl.constexpr, N_STEPS: tl.constexpr,
):
    """Correct matrix-storage pointwise results from fallback tails."""
    _pointwise_compressed_dense_fallback_impl(
        bad_streams, bad_starts, fallback_offsets, fallback_buffer,
        fallback_base, metadata, descriptor, fallback_count,
        sign_mantissa, other, output, auxiliary, n_elements,
        OP, OUTPUT_POLICY, BUFFERED,
        MATRIX_N, MATRIX_K, MATRIX_NUMEL, K_TILE_BLOCKS,
        TILE, BLOCK, N_LANES, N_STEPS,
    )
