# Gluon experiment for the normal decode path.
import triton.experimental.gluon as gluon
import triton.experimental.gluon.language as ttgl
import triton.language as tl

@gluon.jit
def _gluon_decode_experimental(
    encoded, sign_mantissa, output,
    decode_table, n_elements, n_streams, center,
    MATRIX_N: ttgl.constexpr, MATRIX_K: ttgl.constexpr,
    MATRIX_NUMEL: ttgl.constexpr, K_TILE_BLOCKS: ttgl.constexpr,
    FIRST_MASK: ttgl.constexpr, RARE_LENGTH: ttgl.constexpr,
    BLOCK: ttgl.constexpr, N_LANES: ttgl.constexpr, N_STEPS: ttgl.constexpr, FIXED_WORDS: ttgl.constexpr, ON_DEMAND: ttgl.constexpr,
):
    block = ttgl.program_id(0)
    n_warps: ttgl.constexpr = ttgl.num_warps()
    lane_layout: ttgl.constexpr = ttgl.BlockedLayout(
        size_per_thread=[N_LANES // (32 * n_warps)],
        threads_per_warp=[32],
        warps_per_cta=[n_warps],
        order=[0],
    )
    lanes = ttgl.arange(0, N_LANES, layout=lane_layout)
    lane_index = block * N_LANES + lanes
    word = lanes - lanes
    shift = lanes - lanes
    word0 = ttgl.load(encoded + word * n_streams + lane_index).to(ttgl.uint32).to(ttgl.uint64)
    word1 = ttgl.load(encoded + (word + 1) * n_streams + lane_index).to(ttgl.uint32).to(ttgl.uint64)
    window = word0 | word1 << 32
    center_value = ttgl.load(center).to(ttgl.int32)
    if K_TILE_BLOCKS == 1 and MATRIX_K == N_LANES and (block + 1) * BLOCK <= MATRIX_NUMEL:
        # Fast 1-D full-block path: no boundary masks are required.
        storage_offset = block * BLOCK + lanes
        for step in tl.range(0, N_STEPS, 2, flatten=True, warp_specialize=True):
            if not ON_DEMAND:
                word2_prefetch = ttgl.load(encoded + ttgl.minimum(word + 2, FIXED_WORDS - 1) * n_streams + lane_index).to(ttgl.uint32).to(ttgl.uint64)
            current = window >> shift
            first = ttgl.load(decode_table + (current & FIRST_MASK).to(ttgl.int32), cache_modifier='.ca')
            first_length = first & 255
            continuation = first_length == 0
            length = ttgl.where(continuation, RARE_LENGTH + 8, first_length)
            tail = (current >> RARE_LENGTH & 255).to(ttgl.int32)
            tail = ttgl.where(tail >= 128, tail - 256, tail)
            value = ttgl.where(continuation, tail, first >> 8) + center_value
            sm = ttgl.load(sign_mantissa + storage_offset, cache_modifier='.cg')
            packed = (value.to(ttgl.int32) + 127 & 255) << 7 | (sm.to(ttgl.int32) & 127) | (sm.to(ttgl.int32) & 128) << 8
            packed = packed.to(ttgl.int16)
            ttgl.store(output + storage_offset, packed, cache_modifier='.cs')
            shift1 = shift + length
            current1 = window >> shift1
            first1 = ttgl.load(decode_table + (current1 & FIRST_MASK).to(ttgl.int32), cache_modifier='.ca')
            first_length1 = first1 & 255
            continuation1 = first_length1 == 0
            length1 = ttgl.where(continuation1, RARE_LENGTH + 8, first_length1)
            tail1 = (current1 >> RARE_LENGTH & 255).to(ttgl.int32)
            tail1 = ttgl.where(tail1 >= 128, tail1 - 256, tail1)
            value1 = ttgl.where(continuation1, tail1, first1 >> 8) + center_value
            sm1 = ttgl.load(sign_mantissa + storage_offset + N_LANES, cache_modifier='.cg')
            packed1 = (value1.to(ttgl.int32) + 127 & 255) << 7 | (sm1.to(ttgl.int32) & 127) | (sm1.to(ttgl.int32) & 128) << 8
            packed1 = packed1.to(ttgl.int16)
            ttgl.store(output + storage_offset + N_LANES, packed1, cache_modifier='.cs')
            next_shift = shift1 + length1
            crosses_word = next_shift >= 32
            if ON_DEMAND:
                next_word = ttgl.load(encoded + ttgl.minimum(word + 2, FIXED_WORDS - 1) * n_streams + lane_index, mask=crosses_word, other=0).to(ttgl.uint32).to(ttgl.uint64)
                next_window = window >> 32 | next_word << 32
            else:
                next_window = window >> 32 | word2_prefetch << 32
            window = ttgl.where(crosses_word, next_window, window)
            word += crosses_word
            shift = ttgl.where(crosses_word, next_shift - 32, next_shift)
            storage_offset += 2 * N_LANES
    else:
        if K_TILE_BLOCKS == 1 and MATRIX_K == N_LANES:
            storage_offset = block * BLOCK + lanes
            output_offset = storage_offset
            logical_n = block * N_STEPS
        else:
            n_tile = block // K_TILE_BLOCKS
            k_tile = block % K_TILE_BLOCKS
            logical_k = k_tile * N_LANES + lanes
            storage_offset = block * BLOCK + lanes
            output_offset = n_tile * N_STEPS * MATRIX_K + logical_k
            logical_n = n_tile * N_STEPS
        for step in range(0, N_STEPS, 2):
            word2_prefetch = ttgl.load(encoded + ttgl.minimum(word + 2, FIXED_WORDS - 1) * n_streams + lane_index).to(ttgl.uint32).to(ttgl.uint64)
            storage_valid = storage_offset < n_elements
            valid = (logical_n < MATRIX_N) & (output_offset < MATRIX_NUMEL)
            current = window >> shift
            first = ttgl.load(decode_table + (current & FIRST_MASK).to(ttgl.int32), cache_modifier='.ca')
            first_length = first & 255
            continuation = first_length == 0
            length = ttgl.where(continuation, RARE_LENGTH + 8, first_length)
            tail = (current >> RARE_LENGTH & 255).to(ttgl.int32)
            tail = ttgl.where(tail >= 128, tail - 256, tail)
            value = ttgl.where(continuation, tail, first >> 8) + center_value
            sm = ttgl.load(sign_mantissa + storage_offset, mask=storage_valid, other=0, cache_modifier='.cg')
            packed = (value.to(ttgl.int32) + 127 & 255) << 7 | (sm.to(ttgl.int32) & 127) | (sm.to(ttgl.int32) & 128) << 8
            packed = packed.to(ttgl.int16)
            ttgl.store(output + output_offset, packed, mask=valid, cache_modifier='.cs')
            shift1 = shift + ttgl.where(storage_valid, length, 0)
            current1 = window >> shift1
            first1 = ttgl.load(decode_table + (current1 & FIRST_MASK).to(ttgl.int32), cache_modifier='.ca')
            first_length1 = first1 & 255
            continuation1 = first_length1 == 0
            length1 = ttgl.where(continuation1, RARE_LENGTH + 8, first_length1)
            tail1 = (current1 >> RARE_LENGTH & 255).to(ttgl.int32)
            tail1 = ttgl.where(tail1 >= 128, tail1 - 256, tail1)
            value1 = ttgl.where(continuation1, tail1, first1 >> 8) + center_value
            storage_offset1 = storage_offset + N_LANES
            storage_valid1 = storage_offset1 < n_elements
            output_offset1 = output_offset + MATRIX_K
            valid1 = (logical_n + 1 < MATRIX_N) & (output_offset1 < MATRIX_NUMEL)
            sm1 = ttgl.load(sign_mantissa + storage_offset1, mask=storage_valid1, other=0, cache_modifier='.cg')
            packed1 = (value1.to(ttgl.int32) + 127 & 255) << 7 | (sm1.to(ttgl.int32) & 127) | (sm1.to(ttgl.int32) & 128) << 8
            packed1 = packed1.to(ttgl.int16)
            ttgl.store(output + output_offset1, packed1, mask=valid1, cache_modifier='.cs')
            next_shift = shift1 + ttgl.where(storage_valid1, length1, 0)
            crosses_word = next_shift >= 32
            next_window = window >> 32 | word2_prefetch << 32
            window = ttgl.where(crosses_word, next_window, window)
            word += crosses_word
            shift = ttgl.where(crosses_word, next_shift - 32, next_shift)
            storage_offset += 2 * N_LANES
            output_offset += 2 * MATRIX_K
            logical_n += 2
