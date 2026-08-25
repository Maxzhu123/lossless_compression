"""Triton kernel for dense activations and matrix-tiled Huffman weights."""

import triton
from triton import language as tl

from ..codec.primitives import decode_symbol, pack_bf16


MATRIX_TILE = 256


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=8, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=3, maxnreg=64),
    ],
    key=["N", "K", "MATRIX_STEPS", "FIXED_WORDS"],
)
@triton.jit
def decode_matrix_rhs_kernel(
    encoded, sign_mantissa, decode_table,
    fallback_buffer, fallback_descriptor, fallback_count,
    stream_starts, stream_offsets, output, center,
    N: tl.constexpr, K: tl.constexpr, K_TILE_BLOCKS: tl.constexpr,
    MATRIX_STEPS: tl.constexpr, N_STREAMS: tl.constexpr,
    FIXED_WORDS: tl.constexpr, FALLBACK_BASE: tl.constexpr,
    BUFFERED: tl.constexpr, FIRST_MASK: tl.constexpr,
    RARE_LENGTH: tl.constexpr,
):
    """Decode one storage block directly into contiguous ``[K, N]`` GEMM layout."""
    storage_block = tl.program_id(0)
    lane = tl.arange(0, 256)
    stream = storage_block * 256 + lane
    n_tile = storage_block // K_TILE_BLOCKS
    k_tile = storage_block % K_TILE_BLOCKS
    logical_n = n_tile * 256 + lane
    valid_n = logical_n < N
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
        logical_k = k_tile * MATRIX_STEPS + step
        tl.store(
            output + logical_k * N + logical_n, bits,
            mask=valid_n & (logical_k < K),
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
            output + (logical_k + 1) * N + logical_n, bits1,
            mask=valid_n & (logical_k + 1 < K),
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


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 16, "BLOCK_K": 16, "SPLIT_K": 4}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 16, "BLOCK_K": 16, "SPLIT_K": 8}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 32, "BLOCK_K": 16, "SPLIT_K": 8}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 64, "BLOCK_K": 16, "SPLIT_K": 8}, num_warps=4, num_stages=2),
    ],
    key=["M", "N", "K"],
    reset_to_zero=["output"],
)
@triton.jit
def compressed_linear_kernel(
    activations, encoded, sign_mantissa, decode_table,
    fallback_buffer, fallback_descriptor, fallback_count,
    stream_starts, stream_offsets, output, center,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    K_TILE_BLOCKS: tl.constexpr,
    MATRIX_STEPS: tl.constexpr,
    N_STREAMS: tl.constexpr, FIXED_WORDS: tl.constexpr,
    SPLIT_K: tl.constexpr,
    FALLBACK_BASE: tl.constexpr, BUFFERED: tl.constexpr,
    FIRST_MASK: tl.constexpr, RARE_LENGTH: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """Decode tiled weights into register fragments and accumulate tensor-core dots."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    lane = offs_n % 256
    n_tile = offs_n // 256
    center_value = tl.load(center).to(tl.int32)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

    if pid_k >= K_TILE_BLOCKS:
        return

    if BUFFERED:
        allocation_base = tl.load(fallback_descriptor).to(tl.int32)
        bad_count = tl.load(fallback_count).to(tl.int32)
        fallback_base = allocation_base + 9 * bad_count
    else:
        fallback_base = FALLBACK_BASE

    for k_tile in tl.range(pid_k, K_TILE_BLOCKS, SPLIT_K):
        storage_block = n_tile * K_TILE_BLOCKS + k_tile
        stream = storage_block * 256 + lane
        valid_n = offs_n < N
        word = tl.zeros((BLOCK_N,), tl.int32)
        shift = tl.zeros((BLOCK_N,), tl.int32)
        word0 = tl.load(
            encoded + stream, mask=valid_n, other=0,
        ).to(tl.uint32).to(tl.uint64)
        word1 = tl.load(
            encoded + N_STREAMS + stream, mask=valid_n, other=0,
        ).to(tl.uint32).to(tl.uint64)
        window = word0 | (word1 << 32)
        start = tl.load(stream_starts + stream, mask=valid_n, other=255).to(tl.int32)
        extra_offset = tl.load(
            stream_offsets + stream, mask=valid_n, other=-1,
        ).to(tl.int32)

        for k_chunk in tl.range(0, MATRIX_STEPS, BLOCK_K):
            weights = tl.zeros((BLOCK_K, BLOCK_N), tl.bfloat16)
            rows = tl.arange(0, BLOCK_K)[:, None]
            for step in tl.static_range(0, BLOCK_K, 2):
                matrix_step = k_chunk + step
                value, length = decode_symbol(
                    window >> shift, decode_table, center_value,
                    FIRST_MASK, RARE_LENGTH,
                )
                use_fallback = valid_n & (start != 255) & (matrix_step >= start)
                fallback_value = tl.load(
                    fallback_buffer + fallback_base + extra_offset + matrix_step - start,
                    mask=use_fallback, other=0,
                ).to(tl.int32)
                value = tl.where(use_fallback, fallback_value, value)
                storage_offset = storage_block * MATRIX_STEPS * 256 + matrix_step * 256 + lane
                sm = tl.load(
                    sign_mantissa + storage_offset, mask=valid_n, other=0,
                )
                bits = pack_bf16(value, sm).to(tl.int16)
                weight = bits.to(tl.bfloat16, bitcast=True)
                weights = tl.where(rows == step, weight[None, :], weights)

                shift1 = shift + length
                value1, length1 = decode_symbol(
                    window >> shift1, decode_table, center_value,
                    FIRST_MASK, RARE_LENGTH,
                )
                use_fallback1 = valid_n & (start != 255) & (matrix_step + 1 >= start)
                fallback_value1 = tl.load(
                    fallback_buffer + fallback_base + extra_offset + matrix_step + 1 - start,
                    mask=use_fallback1, other=0,
                ).to(tl.int32)
                value1 = tl.where(use_fallback1, fallback_value1, value1)
                sm1 = tl.load(
                    sign_mantissa + storage_offset + 256, mask=valid_n, other=0,
                )
                bits1 = pack_bf16(value1, sm1).to(tl.int16)
                weight1 = bits1.to(tl.bfloat16, bitcast=True)
                weights = tl.where(rows == step + 1, weight1[None, :], weights)

                next_shift = shift1 + length1
                crosses_word = next_shift >= 32
                word2 = tl.load(
                    encoded + tl.minimum(word + 2, FIXED_WORDS - 1) * N_STREAMS + stream,
                    mask=valid_n & crosses_word, other=0,
                ).to(tl.uint32).to(tl.uint64)
                next_window = (window >> 32) | (word2 << 32)
                window = tl.where(crosses_word, next_window, window)
                word += crosses_word
                shift = tl.where(crosses_word, next_shift - 32, next_shift)

            offs_k = k_tile * MATRIX_STEPS + k_chunk + tl.arange(0, BLOCK_K)
            activations_tile = tl.load(
                activations + offs_m[:, None] * K + offs_k[None, :],
                mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
                other=0.0,
            )
            accumulator += tl.dot(activations_tile, weights)

    tl.atomic_add(
        output + offs_m[:, None] * N + offs_n[None, :], accumulator,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )
