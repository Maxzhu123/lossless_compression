"""Huffman decoding with a shared codebook and coalesced BF16 output tiles."""

from functools import lru_cache

import torch
import tilelang
import tilelang.language as T

from ...compression.huffman_tables import get_distribution_tables
from ..generic.decode import _restore_fallback

# Integer intrinsics preserve every BF16 representation, including NaN payloads.
_DEVICE_SOURCE = r"""
__device__ __forceinline__ unsigned lct_decode_prefix(unsigned lo, unsigned hi, int shift) {
    return shift < 32 ? __funnelshift_r(lo, hi, shift) : hi >> (shift - 32);
}
__device__ __forceinline__ void lct_decode_store4(short* output, const short* source) {
    uint2 value = *reinterpret_cast<const uint2*>(source);
    __stcs(reinterpret_cast<uint2*>(output), value);
}
"""


@lru_cache(maxsize=64)
def _decode_kernel(block_symbols, lanes, steps, fixed_words, rare_length, target, index_dtype="int32"):
    """Compile per geometry/architecture; logical sizes are runtime metadata."""
    if lanes != 256 or steps % 8 or fixed_words < 2:
        raise ValueError("TileLang decode requires 256 lanes, eight-row tiles, and at least two words")
    logical_numel = T.dynamic("logical_numel", dtype=index_dtype)
    storage_numel = T.dynamic("storage_numel", dtype=index_dtype)
    payload_numel = T.dynamic("payload_numel", dtype=index_dtype)
    blocks = T.ceildiv(storage_numel, block_symbols)
    streams = blocks * lanes

    @T.prim_func
    def kernel(
        encoded: T.Tensor((payload_numel,), "int32"),
        side: T.Tensor((storage_numel,), "uint8"),
        base_table: T.Tensor((1024,), "int32"),
        center: T.Tensor((1,), "int32"),
        output: T.Tensor((logical_numel,), "int16"),
    ):
        with T.Kernel(blocks, threads=256) as block:
            T.import_source(_DEVICE_SOURCE)
            lane = T.get_thread_binding()
            lookup = T.alloc_shared((1024,), "int32")
            decoded = T.alloc_shared((8, 256), "int16")
            word = T.alloc_local((1,), "int32")
            shift = T.alloc_local((1,), "int32")
            low = T.alloc_local((1,), "int32")
            high = T.alloc_local((1,), "int32")
            next_word = T.alloc_local((1,), "int32")
            prefix = T.alloc_local((1,), "uint32")
            sm = T.alloc_local((1,), "int32")
            center_value = center[0]
            zero_delta = (-127 - center_value) & 255
            stream = block * lanes + lane

            # Translate the shared raw-symbol table once per block. Store biased
            # exponent bytes to avoid signed remapping in the common decode path.
            for i in T.unroll(4):
                index = i * 256 + lane
                packed = base_table[index]
                symbol = (packed >> 8) & 255
                raw = ((symbol - T.Cast("int32", symbol <= zero_delta) + center_value + 127) & 255)
                raw_exponent = raw & (-T.Cast("int32", symbol != 0))
                lookup[index] = (packed & 255) | (raw_exponent << 8)
            T.sync_threads()
            word[0] = 0
            shift[0] = 0
            low[0] = encoded[stream]
            high[0] = encoded[streams + stream]
            next_word[0] = encoded[T.min(2, fixed_words - 1) * streams + stream]

            # Specialize complete blocks separately from the final masked block.
            for path in T.unroll(2):
                if (((block + 1) * block_symbols <= logical_numel) == (path == 0)):
                    for tile in T.serial(steps // 8):
                        for pair in T.unroll(4):
                            for member in T.unroll(2):
                                if member == 0:
                                    prefix[0] = T.call_extern("uint32", "__funnelshift_r", low[0], high[0], shift[0])
                                else:
                                    prefix[0] = T.call_extern("uint32", "lct_decode_prefix", low[0], high[0], shift[0])
                                bits = prefix[0]
                                first = lookup[T.Cast("int32", bits & 1023)]
                                length = first & 255
                                escaped_symbol = T.Cast("int32", (bits >> rare_length) & 255)
                                escaped_raw = ((escaped_symbol - T.Cast("int32", escaped_symbol <= zero_delta) + center_value + 127) & 255)
                                escaped_byte = escaped_raw & (-T.Cast("int32", escaped_symbol != 0))
                                raw_exponent = T.if_then_else(length == 0, escaped_byte, first >> 8)
                                consumed = T.if_then_else(length == 0, rare_length + 8, length)
                                step = tile * 8 + pair * 2 + member
                                storage = block * block_symbols + step * lanes + lane
                                if path == 0:
                                    sm[0] = T.Cast("int32", T.call_extern("uint8", "__ldcg", T.address_of(side[storage])))
                                else:
                                    sm[0] = T.if_then_else(storage < storage_numel, T.Cast("int32", side[storage]), 0)
                                col = (lane + block * steps + step) & 255
                                decoded[pair * 2 + member, col] = T.Cast("int16", (raw_exponent << 7) | (sm[0] & 127) | ((sm[0] & 128) << 8))
                                shift[0] = shift[0] + consumed
                            if shift[0] >= 32:
                                low[0] = high[0]
                                high[0] = next_word[0]
                                word[0] = word[0] + 1
                                shift[0] = shift[0] - 32
                                # Retain the prefetched word until it is consumed;
                                # its replacement is needed at the next crossing.
                                next_word[0] = encoded[T.min(word[0] + 2, fixed_words - 1) * streams + stream]
                        T.sync_threads()
                        # Eight-byte aligned stores remove the global row swizzle.
                        # The rotated writes above place results in logical order.
                        for work in T.serial(2):
                            flat = work * 256 + lane
                            row = flat // 64
                            col_start = (flat % 64) * 4
                            out = (block * steps + tile * 8 + row) * lanes + col_start
                            if path == 0 or out + 3 < logical_numel:
                                T.evaluate(T.call_extern("handle", "lct_decode_store4", T.address_of(output[out]), T.address_of(decoded[row, col_start])))
                            else:
                                for item in T.unroll(4):
                                    if out + item < logical_numel:
                                        output[out + item] = decoded[row, col_start + item]
                        T.sync_threads()

    return tilelang.compile(
        kernel,
        target=target,
        pass_configs={
            "tl.disable_safe_memory_legalize": True,
            "tl.disable_warp_specialized": True,
        },
    )


def decode(data):
    """Decode without changing storage geometry, format, or overflow handling."""
    logical_numel = data.logical_numel
    output = torch.empty(logical_numel, dtype=torch.int16, device=data.data.device)
    if logical_numel:
        _, base_table, rare_length = get_distribution_tables(data.distribution)
        capability = torch.cuda.get_device_capability(data.data.device)
        target = f"cuda -arch=sm_{capability[0]}{capability[1]}"
        # Tensor shapes are inspected on the CPU; the sampled center and all
        # overflow counts stay on the GPU. Use wide sizes near the int32 limit.
        index_dtype = "int32" if max(logical_numel, data.size, data.data.numel()) < 2**31 - 65536 else "int64"
        with torch.cuda.device(data.data.device):
            kernel = _decode_kernel(*data.codec_geometry, rare_length, target, index_dtype)
            kernel(data.data, data.sign_mantissa, base_table, data.center, output)
        _restore_fallback(data, output)
    return output.view(torch.bfloat16).reshape(data.shape)
