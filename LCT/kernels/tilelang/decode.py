"""Huffman decoding with a shared codebook and coalesced BF16 output tiles."""

from functools import lru_cache
import hashlib
import inspect
from threading import RLock

import torch
import tilelang
import tilelang.language as T
from tilelang.autotuner import AutoTuner

from ...compression.huffman_tables import get_distribution_tables
from ..generic.decode import _restore_fallback
from .autotune import DECODE_AUTOTUNE_CONFIGS, DEFAULT_DECODE_CONFIG

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


def _decode_program(block_symbols, lanes, steps, fixed_words, rare_length,
                    index_dtype="int32", threads=256, row_tile=8, unroll=4):
    """Build one configuration with runtime sizes and explicit stream ownership."""
    if lanes != 256 or threads not in (64, 128, 256) or fixed_words < 2:
        raise ValueError("TileLang decode requires 256 lanes and 64, 128, or 256 threads")
    if row_tile < 4 or row_tile % 2 or steps % row_tile or unroll < 1:
        raise ValueError("Decoder row tiles must divide the stream length and contain whole pairs")
    streams_per_thread = lanes // threads
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
        with T.Kernel(blocks, threads=threads) as block:
            T.import_source(_DEVICE_SOURCE)
            thread = T.get_thread_binding()
            lookup = T.alloc_shared((1024,), "int32")
            decoded = T.alloc_shared((row_tile, 256), "int16")
            word = T.alloc_local((streams_per_thread,), "int32")
            shift = T.alloc_local((streams_per_thread,), "int32")
            low = T.alloc_local((streams_per_thread,), "int32")
            high = T.alloc_local((streams_per_thread,), "int32")
            next_word = T.alloc_local((streams_per_thread,), "int32")
            prefix = T.alloc_local((streams_per_thread,), "uint32")
            sm = T.alloc_local((streams_per_thread,), "int32")
            center_value = center[0]
            zero_delta = (-127 - center_value) & 255

            # Translate the shared raw-symbol table once per block. Store biased
            # exponent bytes to avoid signed remapping in the common decode path.
            for i in T.unroll(1024 // threads):
                index = i * threads + thread
                packed = base_table[index]
                symbol = (packed >> 8) & 255
                raw = ((symbol - T.Cast("int32", symbol <= zero_delta) + center_value + 127) & 255)
                raw_exponent = raw & (-T.Cast("int32", symbol != 0))
                lookup[index] = (packed & 255) | (raw_exponent << 8)
            T.sync_threads()
            for owned in T.unroll(streams_per_thread):
                stream = block * lanes + thread + owned * threads
                word[owned] = 0
                shift[owned] = 0
                low[owned] = encoded[stream]
                high[owned] = encoded[streams + stream]
                next_word[owned] = encoded[T.min(2, fixed_words - 1) * streams + stream]

            # Specialize complete blocks separately from the final masked block.
            for path in T.unroll(2):
                if (((block + 1) * block_symbols <= logical_numel) == (path == 0)):
                    for tile in T.serial(steps // row_tile):
                        for pair in T.unroll(row_tile // 2, unroll_factor=unroll if unroll < row_tile // 2 else None):
                            for member in T.unroll(2):
                                for owned in T.unroll(streams_per_thread):
                                    lane = thread + owned * threads
                                    if member == 0:
                                        prefix[owned] = T.call_extern("uint32", "__funnelshift_r", low[owned], high[owned], shift[owned])
                                    else:
                                        prefix[owned] = T.call_extern("uint32", "lct_decode_prefix", low[owned], high[owned], shift[owned])
                                    bits = prefix[owned]
                                    first = lookup[T.Cast("int32", bits & 1023)]
                                    length = first & 255
                                    escaped_symbol = T.Cast("int32", (bits >> rare_length) & 255)
                                    escaped_raw = ((escaped_symbol - T.Cast("int32", escaped_symbol <= zero_delta) + center_value + 127) & 255)
                                    escaped_byte = escaped_raw & (-T.Cast("int32", escaped_symbol != 0))
                                    raw_exponent = T.if_then_else(length == 0, escaped_byte, first >> 8)
                                    consumed = T.if_then_else(length == 0, rare_length + 8, length)
                                    step = tile * row_tile + pair * 2 + member
                                    storage = block * block_symbols + step * lanes + lane
                                    if path == 0:
                                        sm[owned] = T.Cast("int32", T.call_extern("uint8", "__ldcg", T.address_of(side[storage])))
                                    else:
                                        sm[owned] = T.if_then_else(storage < storage_numel, T.Cast("int32", side[storage]), 0)
                                    col = (lane + block * steps + step) & 255
                                    decoded[pair * 2 + member, col] = T.Cast("int16", (raw_exponent << 7) | (sm[owned] & 127) | ((sm[owned] & 128) << 8))
                                    shift[owned] = shift[owned] + consumed
                            for owned in T.unroll(streams_per_thread):
                                stream = block * lanes + thread + owned * threads
                                if shift[owned] >= 32:
                                    low[owned] = high[owned]
                                    high[owned] = next_word[owned]
                                    word[owned] = word[owned] + 1
                                    shift[owned] = shift[owned] - 32
                                    # Retain the prefetched word until it is consumed;
                                    # its replacement is needed at the next crossing.
                                    next_word[owned] = encoded[T.min(word[owned] + 2, fixed_words - 1) * streams + stream]
                        T.sync_threads()
                        # Eight-byte aligned stores remove the global row swizzle.
                        # The rotated writes above place results in logical order.
                        for work in T.serial(row_tile * lanes // (4 * threads)):
                            flat = work * threads + thread
                            row = flat // 64
                            col_start = (flat % 64) * 4
                            out = (block * steps + tile * row_tile + row) * lanes + col_start
                            if path == 0 or out + 3 < logical_numel:
                                T.evaluate(T.call_extern("handle", "lct_decode_store4", T.address_of(output[out]), T.address_of(decoded[row, col_start])))
                            else:
                                for item in T.unroll(4):
                                    if out + item < logical_numel:
                                        output[out + item] = decoded[row, col_start + item]
                        T.sync_threads()

    return kernel


@lru_cache(maxsize=128)
def _decode_kernel(block_symbols, lanes, steps, fixed_words, rare_length, target,
                   index_dtype="int32", threads=256, row_tile=8, unroll=4):
    """Compile per geometry/configuration; tensor sizes remain runtime metadata."""
    program = _decode_program(block_symbols, lanes, steps, fixed_words, rare_length,
                              index_dtype, threads, row_tile, unroll)
    return tilelang.compile(program, target=target, pass_configs=_PASS_CONFIGS)


_PASS_CONFIGS = {"tl.disable_safe_memory_legalize": True, "tl.disable_warp_specialized": True}
_TUNED_KERNELS = {}
_TUNING_LOCK = RLock()


@lru_cache(maxsize=1)
def _program_digest():
    # The tuning wrapper's source alone would not invalidate TileLang's disk
    # cache when the underlying decoder or imported CUDA intrinsics change.
    return hashlib.sha256((inspect.getsource(_decode_program) + _DEVICE_SOURCE).encode()).hexdigest()


@lru_cache(maxsize=16)
def _device_info(device):
    capability = torch.cuda.get_device_capability(device)
    return torch.cuda.get_device_name(device), f"cuda -arch=sm_{capability[0]}{capability[1]}"


def _tuning_key(data, rare_length, target, index_dtype):
    n = data.logical_numel
    # Bucket tiny inputs together; distinguish fully contained blocks from tails.
    size_bucket = 1 << (max(n, 65536) - 1).bit_length()
    dist = data.distribution
    return (_device_info(data.data.device)[0], data.data.device.index,
            target, index_dtype, data.codec_geometry, rare_length, size_bucket,
            n % data.codec_geometry[0] == 0,
            dist.family.value, dist.noise_level.name, dist.param, dist.mean, dist.zero_prob)


def _select_kernel(data, output, base_table, rare_length, target, index_dtype):
    key = _tuning_key(data, rare_length, target, index_dtype)
    cached = _TUNED_KERNELS.get(key)
    if cached is not None:
        return cached
    # Tuning benchmarks and validates on the GPU and must stay outside capture.
    # An uncached capture uses the validated default; a later normal call tunes.
    if torch.cuda.is_current_stream_capturing():
        return _decode_kernel(*data.codec_geometry, rare_length, target, index_dtype, **DEFAULT_DECODE_CONFIG)
    with _TUNING_LOCK:
        cached = _TUNED_KERNELS.get(key)
        if cached is not None:
            return cached
        block_symbols, lanes, steps, fixed_words = data.codec_geometry

        def program(threads=256, row_tile=8, unroll=4):
            return _decode_program(block_symbols, lanes, steps, fixed_words, rare_length,
                                   index_dtype, threads, row_tile, unroll)

        inputs = [data.data, data.sign_mantissa, base_table, data.center, output]
        expected = None
        reference_kernel = None

        def reference(*tensors):
            nonlocal expected, reference_kernel
            if expected is None:
                expected = torch.empty_like(output)
                reference_kernel = _decode_kernel(*data.codec_geometry, rare_length, target, index_dtype)
            reference_kernel(*tensors[:-1], expected)
            # Poison the external output: missing stores must fail validation.
            tensors[-1].fill_(-12345)

        def assert_bits(_actual, _expected):
            if not torch.equal(output, expected):
                raise AssertionError("Autotuned decoder differs from the validated default")

        tuner = AutoTuner(program, configs=list(DECODE_AUTOTUNE_CONFIGS))
        tuner.set_compile_args(target=target, execution_backend="tvm_ffi", pass_configs=_PASS_CONFIGS)
        tuner.set_profile_args(supply_prog=lambda _: inputs, ref_prog=reference,
                               manual_check_prog=assert_bits, cache_input_tensors=False,
                               backend="cudagraph", rtol=0, atol=0, max_mismatched_ratio=0)
        tuner.set_kernel_parameters(
            ((), (("decoder_key", key), ("decoder_source", _program_digest()))),
            inspect.signature(program).parameters,
        )
        # TileLang 0.1.8's manual checker modifies Torch print options. Preserve
        # application formatting while still using exact bitwise validation.
        with torch._tensor_str.printoptions():
            result = tuner.run(warmup=3, rep=15, timeout=30)
        _TUNED_KERNELS[key] = result.kernel
        return result.kernel


def get_decode_config(data):
    """Return the cached selected configuration, or None before first tuning."""
    _, _, rare_length = get_distribution_tables(data.distribution)
    _, target = _device_info(data.data.device)
    index_dtype = _index_dtype(data)
    kernel = _TUNED_KERNELS.get(_tuning_key(data, rare_length, target, index_dtype))
    return None if kernel is None else dict(kernel.get_tuner_result()["config"])


def _index_dtype(data):
    return "int32" if max(data.logical_numel, data.size, data.data.numel()) < 2**31 - 65536 else "int64"


def decode(data):
    """Decode with a cached, bitwise-validated GPU/size-specific configuration.

    First use outside CUDA graph capture tunes candidates and synchronizes for
    validation/timing. Cached calls keep centers and overflow counts on device.
    """
    logical_numel = data.logical_numel
    output = torch.empty(logical_numel, dtype=torch.int16, device=data.data.device)
    if logical_numel:
        _, base_table, rare_length = get_distribution_tables(data.distribution)
        _, target = _device_info(data.data.device)
        # Tensor shapes are inspected on the CPU; the sampled center and all
        # overflow counts stay on the GPU. Use wide sizes near the int32 limit.
        index_dtype = _index_dtype(data)
        with torch.cuda.device(data.data.device):
            kernel = _select_kernel(data, output, base_table, rare_length, target, index_dtype)
            kernel(data.data, data.sign_mantissa, base_table, data.center, output)
        _restore_fallback(data, output)
    return output.view(torch.bfloat16).reshape(data.shape)
