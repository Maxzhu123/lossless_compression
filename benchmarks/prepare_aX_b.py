"""Temporary benchmark for the fused scalar multiply-add pointwise op only."""
import torch

from compress.code_storage import Distribution, DistType
from compress.compress import compress, decompress, compressed_scalar_mul_add
from compress.tensor_buffer import TensorBuffer
from compress.ops.registry import SCALAR_MUL_ADD

SIZES = [512, 1024, 2048, 4096]

def _buffer(size):
    return TensorBuffer(size + 64 * 1024 * 1024, device="cuda")

def _free(encoded, buffer):
    if encoded.fallback_descriptor is not None:
        buffer.free(encoded.fallback_descriptor)

def _time(fn, iters=100):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iters

def main():
    total_dense = 0.0
    total_compressed = 0.0
    for n in SIZES:
        shape = (n, 4 * n)
        x = torch.randn(shape, device="cuda").to(torch.bfloat16)
        other = torch.randn_like(x)
        buf = _buffer(x.numel())
        outbuf = _buffer(x.numel())
        encoded = compress(x, Distribution(DistType.GAUSSIAN), buf)
        restored = decompress(encoded)
        alpha = 0.7
        expected = (restored.float() * alpha + other.float()).to(torch.bfloat16)
        dense = lambda: compressed_scalar_mul_add(encoded, alpha, other, dense_output=True)
        comp = lambda: compressed_scalar_mul_add(
            encoded, alpha, other, dense_output=False, buffer=outbuf,
        )
        dense_ms = _time(dense)
        comp_ms = _time(comp)
        total_dense += dense_ms
        total_compressed += comp_ms
        print(f"shape={shape!s:>14s}  dense={dense_ms:7.4f} ms  compressed={comp_ms:7.4f} ms")
        encoded.free()
        outbuf.reset()
        buf.reset()
    print(f"Total dense: {total_dense:.4f} ms  Total compressed: {total_compressed:.4f} ms")

if __name__ == "__main__":
    main()
