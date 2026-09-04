"""Correctness checks and benchmarks for fused compressed pointwise ops."""

import statistics

import torch

from LCT.compression.format import DistType, Distribution
from LCT.compress import compress, decompress
from LCT.pointwise import pointwise_compressed_dense, POINTWISE_OPS
from LCT.tensor_buffer import Allocation, TensorBuffer


SIZES = [512, 1024, 2048, 4096]


def _buffer(size: int) -> TensorBuffer:
    """Create an aligned fallback arena with room for one operation result."""
    capacity = (size + 64 * 1024 * 1024 + 15) // 16 * 16
    return TensorBuffer(capacity, device="cuda")


def _free(encoded, buffer: TensorBuffer | None) -> None:
    """Release a result's descriptor-backed fallback allocation when present."""
    if buffer is not None and encoded.fallback_descriptor is not None:
        buffer.free(Allocation(encoded.fallback_descriptor, buffer))


def _assert_bits_equal(left: torch.Tensor, right: torch.Tensor) -> None:
    """Compare BF16 tensors by representation so matching NaNs remain equal."""
    assert left.shape == right.shape
    assert torch.equal(
        left.contiguous().view(torch.int16),
        right.contiguous().view(torch.int16),
    )


def _time(function, iterations: int, release=None) -> float:
    """Return median-friendly CUDA event timing after warming the operation."""
    for _ in range(3):
        output = function()
        if release:
            release(output)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        output = function()
        if release:
            release(output)
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iterations


def _compare(first, second, iterations, release=None):
    """Measure two variants in alternating order over three trials."""
    first_times, second_times = [], []
    for trial in range(3):
        ordered = (first, second) if trial % 2 == 0 else (second, first)
        first_ms = _time(ordered[0], iterations, release)
        second_ms = _time(ordered[1], iterations, release)
        target = (first_times, second_times) if trial % 2 == 0 else (second_times, first_times)
        target[0].append(first_ms)
        target[1].append(second_ms)
    return statistics.median(first_times), statistics.median(second_times)


@torch.no_grad()
def _benchmark(n: int, operation) -> tuple[float, float]:
    """Benchmark dense, unfused, and fused variants for one matrix size."""
    shape = (n, 4 * n)
    source = torch.randn(shape, device="cuda").to(torch.bfloat16)
    other = torch.randn_like(source)
    buffer = _buffer(source.numel())
    output_buffer = _buffer(source.numel())
    encoded = compress(source, Distribution(DistType.GAUSSIAN), buffer)
    restored = decompress(encoded)
    alpha = torch.tensor([1.0], device="cuda", dtype=torch.float32)
    op_kwargs = {} if operation.name != "scalar_mul_add" else {"alpha": alpha}
    expected = operation.torch_fn(restored, other)
    actual = pointwise_compressed_dense(encoded, other, operation, **op_kwargs)
    compressed_result = pointwise_compressed_dense(
        encoded, other, operation, **op_kwargs,
        dense_output=False, buffer=output_buffer,
    )
    _assert_bits_equal(source, restored)
    _assert_bits_equal(expected, actual)
    _assert_bits_equal(expected, decompress(compressed_result))
    _free(compressed_result, output_buffer)

    dense = lambda: operation.torch_fn(decompress(encoded), other)
    fused = lambda: pointwise_compressed_dense(encoded, other, operation, **op_kwargs)
    iterations = 100
    dense_ms, fused_dense_ms = _compare(dense, fused, iterations)
    print(
        f"shape={shape!s:>14s}  dense_output: dense={dense_ms:7.4f} ms  "
        f"fused={fused_dense_ms:7.4f} ms  reduction={(dense_ms - fused_dense_ms) / dense_ms:6.2%}"
    )

    unfused = lambda: compress(fused(), encoded.distribution, output_buffer)
    fused_compressed = lambda: pointwise_compressed_dense(
        encoded, other, operation, **op_kwargs,
        dense_output=False, buffer=output_buffer,
    )
    release = lambda value: _free(value, output_buffer)
    unfused_ms, fused_compressed_ms = _compare(unfused, fused_compressed, iterations, release)
    print(
        f"{'':12s} compressed_output: unfused={unfused_ms:7.4f} ms  "
        f"fused={fused_compressed_ms:7.4f} ms  reduction={(unfused_ms - fused_compressed_ms) / unfused_ms:6.2%}"
    )
    _free(encoded, buffer)
    del source, other, encoded, restored
    torch.cuda.empty_cache()
    return fused_dense_ms, fused_compressed_ms


def main() -> None:
    """Benchmark pointwise operations on generated inputs."""
    total_fused_dense_ms = 0.0
    total_fused_compressed_ms = 0.0
    for operation in POINTWISE_OPS.values():
        print(f"\n{operation.name}")
        for n in SIZES:
            fused_dense_ms, fused_compressed_ms = _benchmark(n, operation)
            total_fused_dense_ms += fused_dense_ms
            total_fused_compressed_ms += fused_compressed_ms

    print(f"Total fused dense time: {total_fused_dense_ms:.5g}ms")
    print(f"Total fused compressed time: {total_fused_compressed_ms:.5g}ms")


if __name__ == "__main__":
    main()
