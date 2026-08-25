"""Correctness checks and benchmarks for fused compressed pointwise ops."""

import statistics

import torch

from compress.code_storage import Distribution, DistType
from compress.compress import compress, decompress
from compress.ops.pointwise import pointwise_compressed_dense
from compress.ops.registry import POINTWISE_OPS
from compress.tensor_buffer import Allocation, TensorBuffer


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


def _compare(baseline, fused, iterations, release=None):
    """Measure baseline and fused paths in alternating order over three trials."""
    baseline_times, fused_times = [], []
    for trial in range(3):
        first, second = (baseline, fused) if trial % 2 == 0 else (fused, baseline)
        first_time = _time(first, iterations, release)
        second_time = _time(second, iterations, release)
        target = (baseline_times, fused_times) if trial % 2 == 0 else (fused_times, baseline_times)
        target[0].append(first_time)
        target[1].append(second_time)
    return statistics.median(baseline_times), statistics.median(fused_times)


@torch.no_grad()
def _benchmark(size: int, operation) -> None:
    """Benchmark dense and compressed output modes for one size and operation."""
    source = torch.randn(size, device="cuda").to(torch.bfloat16)
    other = torch.randn_like(source)
    buffer = _buffer(size)
    output_buffer = _buffer(size)
    encoded = compress(source, Distribution(DistType.GAUSSIAN), buffer)
    restored = decompress(encoded)
    expected = operation.torch_fn(restored, other)
    actual = pointwise_compressed_dense(encoded, other, operation)
    compressed_result = pointwise_compressed_dense(
        encoded, other, operation,
        dense_output=False, buffer=output_buffer,
    )
    _assert_bits_equal(source, restored)
    _assert_bits_equal(expected, actual)
    _assert_bits_equal(expected, decompress(compressed_result))
    _free(compressed_result, output_buffer)

    baseline = lambda: operation.torch_fn(decompress(encoded), other)
    fused = lambda: pointwise_compressed_dense(encoded, other, operation)
    iterations = 20
    baseline_ms, fused_ms = _compare(baseline, fused, iterations)
    print(
        f"n={size / 1e6:7.3f}M  dense: baseline={baseline_ms:7.4f} ms  "
        f"fused={fused_ms:7.4f} ms  speedup={baseline_ms / fused_ms:5.3f}x  "
        f"reduction={(baseline_ms - fused_ms) / baseline_ms:6.2%}"
    )

    dense_then_compress = lambda: compress(fused(), encoded.distribution, output_buffer)
    compressed = lambda: pointwise_compressed_dense(
        encoded, other, operation,
        dense_output=False, buffer=output_buffer,
    )
    release = lambda value: _free(value, output_buffer)
    baseline_ms, fused_ms = _compare(
        dense_then_compress, compressed, iterations, release,
    )
    print(
        f"{'':12s} compressed: baseline={baseline_ms:7.4f} ms  "
        f"fused={fused_ms:7.4f} ms  speedup={baseline_ms / fused_ms:5.3f}x  "
        f"reduction={(baseline_ms - fused_ms) / baseline_ms:6.2%}"
    )
    _free(encoded, buffer)
    del source, other, encoded, restored
    torch.cuda.empty_cache()


def main() -> None:
    """Benchmark pointwise operations on generated inputs."""

    for operation in POINTWISE_OPS.values():
        print(f"\n{operation.name}")
        for size in (1_000_003, 10_000_003, 50_000_003, 200_000_003):
            _benchmark(size, operation)


if __name__ == "__main__":
    main()
