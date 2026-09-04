"""Correctness checks and benchmarks for dense @ compressed matmul."""
import statistics

import torch

from LCT.compression.format import DistType, Distribution
from LCT.codec.runtime import compress_dense
from LCT.compress import A_compB, decompress
from LCT.tensor_buffer import TensorBuffer


# Benchmark matrix sizes as (n, 4n): the dense activation is [n, 4n] and the
# compressed weight is [4n, n], so the result is [n, n].
SIZES = [512, 1024, 1500, 2048, 3000, 4000, 4096]
WARMUP = 3
ITERATIONS = 30
TRIALS = 4


def _buffer(weight_bytes: int) -> TensorBuffer:
    """Create an aligned fallback arena for a compressed weight."""
    capacity = (weight_bytes + 64 * 1024 * 1024 + 15) // 16 * 16
    return TensorBuffer(capacity, device="cuda")


def _assert_bits_equal(left: torch.Tensor, right: torch.Tensor) -> None:
    """Compare BF16 tensors by representation so matching NaNs remain equal."""
    assert left.shape == right.shape
    assert torch.equal(
        left.contiguous().view(torch.int16),
        right.contiguous().view(torch.int16),
    )


def _time(function, iterations: int) -> float:
    """Return average CUDA event timing after warming the operation."""
    for _ in range(WARMUP):
        function()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        function()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iterations


def _compare(functions: tuple) -> list[float]:
    """Measure all paths in alternating order over trials."""
    times = [[] for _ in functions]
    for trial in range(TRIALS):
        order = tuple(range(len(functions)))
        if trial % 2:
            order = tuple(reversed(order))
        for index in order:
            times[index].append(_time(functions[index], ITERATIONS))
    return [statistics.median(sample) for sample in times]


@torch.no_grad()
def _benchmark(n: int) -> tuple[float, float]:
    """Benchmark one n @ 4n dense @ compressed matmul."""
    k = 4 * n
    generator = torch.Generator(device="cuda").manual_seed(n)
    activations = torch.randn(
        (n, k), device="cuda", generator=generator
    ).to(torch.bfloat16)
    weight = (
        torch.randn((k, n), device="cuda", generator=generator) * 0.25
    ).to(torch.bfloat16)

    buffer = _buffer(weight.numel() * weight.element_size())
    encoded = compress_dense(
        weight, Distribution(DistType.GAUSSIAN, 0.25),
        buffer, allow_raw=False,
    )
    restored = decompress(encoded)
    _assert_bits_equal(weight, restored)

    expected = activations @ weight
    actual = A_compB(activations, encoded)
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)

    dense = lambda: activations @ weight
    compressed = lambda: A_compB(activations, encoded)
    dense_ms, compressed_ms = _compare((dense, compressed))

    overhead = (compressed_ms - dense_ms) / dense_ms
    print(
        f"n={n:5d}  dense={dense_ms:7.4f} ms  compressed={compressed_ms:7.4f} ms  "
        f"overhead={overhead:6.2%}"
    )

    encoded.free()
    buffer.reset()
    del activations, weight, restored, encoded
    torch.cuda.empty_cache()
    return dense_ms, compressed_ms


def main() -> None:
    """Benchmark dense @ compressed matmul for several n @ 4n sizes."""
    total_baseline_ms = 0.0
    total_time_ms = 0.0
    for n in SIZES:
        dense_ms, compressed_ms = _benchmark(n)
        total_baseline_ms += dense_ms
        total_time_ms += compressed_ms

    print(f"Total dense time: {total_baseline_ms:.5g}ms")
    print(f"Total compressed time: {total_time_ms:.5g}ms")


if __name__ == "__main__":
    main()
