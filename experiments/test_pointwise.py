"""Correctness checks and benchmarks for fused compressed pointwise ops."""

import statistics

import torch

from compress.code_storage import Distribution, DistType, NoiseLevel
from compress.compress import compress, decompress
from compress.ops import compressed_add
from compress.tensor_buffer import Allocation, TensorBuffer


def _buffer(size: int) -> TensorBuffer:
    capacity = (size + 64 * 1024 * 1024 + 15) // 16 * 16
    return TensorBuffer(capacity, device="cuda")


def _free(encoded, buffer: TensorBuffer | None) -> None:
    if buffer is not None and encoded.fallback_descriptor is not None:
        buffer.free(Allocation(encoded.fallback_descriptor, buffer))


def _assert_bits_equal(left: torch.Tensor, right: torch.Tensor) -> None:
    assert left.shape == right.shape
    assert torch.equal(
        left.contiguous().view(torch.int16),
        right.contiguous().view(torch.int16),
    )


@torch.no_grad()
def _check(source, other, distribution, *, buffered=True) -> None:
    buffer = _buffer(source.numel()) if buffered else None
    output_buffer = _buffer(source.numel()) if buffered else None
    encoded = compress(source, distribution, buffer)
    restored = decompress(encoded)
    actual = compressed_add(encoded, other)
    compressed = compressed_add(
        encoded, other, output="compressed", buffer=output_buffer,
    )
    _assert_bits_equal(source, restored)
    _assert_bits_equal(restored + other, actual)
    _assert_bits_equal(restored + other, decompress(compressed))
    _free(encoded, buffer)
    _free(compressed, output_buffer)


def _time(function, iterations: int, release=None) -> float:
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
def _benchmark(size: int) -> None:
    source = torch.randn(size, device="cuda").to(torch.bfloat16)
    other = torch.randn_like(source)
    buffer = _buffer(size)
    output_buffer = _buffer(size)
    encoded = compress(source, Distribution(DistType.GAUSSIAN), buffer)
    _assert_bits_equal(decompress(encoded) + other, compressed_add(encoded, other))

    baseline = lambda: decompress(encoded) + other
    fused = lambda: compressed_add(encoded, other)
    iterations = 50 if size <= 10_000_003 else 20 if size <= 50_000_003 else 10
    baseline_ms, fused_ms = _compare(baseline, fused, iterations)
    print(
        f"n={size / 1e6:7.3f}M  dense: baseline={baseline_ms:7.4f} ms  "
        f"fused={fused_ms:7.4f} ms  speedup={baseline_ms / fused_ms:5.3f}x  "
        f"reduction={(baseline_ms - fused_ms) / baseline_ms:6.2%}"
    )

    dense_then_compress = lambda: compress(fused(), encoded.distribution, output_buffer)
    compressed = lambda: compressed_add(
        encoded, other, output="compressed", buffer=output_buffer,
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
    del source, other, encoded
    torch.cuda.empty_cache()


def main() -> None:
    generator = torch.Generator(device="cuda").manual_seed(17)

    # Raw fallback and a non-contiguous dense operand.
    source = torch.randn((17, 31), device="cuda", generator=generator).to(torch.bfloat16)
    other = torch.randn((31, 17), device="cuda", generator=generator).to(torch.bfloat16).T
    _check(source, other, Distribution(DistType.GAUSSIAN))

    # Arbitrary BF16 bits exercise NaNs, infinities and substantial overflow.
    bits = torch.randint(
        -32768, 32768, (1_000_003,), dtype=torch.int16,
        device="cuda", generator=generator,
    )
    source = bits.view(torch.bfloat16)
    other = torch.randn(source.shape, device="cuda", generator=generator).to(torch.bfloat16)
    noisy = Distribution(DistType.EMPIRICAL, noise_level=NoiseLevel.HIGH)
    _check(source, other, noisy)
    _check(source, other, noisy, buffered=False)
    print("correctness checks passed")

    for size in (1_000_003, 10_000_003, 50_000_003, 200_000_003):
        _benchmark(size)


if __name__ == "__main__":
    main()
