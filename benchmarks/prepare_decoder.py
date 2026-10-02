"""Decoder-only counterpart to benchmarks.prepare."""

import time
import torch

from benchmarks.prepare import CASES, SIZE_WEIGHTS, WARMUP, ITERS, make_data
from LCT.compress import compress, decompress
from LCT.tensor_buffer import TensorBuffer
from LCT.comp_format import DistType, Distribution, NoiseLevel


def run_case(
    name: str, n: int, max_ratio: float, buffer: TensorBuffer,
) -> float:
    _, family, noise = name.split("/")
    distribution = Distribution(DistType(family), noise_level=NoiseLevel[noise.upper()])
    x = make_data(name, n, distribution)

    # Encode once to prepare the payload, outside decoder timing.
    compressed = compress(x, distribution=distribution, buffer=buffer)
    restored = decompress(compressed)
    assert torch.equal(x, restored), f"roundtrip mismatch: {name}"

    for _ in range(WARMUP):
        restored = decompress(compressed)

    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(ITERS):
        restored = decompress(compressed)
    torch.cuda.synchronize()
    decode_ms = (time.perf_counter() - start) / ITERS * 1000.0

    ratio = compressed.memory_size() / x.nbytes
    assert ratio <= max_ratio, (
        f"{name} n={n}: ratio {ratio:.4f} exceeds {max_ratio:.4f}"
    )

    print(
        f"{name:32s} n={n / 1e6:6.0f}M  "
        f"decode={decode_ms:7.3f} ms"
    )
    compressed.free()
    del x, compressed, restored
    torch.cuda.empty_cache()
    return decode_ms


def main() -> None:
    print(f"SHAPES = {list(SIZE_WEIGHTS)}")

    buffer = TensorBuffer(
        max(SIZE_WEIGHTS) + 64 * 1024 * 1024,
        device="cuda",
    )

    weighted_time = 0.0
    total_time = 0.0
    total_weight = 0
    for n, weight in SIZE_WEIGHTS.items():
        for name, max_ratio in CASES:
            elapsed_ms = run_case(name, n, max_ratio, buffer)
            weighted_time += weight * elapsed_ms
            total_time += elapsed_ms
            total_weight += weight

    print("Passed")
    task_count = len(CASES) * len(SIZE_WEIGHTS)
    print(f"Average time per task: {total_time / task_count:.5g}ms")
    print(f"Final time: {weighted_time / total_weight:.5g}ms")


if __name__ == "__main__":
    main()
