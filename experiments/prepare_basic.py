"""Basic bufferless benchmark for the bfloat16 compression codec.

This harness exercises the codec without a shared TensorBuffer.  In this mode
compress() allocates fallback metadata and data as ordinary GPU tensors and
uses base offset 0, so the same kernels work with a private fallback buffer.
"""

import time
from random import Random

import torch

from compress.compress import compress, decompress
from prepare import (
    CASES,
    ITERS,
    SHAPE_OPTIONS,
    SHAPE_SEED,
    WARMUP,
    get_compressed_size,
    make_data,
)


def run_case(name, n, distribution, max_ratio):
    x = make_data(name, n, distribution)

    # Warmup + correctness pass using the bufferless path.
    for _ in range(WARMUP):
        compressed = compress(x, distribution=distribution)
        restored = decompress(compressed)
        assert torch.equal(x, restored), f"roundtrip mismatch: {name}"

    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(ITERS):
        compressed = compress(x, distribution=distribution)
        restored = decompress(compressed)
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) / ITERS * 1000.0

    ratio = get_compressed_size(compressed) / x.nbytes
    assert ratio <= max_ratio, (
        f"{name} n={n}: ratio {ratio:.4f} exceeds {max_ratio:.4f}"
    )

    print(
        f"{name:32s} n={n / 1e6:6.0f}M  "
        f"time={elapsed_ms:7.3f} ms  ratio={ratio:.4f}"
    )
    del x, compressed, restored
    torch.cuda.empty_cache()
    return elapsed_ms


def main():
    shape_rng = Random(SHAPE_SEED)
    sizes = [SHAPE_OPTIONS[0]] * (len(CASES) // 2)
    sizes += [SHAPE_OPTIONS[1]] * (len(CASES) - len(sizes))
    shape_rng.shuffle(sizes)
    scheduled_cases = [
        (*case, size) for case, size in zip(CASES, sizes)
    ]
    weights = [weight for _, _, _, weight, _ in scheduled_cases]
    total_weight = sum(weights)
    weights = [weight / total_weight for weight in weights]
    print(f"WEIGHTINGS = {[round(weight, 4) for weight in weights]}")
    print(f"SHAPES = {[shape for *_, shape in scheduled_cases]}")

    total_time = 0.0
    for weight, (name, distribution, max_ratio, _, n) in zip(
        weights, scheduled_cases
    ):
        elapsed_ms = run_case(name, n, distribution, max_ratio)
        total_time += weight * elapsed_ms

    print("passed")
    print(f"Total time: {total_time:.5g}ms")


if __name__ == "__main__":
    main()
