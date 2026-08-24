"""Distribution-aware BF16 exponent-compression benchmark."""
import math
import time
from random import Random
from dataclasses import fields
import torch

from compress.compress import compress, decompress
from compress.tensor_buffer import Allocation, TensorBuffer
from compress.code_storage import CompressedTensor, CompressionLayout, Distribution, DistType


SHAPE_OPTIONS = [50_000_000, 200_000_000]
SHAPE_SEED = 0
WARMUP = 3
ITERS = 15


def get_compressed_size(data: CompressedTensor) -> int:
    """Return GPU allocation bytes owned by one compressed tensor.

    A descriptor-backed tensor counts its reserved region, rather than the
    entire shared arena that may also serve other compressed tensors.
    """
    allocations: dict[tuple[torch.device, int], int] = {}
    for f in fields(data):
        tensor = getattr(data, f.name)
        if isinstance(tensor, torch.Tensor):
            if f.name == "fallback_buffer" and data.fallback_descriptor is not None:
                continue
            storage = tensor.untyped_storage()
            key = (tensor.device, storage.data_ptr())
            allocations[key] = storage.nbytes()
    total = sum(allocations.values())
    if data.fallback_descriptor is not None:
        total += int(data.fallback_descriptor[1].item())
    return total


def free_compressed(
    data: CompressedTensor,
    buffer: TensorBuffer,
) -> None:
    """Release every buffer-backed allocation held by a compressed tensor."""
    if (
        data.fallback_descriptor is None
        or data.fallback_buffer is None
        or data.fallback_buffer is not buffer.data
    ):
        return
    buffer.free(Allocation(data.fallback_descriptor, buffer))


def make_standard(n: int, scale: float = 0.5, seed: int = 0) -> torch.Tensor:
    """Sample signed values with an exponential body and power-law tail."""
    G = torch.Generator(device="cuda").manual_seed(seed)
    tail_probability = 0.05
    tail_alpha = 2.8
    tail_start = -scale * math.log(tail_probability)
    u = torch.rand(n, device="cuda", dtype=torch.float32, generator=G)
    body = u < (1.0 - tail_probability)
    values = torch.empty_like(u)
    values[body] = -scale * torch.log1p(-u[body])
    values[~body] = tail_start * (
        tail_probability / (1.0 - u[~body])
    ) ** (1.0 / (tail_alpha - 1.0))
    signs = torch.randint(
        0, 2, (n,), device="cuda", dtype=torch.int8, generator=G
    ).to(torch.float32)
    return (values * (signs * 2.0 - 1.0)).to(torch.bfloat16)


def make_gaussian_values(
    n: int, mean: float = 0.0, std: float = 2.0, seed: int = 0,
) -> torch.Tensor:
    G = torch.Generator(device="cuda").manual_seed(seed)
    values = torch.randn(n, device="cuda", dtype=torch.float32, generator=G)
    return values * std + mean


def make_gaussian(
    n: int, mean: float = 0.0, std: float = 2.0, seed: int = 0,
) -> torch.Tensor:
    return make_gaussian_values(n, mean=mean, std=std, seed=seed).to(
        torch.bfloat16
    )


def make_laplace(
    n: int, scale: float = 1.5, seed: int = 0,
) -> torch.Tensor:
    G = torch.Generator(device="cuda").manual_seed(seed)
    u = torch.rand(n, device="cuda", dtype=torch.float32, generator=G) - 0.5
    values = -scale * torch.sign(u) * torch.log1p(-2.0 * u.abs())
    return values.to(torch.bfloat16)


def make_localized_noise(n: int, noise_fraction: float = 0.2, seed: int = 0):
    """Gaussian values with a contiguous uniform-value noise region."""
    values = make_gaussian_values(n, mean=0.0, std=2.0, seed=seed)
    start = n // 2
    end = min(n, start + int(n * noise_fraction))
    G = torch.Generator(device="cuda").manual_seed(seed + 1)
    values[start:end] = torch.empty(
        end - start, device="cuda", dtype=torch.float32
    ).uniform_(
        -32.0, 32.0, generator=G
    )
    return values.to(torch.bfloat16)


def _bf16_ratio(exponent_ratio: float) -> float:
    """Convert an exponent-stream ratio to a total BF16 storage ratio."""
    return (1.0 + exponent_ratio) / 2.0


# Each case: (name, distribution, max_total_bf16_ratio, weight)
DIST_STANDARD_CLEAN = Distribution(DistType.STANDARD)
DIST_STANDARD_MEDIUM = Distribution(
    DistType.STANDARD, layout=CompressionLayout.MEDIUM
)
DIST_STANDARD_HIGH = Distribution(
    DistType.STANDARD, layout=CompressionLayout.HIGH
)
DIST_GAUSSIAN_CLEAN = Distribution(DistType.GAUSSIAN)
DIST_GAUSSIAN_MEDIUM = Distribution(
    DistType.GAUSSIAN, layout=CompressionLayout.MEDIUM
)
DIST_GAUSSIAN_HIGH = Distribution(
    DistType.GAUSSIAN, layout=CompressionLayout.HIGH
)
DIST_LAPLACE_CLEAN = Distribution(DistType.LAPLACE)


CASES = [
    ("standard/standard/clean", DIST_STANDARD_CLEAN, _bf16_ratio(0.42), 5),
    ("standard/standard/medium", DIST_STANDARD_MEDIUM, _bf16_ratio(0.60), 5),
    ("gaussian/gaussian/clean", DIST_GAUSSIAN_CLEAN, _bf16_ratio(0.42), 5),
    ("gaussian/standard/clean", DIST_STANDARD_CLEAN, _bf16_ratio(0.65), 1),
    ("laplace/laplace/clean", DIST_LAPLACE_CLEAN, _bf16_ratio(0.43), 5),
    ("laplace/gaussian/clean", DIST_GAUSSIAN_CLEAN, _bf16_ratio(0.55), 1),
    ("shifted_gaussian/gaussian/clean", DIST_GAUSSIAN_CLEAN, _bf16_ratio(0.42), 5),
    ("shifted_gaussian/standard/clean", DIST_STANDARD_CLEAN, _bf16_ratio(0.65), 1),
    ("localized/gaussian/high", DIST_GAUSSIAN_HIGH, _bf16_ratio(0.83), 5),
    ("localized/standard/high", DIST_STANDARD_HIGH, _bf16_ratio(0.84), 1),
    ("localized/gaussian/medium", DIST_GAUSSIAN_MEDIUM, _bf16_ratio(0.76), 5),
    ("localized/standard/medium", DIST_STANDARD_MEDIUM, _bf16_ratio(0.76), 1),
]


def make_data(
    name: str,
    n: int,
    distribution: Distribution | None = None,
) -> torch.Tensor:
    if name in {"standard/standard/clean", "standard/standard/medium"}:
        scale = distribution.param if distribution is not None else 0.5
        return make_standard(n, scale)
    if name in {"gaussian/gaussian/clean", "gaussian/standard/clean"}:
        return make_gaussian(n)
    if name in {"laplace/laplace/clean", "laplace/gaussian/clean"}:
        return make_laplace(n)
    if name in {
        "shifted_gaussian/gaussian/clean",
        "shifted_gaussian/standard/clean",
    }:
        return make_gaussian(n, mean=50.0)
    if name in {
        "localized/gaussian/high",
        "localized/standard/high",
        "localized/gaussian/medium",
        "localized/standard/medium",
    }:
        return make_localized_noise(n)
    raise ValueError(f"unknown case: {name}")


def run_case(name, n, distribution, max_ratio, buffer):
    x = make_data(name, n, distribution)

    # Correctness pass: allocate, decode, then release the buffer regions.
    compressed = compress(
        x, distribution=distribution, buffer=buffer
    )
    restored = decompress(compressed)
    assert torch.equal(x, restored), f"roundtrip mismatch: {name}"
    free_compressed(compressed, buffer)

    for _ in range(WARMUP):
        compressed = compress(
            x, distribution=distribution, buffer=buffer
        )
        restored = decompress(compressed)
        free_compressed(compressed, buffer)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for i in range(ITERS):
        compressed = compress(
            x, distribution=distribution, buffer=buffer
        )
        restored = decompress(compressed)
        # Keep the final compressed object live so ratio can be measured after
        # the timed loop.  Every earlier iteration is released and reused.
        if i != ITERS - 1:
            free_compressed(compressed, buffer)
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
    free_compressed(compressed, buffer)
    del x, compressed, restored
    torch.cuda.empty_cache()
    return elapsed_ms


def main():
    # Each case runs once. Shuffle a balanced size assignment with a fixed
    # seed, keeping the benchmark reproducible and as close to 50/50 as nine
    # cases allow.
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

    # Persistent buffer used by the codec for fallback storage.  It is sized
    # for one worst-case raw-exponent fallback (max SHAPE_OPTIONS bytes) plus
    # full-size int32 fallback metadata arrays.  Inside run_case buffer-backed
    # regions are freed after each iteration so the space is reused.
    buffer = TensorBuffer(
        max(SHAPE_OPTIONS) + 64 * 1024 * 1024,
        device="cuda",
    )

    total_time = 0.0
    for weight, (name, distribution, max_ratio, _, n) in zip(
        weights, scheduled_cases
    ):
        elapsed_ms = run_case(name, n, distribution, max_ratio, buffer)
        total_time += weight * elapsed_ms

    # Individual buffer-backed regions are freed inside run_case.  Reset the
    # allocator once more so repeated benchmark runs start from a clean state.
    buffer.reset()

    print("passed")
    print(f"Total time: {total_time:.5g}ms")


if __name__ == "__main__":
    main()
