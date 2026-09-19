"""Benchmark 1 GiB of BF16 data; save timing means and standard errors to CSV."""
import csv
from math import sqrt
from pathlib import Path
from statistics import mean, stdev
import time

import torch

from benchmarks.prepare import make_empirical, make_gaussian, make_laplace
from LCT.comp_format import DistType, Distribution
from LCT.compress import compress, decompress
from LCT.tensor_buffer import TensorBuffer


def main():
    # Edit these settings before running.
    family = DistType.GAUSSIAN  # GAUSSIAN, EMPIRICAL, LAPLACE, or GAMMA
    elements = (1024 ** 3) // 2
    warmup, iterations = 3, 50
    seed = 0
    gaussian_std = 2.0
    empirical_scale = 0.5
    laplace_scale = 1.5
    gamma_shape, gamma_scale = 0.82, 2.43  # LCT's Gamma codebook stays fixed.

    if family == DistType.GAUSSIAN:
        distribution = Distribution(family, param=gaussian_std)
        x = make_gaussian(elements, std=distribution.param, seed=seed)
    elif family == DistType.EMPIRICAL:
        distribution = Distribution(family, param=empirical_scale)
        x = make_empirical(elements, scale=distribution.param, seed=seed)
    elif family == DistType.LAPLACE:
        distribution = Distribution(family, param=laplace_scale)
        x = make_laplace(elements, scale=distribution.param, seed=seed)
    else:
        distribution = Distribution(family)
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            torch.cuda.manual_seed(seed)
            gamma = torch.distributions.Gamma(
                torch.tensor(gamma_shape, device="cuda"),
                torch.tensor(1 / gamma_scale, device="cuda"),
            )
            x = gamma.sample((elements,)).to(torch.bfloat16)
    buffer = TensorBuffer(elements + 64 * 1024 * 1024, device=x.device)

    for _ in range(warmup):
        packed = compress(x, distribution=distribution, buffer=buffer)
        restored = decompress(packed)
        assert torch.equal(x.view(torch.int16), restored.view(torch.int16))
        packed.free()
        del packed, restored

    timings = []
    for _ in range(iterations):
        torch.cuda.synchronize(x.device)
        start = time.perf_counter()
        packed = compress(x, distribution=distribution, buffer=buffer)
        torch.cuda.synchronize(x.device)
        encode_ms = (time.perf_counter() - start) * 1000

        start = time.perf_counter()
        restored = decompress(packed)
        torch.cuda.synchronize(x.device)
        decode_ms = (time.perf_counter() - start) * 1000
        timings.append((encode_ms, decode_ms))
        packed.free()
        del packed, restored

    output = Path(__file__).with_name("results") / f"lct_{family.value}_1gib.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    encode_times, decode_times = zip(*timings)
    with output.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("encode_mean_ms", "encode_sem_ms", "decode_mean_ms", "decode_sem_ms"))
        # SEM uses the sample standard deviation; one measurement is insufficient.
        encode_sem = stdev(encode_times) / sqrt(iterations) if iterations > 1 else float("nan")
        decode_sem = stdev(decode_times) / sqrt(iterations) if iterations > 1 else float("nan")
        writer.writerow((mean(encode_times), encode_sem, mean(decode_times), decode_sem))
    print(
        f"LCT | encode: {mean(encode_times):.3f} ± {encode_sem:.3f} ms"
        f" | decode: {mean(decode_times):.3f} ± {decode_sem:.3f} ms",
        flush=True,
    )


if __name__ == "__main__":
    main()
