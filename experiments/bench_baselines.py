"""Run with python -m experiments.bench_baselines; edit settings below."""
import csv
from pathlib import Path
import sys

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from baselines import DFloat11, SplitZip, ZipNN
from baselines.benchmark import benchmark
from benchmarks.prepare import make_empirical, make_gaussian, make_laplace
from LCT.comp_format import DistType, Distribution


def main():
    # Same task settings as experiments/benchmark_lct.py. Edit before running.
    family = DistType.GAUSSIAN  # GAUSSIAN, EMPIRICAL, LAPLACE, or GAMMA
    methods = ["splitzip", "dfloat11", "zipnn"]
    elements = (1024 ** 3) // 2
    warmup, iterations = 3, 50
    seed = 0
    gaussian_std = 2.0
    empirical_scale = 0.5
    laplace_scale = 1.5
    gamma_shape, gamma_scale = 0.82, 2.43
    threads = 1  # ZipNN CPU threads.
    dfloat11_encoder = "native"  # native (compiled C) or reference (upstream Python).

    def make_data(data_seed):
        if family == DistType.GAUSSIAN:
            std = Distribution(family, param=gaussian_std).param
            return make_gaussian(elements, std=std, seed=data_seed)
        if family == DistType.EMPIRICAL:
            scale = Distribution(family, param=empirical_scale).param
            return make_empirical(elements, scale=scale, seed=data_seed)
        if family == DistType.LAPLACE:
            scale = Distribution(family, param=laplace_scale).param
            return make_laplace(elements, scale=scale, seed=data_seed)
        if family != DistType.GAMMA:
            raise ValueError(f"Unknown distribution: {family}")
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            torch.cuda.manual_seed(data_seed)
            gamma = torch.distributions.Gamma(
                torch.tensor(gamma_shape, device="cuda"),
                torch.tensor(1 / gamma_scale, device="cuda"),
            )
            return gamma.sample((elements,)).to(torch.bfloat16)

    x = make_data(seed)
    output_dir = Path(__file__).resolve().parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in methods:
        if name == "splitzip":
            calibration = make_data(seed + 1)
            codec = SplitZip(calibration)
            del calibration
        else:
            codec = {
                "dfloat11": lambda: DFloat11(encoder=dfloat11_encoder),
                "zipnn": lambda: ZipNN(threads=threads),
            }[name]()
        result = benchmark(codec, x, warmup=warmup, iterations=iterations, verify=False)
        print(
            f"{name} | encode: {result.compress_ms:.3f} ± {result.compress_sem_ms:.3f} ms"
            f" | decode: {result.decompress_ms:.3f} ± {result.decompress_sem_ms:.3f} ms"
            f" | compression ratio (original/compressed): {result.original_bytes / result.compressed_bytes:.3f}x",
            flush=True,
        )
        output = output_dir / f"{name}_{family.value}_{x.nbytes / 1024 ** 3:g}gib.csv"
        with output.open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(("encode_mean_ms", "encode_sem_ms", "decode_mean_ms", "decode_sem_ms"))
            writer.writerow((result.compress_ms, result.compress_sem_ms,
                             result.decompress_ms, result.decompress_sem_ms))
        del codec


if __name__ == "__main__":
    main()
