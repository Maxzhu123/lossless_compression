"""Run with python -m experiments.bench_baselines; edit settings below."""
import csv
from pathlib import Path
import sys
import time

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

    if any(name not in {"splitzip", "dfloat11", "zipnn"} for name in methods):
        raise ValueError("Methods must be splitzip, dfloat11, or zipnn")
    if not torch.cuda.is_available():
        raise RuntimeError("This experiment requires CUDA")
    print(f"GPU: {torch.cuda.get_device_name()} | PyTorch: {torch.__version__} | CUDA: {torch.version.cuda}", flush=True)
    print(f"Input: {elements:,} BF16 elements ({elements * 2 / 1024 ** 3:g} GiB) | {family.value} | seed={seed}")
    parameters = {
        DistType.GAUSSIAN: f"std={Distribution(family, param=gaussian_std).param}",
        DistType.EMPIRICAL: f"scale={Distribution(family, param=empirical_scale).param}",
        DistType.LAPLACE: f"scale={Distribution(family, param=laplace_scale).param}",
        DistType.GAMMA: f"shape={gamma_shape}, scale={gamma_scale}",
    }
    print(f"Distribution parameters: {parameters[family]}")
    print(f"Warmup: {warmup} | Iterations: {iterations} | ZipNN threads: {threads}")
    print("Timing: synchronized wall-clock; CPU/GPU staging included; SplitZip calibration excluded.")
    print("Generating input...", flush=True)
    x = make_data(seed)
    free_bytes, total_bytes = torch.cuda.mem_get_info(x.device)
    print(f"GPU memory after input generation: {free_bytes / 1024 ** 3:.2f} / {total_bytes / 1024 ** 3:.2f} GiB free", flush=True)
    output_dir = Path(__file__).resolve().parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in methods:
        print(f"\n{name}: preparing codec...", flush=True)
        if name == "splitzip":
            print(f"Calibrating on held-out data (seed={seed + 1})...", flush=True)
            calibration = make_data(seed + 1)
            codec = SplitZip(calibration)
            del calibration
        elif name == "dfloat11":
            print(f"DFloat11 encoder: {dfloat11_encoder} (CPU); decoder: upstream CUDA.", flush=True)
            codec = DFloat11(encoder=dfloat11_encoder)
        else:
            codec = ZipNN(threads=threads)
        print("Checking bitwise recovery, warming up, and measuring...", flush=True)
        torch.cuda.synchronize(x.device)
        torch.cuda.reset_peak_memory_stats(x.device)
        start = time.perf_counter()
        result = benchmark(codec, x, warmup=warmup, iterations=iterations)
        elapsed = time.perf_counter() - start
        print("Bitwise recovery and input immutability: PASS")
        print(f"Encode: {result.compress_ms:.3f} ± {result.compress_std_ms:.3f} ms (mean ± std)")
        print(f"Decode: {result.decompress_ms:.3f} ± {result.decompress_std_ms:.3f} ms (mean ± std)")
        gib = result.original_bytes / 1024 ** 3
        print(f"Throughput (input bytes): encode {gib * 1000 / result.compress_ms:.2f} GiB/s | decode {gib * 1000 / result.decompress_ms:.2f} GiB/s")
        print(f"Compressed: {result.compressed_bytes / 1024 ** 2:.2f} MiB | storage ratio: {result.storage_ratio:.4f} | saved: {(1 - result.storage_ratio) * 100:.2f}%")
        print(f"Retained codec storage: {result.retained_bytes / 1024 ** 2:.2f} MiB")
        print(f"Peak PyTorch GPU allocation (including input/validation): {torch.cuda.max_memory_allocated(x.device) / 1024 ** 3:.2f} GiB | benchmark elapsed: {elapsed:.2f} s")
        output = output_dir / f"{name}_{family.value}_{x.nbytes / 1024 ** 3:g}gib.csv"
        with output.open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(("encode_mean_ms", "encode_std_ms", "decode_mean_ms", "decode_std_ms"))
            writer.writerow((result.compress_ms, result.compress_std_ms,
                             result.decompress_ms, result.decompress_std_ms))
        print(f"CSV: {output}", flush=True)
        del codec


if __name__ == "__main__":
    main()
