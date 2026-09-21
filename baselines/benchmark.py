"""End-to-end codec timings, including staging, allocation and table building."""
from dataclasses import asdict, dataclass
from math import sqrt
import json
from pathlib import Path
from statistics import mean, stdev
import sys
import time

import torch


@dataclass(frozen=True)
class Result:
    method: str
    original_bytes: int
    compressed_bytes: int
    retained_bytes: int
    storage_ratio: float
    compress_ms: float
    decompress_ms: float
    compress_sem_ms: float = float("nan")
    decompress_sem_ms: float = float("nan")


def benchmark(codec, tensor, *, warmup=1, iterations=5, verify=True):
    """Optionally validate bits, warm up, then time encode/decode independently.

    Calibrate SplitZip before calling. CPU↔GPU copies are included for ZipNN
    and DFloat11 encoding. Times are wall-clock, synchronized on tensor.device;
    they are not kernel-only throughput numbers. No autograd is retained.
    SEM is sample standard deviation / sqrt(iterations), or NaN for one sample.
    Set verify=False to skip the input clone and round-trip correctness pass.
    """
    if warmup < 0 or iterations < 1:
        raise ValueError("warmup must be nonnegative and iterations positive")

    def sync():
        if tensor.device.type == "cuda":
            torch.cuda.synchronize(tensor.device)

    if verify:
        original = tensor.detach().clone()
    encoded = codec.compress(tensor)
    if verify:
        restored = codec.decompress(encoded)
        sync()
        expected = original.contiguous().reshape(-1).view(torch.int16)
        if not torch.equal(expected, tensor.detach().contiguous().reshape(-1).view(torch.int16)):
            raise AssertionError("compress modified the input")
        if (restored.shape != tensor.shape or restored.dtype != tensor.dtype
                or restored.device != tensor.device
                or not torch.equal(expected, restored.contiguous().reshape(-1).view(torch.int16))):
            raise AssertionError("baseline failed bitwise round-trip verification")
        del original, restored, expected

    for _ in range(warmup):
        temporary = codec.compress(tensor)
        output = codec.decompress(temporary)
        sync()
        del temporary, output

    compress_ms, decompress_ms = [], []
    for _ in range(iterations):
        sync()
        start = time.perf_counter()
        temporary = codec.compress(tensor)
        sync()
        compress_ms.append((time.perf_counter() - start) * 1000)
        del temporary
        sync()
        start = time.perf_counter()
        output = codec.decompress(encoded)
        sync()
        decompress_ms.append((time.perf_counter() - start) * 1000)
        del output
    return Result(encoded.method, tensor.nbytes, encoded.compressed_bytes,
                  encoded.memory_size(), encoded.storage_ratio,
                  mean(compress_ms), mean(decompress_ms),
                  stdev(compress_ms) / sqrt(iterations) if iterations > 1 else float("nan"),
                  stdev(decompress_ms) / sqrt(iterations) if iterations > 1 else float("nan"))


def main():
    # Edit these settings before running this file directly or from the IDE.
    method = "all"  # all, splitzip, dfloat11, or zipnn
    device = "cuda" if torch.cuda.is_available() else "cpu"
    elements = (1024 ** 3) // 2  # 1 GiB per BF16 tensor (2 bytes per element).
    warmup = 1
    iterations = 5
    threads = 1  # ZipNN CPU threads

    if __package__ in (None, ""):
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from baselines import DFloat11, SplitZip, ZipNN

    if method not in ("all", "splitzip", "dfloat11", "zipnn"):
        raise ValueError("method must be all, splitzip, dfloat11, or zipnn")
    if elements < 1:
        raise ValueError("elements must be positive")
    generator = torch.Generator(device=device).manual_seed(0)
    x = torch.randn(elements, dtype=torch.bfloat16, device=device, generator=generator)
    calibration = torch.randn(elements, dtype=torch.bfloat16, device=device, generator=generator)
    names = ["splitzip", "dfloat11", "zipnn"] if method == "all" else [method]
    for name in names:
        if name == "splitzip":
            if x.device.type != "cuda":
                if method != "all":
                    raise ValueError("SplitZip requires device='cuda'")
                print(json.dumps({"method": name, "skipped": "CUDA required"}))
                continue
            codec = SplitZip(calibration)
        elif name == "dfloat11":
            codec = DFloat11()
        else:
            codec = ZipNN(threads=threads)
        print(json.dumps(asdict(benchmark(codec, x, warmup=warmup, iterations=iterations))))


if __name__ == "__main__":
    main()
