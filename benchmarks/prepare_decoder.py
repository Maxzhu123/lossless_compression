"""Compare complete Triton and TileLang decoder pipelines on identical payloads.

Run in the optimiser environment with ``python -m benchmarks.prepare_decoder``.
Normal timings include Python dispatch and output allocation. Optional CUDA
graph timings isolate GPU execution and are reported separately.
"""

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path
import statistics
import time

import torch

from benchmarks.prepare import CASES, SIZE_WEIGHTS, make_data
from LCT.comp_format import DistType, Distribution, NoiseLevel
from LCT.compress import compress
from LCT.kernels.generic.decode import decode as decode_triton
from LCT.kernels.tilelang.decode import decode as decode_tilelang
from LCT.tensor_buffer import TensorBuffer


def _time(function, iterations):
    for _ in range(5):
        function()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        function()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000 / iterations


def _graph(function):
    for _ in range(3):
        function()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(3):
            function()
    return graph


def _gpu_time(graph, iterations):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / (3 * iterations)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=list(SIZE_WEIGHTS))
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--trials", type=int, default=7)
    parser.add_argument("--cuda-graphs", action="store_true")
    parser.add_argument("--csv", type=Path, default=Path("benchmarks/results/tilelang_decoder.csv"))
    args = parser.parse_args()
    if args.iterations < 1 or args.trials < 1 or min(args.sizes) < 1:
        parser.error("sizes, iterations, and trials must be positive")
    timestamp = datetime.now(timezone.utc).isoformat()
    gpu = torch.cuda.get_device_name()
    records = []
    old_total = new_total = total_weight = 0
    for n in args.sizes:
        for name, max_ratio in CASES:
            _, family, noise = name.split("/")
            distribution = Distribution(DistType(family), noise_level=NoiseLevel[noise.upper()])
            source = make_data(name, n, distribution)
            buffer = TensorBuffer((n + 64 * 1024**2 + 15) // 16 * 16, device="cuda")
            encoded = compress(source, distribution, buffer)
            functions = (lambda: decode_triton(encoded), lambda: decode_tilelang(encoded))
            try:
                for function in functions:
                    restored = function()
                    assert torch.equal(restored.view(torch.int16), source.view(torch.int16)), (name, n)
                    del restored
                assert encoded.memory_size() / source.nbytes <= max_ratio
                methods = [("wall", functions, _time)]
                graphs = None
                if args.cuda_graphs:
                    graphs = tuple(_graph(function) for function in functions)
                    methods.append(("cuda_graph", graphs, _gpu_time))
                for method, variants, timer in methods:
                    samples = [[], []]
                    for trial in range(args.trials):
                        for index in ((0, 1) if trial % 2 == 0 else (1, 0)):
                            samples[index].append(timer(variants[index], args.iterations))
                    before, after = (statistics.median(values) for values in samples)
                    record = dict(timestamp=timestamp, gpu=gpu, case=name, elements=n, method=method,
                                  triton_ms=before, tilelang_ms=after, reduction_pct=(1 - after / before) * 100)
                    records.append(record)
                    print(f"{name:30s} n={n / 1e6:7.2f}M {method:10s} before={before:.4f} ms after={after:.4f} ms reduction={record['reduction_pct']:.2f}%", flush=True)
                    if method == "wall":
                        weight = SIZE_WEIGHTS.get(n, 1)
                        old_total += before * weight
                        new_total += after * weight
                        total_weight += weight
                del methods, graphs
            finally:
                encoded.free()
            del encoded, source, buffer, functions
            torch.cuda.empty_cache()
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(f"Weighted decoder time: {old_total / total_weight:.5f} -> {new_total / total_weight:.5f} ms")
    print(f"Results saved to {args.csv}")


if __name__ == "__main__":
    main()
