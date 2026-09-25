# Lossless BF16 Compression Autoresearch

This repository implements a lossless CUDA/Triton codec for `torch.bfloat16` tensors. It stores each value's sign and seven mantissa bits verbatim, then Huffman-encodes the exponent byte. Fixed-size per-lane streams keep the hot path regular; streams that exceed their bit budget are placed in fallback storage. The objective is to reduce end-to-end GPU time without sacrificing losslessness or the required compression ratio. Use the optimiser mamba env to run experiments, nsight profiler is installed inside as well. 

## Scope

**Read but never modify:**

- `benchmarks/prepare.py`, `benchmarks/prepare_matmul.py` and `benchmarks/prepare_pointwise.py` are the authoritative benchmark and correctness harnesses. Run it from the repository root with `python3 experiments/prepare.py`.
- Any other file in `benchmarks/` is an experiment harness: it may be run or read, but must not be changed.

**Editable implementation:** everything under `LCT/`:

- `LCT/compress.py` owns the public `compress()` / `decompress()` flow, codec geometry, allocation decisions, and Triton launches.
- `main_kernels.py` contains the encode, decode, fallback, and center estimation kernels.
- `huffman_tables.py` and `probabilities.py` construct the distribution-aware codebooks.
- `tensor_buffer.py` implements the GPU-resident first-fit allocator used by the benchmark's fallback buffer.
- `comp_tensor.py` defines the compressed representation and distribution selectors.

Do not alter the input generation, benchmark cases, correctness assertions, size thresholds, iteration counts, or timing logic. Do not add dependencies or bypass compression by retaining the original input.

## Benchmark contract

`benchmarks/prepare.py` runs compression **and** decompression for sixteen deterministic distribution/codec combinations. Eight inputs have 50 million elements and eight have 200 million elements. The final score is a weight-normalized average of their per-case milliseconds; lower is better.

Each candidate must satisfy all of the following:

- Exact bitwise round trip: `torch.equal(x, decompress(compress(x)))`.
- The compressed allocation ratio must not exceed the case-specific limit (roughly 0.71 to 0.92 of the original BF16 storage).
- Correct ownership and release of `TensorBuffer` fallback regions, so every timed iteration except the final one can reuse the shared buffer.

The benchmark prints `passed` and then `Total time: …ms` on success. The total is the primary metric; individual timings identify regressions or gains by distribution and input size.

## Working method

1. Inspect the current branch and worktree before making a change. Preserve unrelated user changes, including changes outside `LCT/`.
2. Establish a baseline before tuning. Save benchmark output to a log rather than streaming the large run into context:

   ```bash
   python3 experiments/prepare.py > run.log 2>&1
   rg '^(passed|Total time:)' run.log
   ```

3. Make one focused, reversible change under `LCT/`. Prefer changes that address an observed hot path: launch geometry, memory coalescing, stream layout, decode work, fallback compaction, or host/device synchronization.
4. Run the benchmark, check for `passed`, record the total and relevant per-case timings, then keep a change only when it is reliably faster while still meeting every ratio limit.
5. Use minimal, targeted smoke checks when needed after changes to `tensor_buffer.py` or allocation/descriptors, without adding persistent test files. Revert unsuccessful experiments without touching unrelated work.

The full benchmark uses very large GPU tensors and can take time. Treat a crash, missing `passed`, assertion failure, or CUDA error as a failed experiment. If a trial runs longer than two minutes, stop it, inspect its log, and discard or repair the change.

## Design constraints and useful details

- The benchmark always supplies a persistent `TensorBuffer`; optimize this descriptor-backed path first. The buffer is sized for a worst-case raw exponent fallback plus metadata and is reset between benchmark runs.
- The compressed size calculation counts unique allocations. For a descriptor-backed fallback, it counts only that tensor's reserved region, not the whole shared arena.
- `Distribution` selects empirical, Gaussian, or Laplace tables, as well as clean, medium, or high-noise geometry. Shifted Gaussian data relies on the sampled exponent center, so preserve its symmetry between encode and decode.
- `compress()` has a raw-source fallback when fixed payload alone would be larger than the input. Preserve its shape, dtype, and lossless behavior.
- Kernel indexing must handle partial final blocks and use masks for out-of-range elements. Fixed stream payloads include padding words for safe decode lookahead; do not remove this safety margin without proving the accesses remain valid.
- Avoid host synchronization in the benchmarked buffer path. In particular, device scalars and descriptors exist to avoid materializing fallback sizes on the CPU.

Favor simple, maintainable improvements. A substantial measured gain or a clear simplification is worth keeping; a marginal gain that makes correctness, memory ownership, or kernels much harder to reason about is not.
