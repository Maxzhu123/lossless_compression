# A100 LCT optimization notes

Recorded: 2026-09-23.

Describes changes applied to the remote checkout at `/root/lossless_compression`
on the NVIDIA A100 80GB PCIe instance (`ssh -p 30053 root@65.109.7.69`). They
are not copied into this local checkout. At last inspection the remote changes
were uncommitted: eight modified files and two new files.

## Final benchmark

Workload: 1 GiB Gaussian BF16, 536,870,912 elements, std 2.0, seed 0;
`experiments/benchmark_lct.py`, three warmups, 50 measured iterations;
uncertainty is the standard error of the mean.

| Metric | Final result |
| --- | ---: |
| Encode | 1.730 ± 0.004 ms |
| Decode | 1.923 ± 0.003 ms |
| Compression ratio, original/compressed | 1.4545× |

Timings include the host-side pipeline and GPU synchronization, not just the
main kernels, so short experimental runs and direct kernel timings are not
interchangeable with them. Earlier 2 GiB measurements predate the latest
payload-layout and pair-decoder changes; the final configuration is unbenchmarked
at 2 GiB.

## Applied changes

### 1. Shorter streams for clean Gaussian data

`LCT/codec/runtime.py:geometry()` now returns
`block_symbols, lanes, steps, fixed_words = 32768, 256, 128, 12` for
`DistType.GAUSSIAN` + `NoiseLevel.CLEAN` (block 65,536 → 32,768; symbols/stream
256 → 128; fixed 32-bit words/stream 24 → 12; lanes stay 256). The fixed Huffman
payload stays three bits per symbol; overflow stays exact. This is a conditional
override, not a change to `BLOCK_SYMBOLS`/`LANES`/`LANE_BITS`; other distributions
and noise levels keep their previous geometry.

### 2. Block-local compressed payload ordering

The fixed exponent payload changes from `[word, block, lane]` to
`[block, word, lane]`:

```python
payload_offset = word * n_streams + block * N_LANES + lane              # previous
payload_offset = block * FIXED_WORDS * N_LANES + word * N_LANES + lane  # new
```

`CompressedTensor` gains `blocked_payload: bool = False`; new tensors set it
`True` and readers dispatch on it. Both raw-BF16 and precomputed-component
encoding write the new order. The flag only distinguishes payload orders for
otherwise matching geometry — it is not compatible with older geometry or
XOR/linear swizzle formats, and old software cannot read the new order.

### 3. Two-symbol lookup decoder

New file `LCT/kernels/pair_decode.py`:

- 14-bit prefix table, 16,384 entries × 4 B = 64 KiB; each entry packs
  `length0`, biased exponent0, `length1`, biased exponent1 (`length1 == 0` marks
  a pair needing the existing serial fallback).
- Groups resolving every lane skip fallback and use both cached values.
- Escapes keep exact single-symbol handling; partial blocks keep a masked tail.
- The table is built per decompression call as temporary workspace, not stored.
- Used for clean Gaussian data with ≥ `2**20` logical elements; smaller tensors
  and other distributions use the single-symbol decoder. Tuning focused on the
  1 GiB workload, not every size the dispatch covers.

### 4. Smaller, single-warp decode groups

Storage keeps 256 lanes/block, but the pair decoder launches four programs per
block, each handling 64 logical lanes with one warp (a warp is 32 threads).

```python
groups_per_block = N_LANES // DECODE_LANES  # 256 // 64 = 4
block = program_id // groups_per_block
lane_base = (program_id % groups_per_block) * DECODE_LANES
```

The fallback decision is group-local (no cross-warp reduction). Measured on the
Gaussian workload:

| Logical lanes/group | Pair-table miss rate | Groups entering lookup fallback |
| ---: | ---: | ---: |
| 32 | 0.192% | 5.96% |
| 64 | 0.192% | 11.57% |
| 128 | 0.192% | 21.81% |
| 256 | 0.192% | 38.86% |

Per-pair miss rate is unchanged; fewer lanes are affected per miss. This is
distinct from overflow-storage fallback. 32-lane groups had fewer fallback
branches but were slower overall than 64.

### 5. Larger overflow tiles and skipping inactive iterations

- Overflow count/metadata compaction block: 1,024 → 4,096 streams.
- Overflow payload compaction tile: 32 → 256.
- Standalone decode overflow-scatter tile: 64 → 512; grids follow the tile size.
- The scatter kernel starts at the earliest fallback step among valid streams
  rather than iterating from zero:

```python
first_step = tl.min(tl.where(valid, start, N_STEPS), axis=0)
for step in tl.range(first_step, N_STEPS):
    ...
```

### 6. Final launch settings

| Kernel | Warps | Stages | Reg cap | Other |
| --- | ---: | ---: | ---: | --- |
| Encode | 8 | 2 | none | unroll factor 4 |
| Pair decode | 1 | 1 | 64 | `DECODE_LANES=64`, `PAIR_BITS=14`, `ON_DEMAND=False` |
| Single-symbol dense decode | 2 | 1 | 96 | `ON_DEMAND=False` |
| Overflow count/metadata compaction | 4 | 2 | none | `BLOCK=4096` |
| Overflow payload compaction | 4 | 2 | none | `TILE=256` |
| Overflow scatter | 2 | 2 | none | standalone decode `TILE=512` |

Tuned lists are pinned single configurations; pair-decode settings are set
directly in `decode_dense()`. The fused-operation `DECODE_AUTOTUNE_CONFIGS` and
`DUAL_DECODE_AUTOTUNE_CONFIGS` remain separate. `ON_DEMAND=False` prefetches the
next compressed word unconditionally. The encode unroll factor, original additive
swizzle, and existing cache/loop hints are unchanged unless noted.

### 7. Fused-operation compatibility and regression checks

Fused compressed add/multiply/scaled-add/dual-compressed kernels now read either
payload order (dual-compressed takes separate flags per operand, so mixed orders
work). Payload-order flags joined the relevant autotune keys. Fused operations
producing compressed output use the updated component encoder but still use the
single-symbol decoder.

New file `experiments/check_pair_decode.py` validated: bitwise round trips with
both payload orders; odd-sized tails; fused add/multiply/scaled-add/dual add;
dense and compressed outputs; mixed payload orders; escape-heavy arbitrary BF16
patterns; private and buffer-backed overflow. The standard benchmark's bitwise
round-trip checks also passed.

## Swizzle status

The original additive mapping is active:
`logical_lane = (lane + (row & 255)) & 255`. XOR was implemented, tested, and
reverted; linear mapping is not active. The `_encode_impl` docstring still has
stale XOR/odd-multiplier wording and should be corrected separately.

## Experiments not retained

- XOR and fully linear lane mapping.
- Alternative storage widths/block sizes that lost throughput.
- Alternative unrolling and cache policies without end-to-end wins.
- A 32-bit second-symbol prefix shortcut.
- The initial pair table (still ran fallback routinely) and combined-length entries.
- Explicit shared-memory payload staging.
- Three-symbol decoding: correct but ~3.0 ms short decode-only vs ~1.9 ms for the pair decoder.
- 32- and 128-lane decode groups (vs selected 64).

Rejected prototypes were moved to `/tmp/lct-decode-experiments.zzOcEC`. Temporary
tuning scripts and earlier local copies are not the source of truth.

## Remote files changed

Modified: `LCT/codec/autotune.py`, `LCT/codec/runtime.py`, `LCT/codec/pointwise.py`,
`LCT/comp_tensor.py`, `LCT/kernels/main_kernels.py`, `LCT/kernels/pointwise.py`,
`LCT/kernels/pointwise_scalar.py`, `LCT/kernels/pointwise_scalar_dual.py`.
Added: `LCT/kernels/pair_decode.py`, `experiments/check_pair_decode.py`.

Benchmark rows were appended to `experiments/results/lct_gaussian.csv`. Temporary
2 GiB runs changed the element count in memory; the file remains configured for 1 GiB.

