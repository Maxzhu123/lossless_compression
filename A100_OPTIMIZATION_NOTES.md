# A100 LCT optimization notes

Recorded: 2026-09-23.

These notes describe the changes applied to the remote checkout at
`/root/lossless_compression` on the NVIDIA A100 80GB PCIe instance accessed with
`ssh -p 30053 root@65.109.7.69`. They do not imply that the changes have been
copied into this local checkout. At the last inspection, the remote changes
were uncommitted: eight modified files and two new files.

## Final benchmark

Workload: 1 GiB Gaussian BF16 tensor, 536,870,912 elements, standard deviation
2.0, seed 0. `experiments/benchmark_lct.py` used three warmups and 50 measured
iterations. Reported uncertainty is the standard error of the mean.

| Metric | Final result |
| --- | ---: |
| Encode | 1.730 ± 0.004 ms |
| Decode | 1.923 ± 0.003 ms |
| Compression ratio, original/compressed | 1.4545× |

Timings include the host-side compression/decompression pipeline and GPU
synchronization, not just the main kernels. Short experimental runs and direct
kernel timings are not interchangeable with this final benchmark. Earlier
2 GiB measurements predate the latest payload-layout and pair-decoder changes;
the final configuration has not been benchmarked at 2 GiB.

## Applied changes

### 1. Shorter streams for clean Gaussian data

`LCT/codec/runtime.py:geometry()` returns the following for
`DistType.GAUSSIAN` with `NoiseLevel.CLEAN`:

```python
block_symbols, lanes, steps, fixed_words = 32768, 256, 128, 12
```

- Block size reduced from 65,536 to 32,768 elements.
- Lanes per storage block remain 256.
- Symbols per stream reduced from 256 to 128.
- Fixed 32-bit words per stream reduced from 24 to 12.
- Fixed Huffman payload remains three bits per symbol; overflow remains exact.
- Other distributions and noise levels retain their previous geometry.

This is a conditional geometry override, not a global change to
`BLOCK_SYMBOLS`, `LANES`, or `LANE_BITS`.

### 2. Block-local compressed payload ordering

The fixed exponent payload changes from `[word, block, lane]` to
`[block, word, lane]`. Each block's compressed words are stored together.

```python
# Previous ordering
payload_offset = word * n_streams + block * N_LANES + lane

# New ordering
payload_offset = block * FIXED_WORDS * N_LANES + word * N_LANES + lane
```

`CompressedTensor` gains `blocked_payload: bool = False`. Newly compressed
tensors set it to `True`; readers dispatch using this field. Both raw-BF16
encoding and encoding of precomputed components write the new order.

This flag distinguishes payload orders for otherwise matching codec geometry.
It does not supply compatibility with older geometry or XOR/linear swizzle
formats, and old software does not understand the new payload ordering.

### 3. Two-symbol lookup decoder

New file: `LCT/kernels/pair_decode.py`.

- Builds a 14-bit prefix table: 16,384 entries × 4 bytes = 64 KiB.
- Each entry packs `length0`, biased exponent0, `length1`, biased exponent1.
- A zero `length1` marks a pair requiring the existing serial fallback.
- If all lanes in a decode group resolve their pair, the group skips fallback
  logic and uses both cached exponents and lengths.
- Escape codes retain exact single-symbol handling; partial blocks retain a
  masked tail path.
- The pair table is built during each applicable decompression call and is
  temporary workspace, not part of the stored compressed tensor.
- Runtime selection currently uses the pair decoder for clean Gaussian data
  with at least `2**20` logical elements. Smaller tensors and other
  distributions use the single-symbol decoder. Performance tuning focused on
  the 1 GiB workload, not every size covered by that dispatch condition.

### 4. Smaller, single-warp decode groups

Storage still has 256 lanes per block, but the pair decoder launches four
independent programs per block, each processing 64 logical lanes with one
warp. The logical lanes are not the same thing as hardware threads: one warp
contains 32 threads.

```python
groups_per_block = N_LANES // DECODE_LANES  # 256 // 64 = 4
block = program_id // groups_per_block
lane_base = (program_id % groups_per_block) * DECODE_LANES
```

The lookup-fallback decision is local to this smaller group and requires no
cross-warp reduction. Measured on the Gaussian workload:

| Logical lanes per decode group | Pair-table miss rate | Groups entering lookup fallback |
| ---: | ---: | ---: |
| 32 | 0.192% | 5.96% |
| 64 | 0.192% | 11.57% |
| 128 | 0.192% | 21.81% |
| 256 | 0.192% | 38.86% |

The per-pair miss rate is unchanged; fewer neighboring lanes are affected by
each miss. This is distinct from overflow-storage fallback. Thirty-two-lane
groups had fewer fallback branches but were slower overall than 64-lane groups.

### 5. Larger overflow-processing tiles

- Overflow count/metadata compaction block: 1,024 → 4,096 streams.
- Overflow payload compaction tile: 32 → 256.
- Standalone decode overflow-scatter tile: 64 → 512.
- Scatter grids use the selected tile size consistently.

### 6. Skip inactive scatter-loop iterations

The overflow-scatter kernel starts at the earliest fallback step among valid
streams in its tile instead of iterating from zero:

```python
first_step = tl.min(tl.where(valid, start, N_STEPS), axis=0)
for step in tl.range(first_step, N_STEPS):
    ...
```

### 7. Final launch settings

| Kernel | Warps | Stages | Register cap | Other settings |
| --- | ---: | ---: | ---: | --- |
| Encode | 8 | 2 | Uncapped | Existing unroll factor 4 |
| Pair decode | 1 | 1 | 64 | `DECODE_LANES=64`, `PAIR_BITS=14`, `ON_DEMAND=False` |
| Single-symbol dense decode | 2 | 1 | 96 | `ON_DEMAND=False` |
| Overflow count/metadata compaction | 4 | 2 | Uncapped | `BLOCK=4096` |
| Overflow payload compaction | 4 | 2 | Uncapped | `TILE=256` |
| Overflow scatter | 2 | 2 | Uncapped | Standalone decode uses `TILE=512` |

The tuned lists are pinned single configurations. Pair-decode launch settings
are supplied directly in `decode_dense()`. The original fused-operation
`DECODE_AUTOTUNE_CONFIGS` and `DUAL_DECODE_AUTOTUNE_CONFIGS` remain separate.

`ON_DEMAND=False` means prefetching the next compressed word unconditionally.
The encode unroll factor, original additive swizzle, and existing cache/loop
hints remain unless explicitly changed above.

### 8. Fused-operation compatibility

Updated compressed add, multiply, scaled add, and dual-compressed kernels to
read either fixed-payload order. Dual-compressed kernels accept separate
layout flags for their two operands, so mixed payload orders work.

Payload-order flags were added to relevant autotune keys. Fused operations
that produce compressed output use the updated component encoder. They have
not been converted to use the new pair lookup decoder.

### 9. Regression checks

New file: `experiments/check_pair_decode.py`.

Validated:

- Bitwise round trips with both payload orders.
- Odd-sized tails.
- Fused add, multiply, scaled add, and dual-compressed add.
- Dense and compressed outputs.
- Mixed payload orders for dual-compressed operations.
- Escape-heavy arbitrary BF16 bit patterns.
- Private and buffer-backed overflow storage.

The standard benchmark's bitwise round-trip checks also passed.

## Swizzle status

The original additive mapping is active:

```python
logical_lane = (lane + (row & 255)) & 255
```

XOR was implemented and tested, then reverted. Linear mapping is not active.
At the last inspection, the `_encode_impl` docstring still contained stale XOR
and odd-multiplier wording; the executable mapping is additive. That comment
should be corrected separately.

## Experiments not retained

- XOR and fully linear lane mapping.
- Alternative storage widths/block sizes that lost overall throughput.
- Alternative loop unrolling and cache policies without end-to-end wins.
- A 32-bit-prefix shortcut for the second decoded symbol.
- The initial pair-table version that still executed fallback work routinely.
- Combined-length pair-table entries.
- Explicit shared-memory payload staging.
- Three-symbol decoding: correct but around 3.0 ms in short decode-only runs,
  versus roughly 1.9 ms for the winning smaller-group pair decoder.
- Thirty-two- and 128-lane decode groups in place of the selected 64 lanes.

Rejected group/shared-memory/triple prototypes were moved out of the remote
source tree to `/tmp/lct-decode-experiments.zzOcEC`. Temporary tuning scripts
and earlier local copies are experimental artifacts, not the final source of
truth.

## Remote files changed

Modified:

1. `LCT/codec/autotune.py`
2. `LCT/codec/runtime.py`
3. `LCT/codec/pointwise.py`
4. `LCT/comp_tensor.py`
5. `LCT/kernels/main_kernels.py`
6. `LCT/kernels/pointwise.py`
7. `LCT/kernels/pointwise_scalar.py`
8. `LCT/kernels/pointwise_scalar_dual.py`

Added:

9. `LCT/kernels/pair_decode.py`
10. `experiments/check_pair_decode.py`

Benchmark rows were appended to
`experiments/results/lct_gaussian.csv`. Temporary 2 GiB runs changed the
element count in memory; the benchmark file remains configured for 1 GiB.
