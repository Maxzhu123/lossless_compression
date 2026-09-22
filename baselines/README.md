# Lossless BF16 experiment baselines

Portable wrappers for the official SplitZip, DFloat11 and ZipNN implementations.
Source, CUDA PTX, C dependencies and licenses are copied into `_vendor`; none of
these three packages needs to be installed. Imports use local package names,
without changing `sys.path` or relying on globally installed baseline packages.

NVIDIA nvCOMP is also supported through `NVComp("LZ4")`,
`NVComp("Cascaded")`, and `NVComp("Bitcomp")`. It requires the separately
installed `nvidia.nvcomp` Python bindings and a compatible CUDA environment;
the dependency loads only when used. All three are enabled by default in
`experiments/bench_baselines.py`, with separate timing CSVs and printed ratios.
They encode raw BF16 bytes without quantization. Compressed payloads use
PyTorch-owned storage trimmed to the actual encoded size. Codec construction,
allocation, and the trimming copy are included in encode/decode timings, as
applicable. Empty CUDA tensors are handled without invoking nvCOMP.

## Setup

Use the project's Python environment, with PyTorch and NumPy available.

- **SplitZip:** CUDA PyTorch + Triton, already used by LCT.
- **DFloat11:** PyTorch + NumPy + a C compiler for the default native encoder
  (the reference encoder requires no compiler). The CUDA path loads the upstream PTX using the
  NVIDIA driver via Python's `ctypes`; CuPy, nvcc and model-loading packages are
  not needed. The small `dahuffman` dependency is vendored. A compatible NVIDIA
  driver supporting PTX 8.2 is required for GPU decoding. CPU decoding is a slow
  local reference implementation, not the paper's runtime.
- **ZipNN:** PyTorch + NumPy + safetensors; a C compiler, Python development
  headers, setuptools and POSIX pthreads to build the native extension. All
  required FiniteStateEntropy sources are included. Build once per Python ABI
  and target platform, from the project root:

```bash
python baselines/_vendor/zipnn/build.py build_ext --inplace
```

This builds locally and does not install anything. Compiled `.so`/`.pyd` and
build products are ignored by Git. ZipNN's native backend is CPU-only. Its
wrapper copies CUDA inputs to CPU and restores decoded outputs to their original
device; CPU inputs are copied too because the upstream encoder mutates buffers.

## Tensor API

```python
import torch
from baselines import SplitZip, DFloat11, ZipNN
from baselines.benchmark import benchmark

x = torch.randn(1024, 4096, dtype=torch.bfloat16, device="cuda")
calibration = torch.randn_like(x)  # use representative held-out data in experiments

codecs = [SplitZip(calibration), DFloat11(), ZipNN(threads=4)]
for codec in codecs:
    compressed = codec.compress(x)
    restored = codec.decompress(compressed)
    assert torch.equal(x.view(torch.int16), restored.view(torch.int16))
    print(compressed.compressed_bytes, compressed.memory_size())
    print(benchmark(codec, x, warmup=1, iterations=3))
```

DFloat11 uses a local compiled C encoder by default, preserving the upstream
payload format and CUDA decoder. A C compiler (`cc`) builds the small shared
library on first use and caches it in `baselines/__pycache__`. This build is
outside benchmark timing. Use `DFloat11(encoder="reference")` for the original
Python encoder. Encoding still fits a fresh Huffman table and includes CPU/GPU
staging on every call. Report native encoder measurements as a local optimization,
not upstream encoder performance.

Each module also exposes `compress` and `decompress` functions:

```python
from baselines import splitzip, dfloat11, zipnn
packed = dfloat11.compress(x)
x_restored = dfloat11.decompress(packed)
# Convenience SplitZip fitting is charged to this call:
packed = splitzip.compress(x, calibration=calibration)
```

`SplitZip.calibrate(sample)` returns top-16 coverage. Use the class and calibrate
once outside timed loops for paper-style experiments. Every payload owns a copy
of its decode codebook; it remains decodable after recalibration or with a fresh
wrapper instance. No input tensor is retained by the compressed object.

All wrappers accept dense BF16 tensors, including noncontiguous, scalar and
empty tensors, and restore shape, dtype, device and exact bits. Strides and
autograd history are not preserved. The convenience SplitZip encoder requires
a nonempty calibration sample even for an empty input.

**DFloat11 restriction:** upstream's CUDA lookup format uses byte values
240–255 as links to other tables. Inputs with those exponent fields (including
Inf/NaN and some very large finite numbers) are rejected explicitly. There is no
silent substitute codec. SplitZip and ZipNN support all BF16 bit patterns.
DFloat11 also checks its decoder's table and signed 32-bit size limits.

## Measurements and MLP experiments

```bash
python -m baselines.benchmark --device cuda --elements 65536 --iterations 5
python -m baselines.benchmark --method zipnn --device cpu --threads 4
python -m unittest baselines.test_baselines -v
```

The benchmark verifies bitwise recovery and input immutability, then reports
synchronized wall-clock encode/decode latency separately. Calibration is outside
timing for SplitZip. Allocations, table construction for DFloat11, private input
copies for ZipNN, and all CPU/GPU staging are included. JIT/driver warmup is outside
timed iterations. These measurements must not be presented as kernel-only paper
throughput. CPU runs of `--method all` explicitly skip SplitZip.

- `compressed_bytes`: codec buffer bytes needed for decoding, including decode
  tables. Excludes Python shape/device metadata; this is not a serialized file
  size. SplitZip follows its released payload layout, including a 32-bit chunk
  ID per escape, rather than the paper's simplified 3-byte escape estimate.
- `memory_size()`: unique retained tensor storage and byte-buffer allocations,
  including padding and SplitZip construction scratch. Excludes Python object
  overhead, shared encoder state, driver/JIT caches and transient peak memory.
- `storage_ratio`: compressed/original bytes (smaller is better), matching
  `benchmarks/prepare.py`; undefined (`NaN`) for an empty input.

`mlp_model/mlp_train.py` currently uses `MyCompressed`, custom autograd functions
and `SparseSGDM` with LCT-specific operations. These helpers let you benchmark
actual BF16 weights or activations before wrapping them in `MyCompressed`.
They are tensor codecs, not drop-in replacements for that training system:
DFloat11 and ZipNN do not implement LCT's compressed sparse updates or autograd.
The existing training loop is unchanged.

See [SOURCES.md](SOURCES.md) for pinned revisions, licenses and local changes.
