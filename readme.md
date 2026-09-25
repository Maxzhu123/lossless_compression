# LCT: Lossless Tensor Compression with GPU-Friendly Execution

Supplementary code for the accompanying ICLR paper, *LCT: Lossless Tensor Compression with GPU-Friendly Execution*. LCT is a lossless codec for BF16 tensors that compresses exponents while preserving every input bit. The project includes GPU kernels and training integrations for storing activations, weights, and optimizer momentum in compressed form.

## Folder guide

| Folder | Contents and purpose |
| --- | --- |
| [`LCT/`](LCT/) | Core compression library: compressed tensor representations, encoding/decoding, and overflow-buffer management. `compression/` builds probability models and Huffman tables; `codec/` provides runtime and autotuning support; `kernels/` implements GPU operations; `components/` integrates compressed layers, autograd, and optimizers into training. |
| [`baselines/`](baselines/) | Wrappers for SplitZip, DFloat11, ZipNN, and NVIDIA nvCOMP, used to compare compression performance. |
| [`benchmarks/`](benchmarks/) | Tensor generation, correctness checks, and benchmarks for compression, matrix multiplication, pointwise operations, and compressed updates. |
| [`experiments/`](experiments/) | Experiment scripts for measuring LCT and baseline codec performance, including timing results and compression ratios. |
| [`nanogpt/`](nanogpt/) | NanoGPT training and evaluation, with configurable LCT compression, time/memory measurements, and weight/momentum exponent collection. `data/` is the local FineWeb training-data location. |
| [`qwen/`](qwen/) | Qwen3-4B fine-tuning with LCT, including model components, data preparation/loading, optimizer integration, and time/memory measurements. [`train_qwen_lct.py`](qwen/train_qwen_lct.py) is the main fine-tuning script. |
| [`llm_analysis/`](llm_analysis/) | Scripts and notebooks for studying model weight and activation distributions, collecting histograms, and evaluating feed-forward compression. |
| `artefacts/` | Local model checkpoints, prepared datasets, training outputs, and collected analysis results. |

## Running experiments

The main training entry points are `nanogpt/train_gpt_lct.py` and `qwen/train_qwen_lct.py`. Codec comparisons are in `experiments/benchmark_lct.py` and `experiments/bench_baselines.py`.

Dependencies are listed in [`requirements.txt`](requirements.txt). GPU experiments require a compatible CUDA-enabled PyTorch and Triton environment. Before running an experiment, configure the settings and paths in its script and prepare the required datasets and model checkpoints. Refer to the accompanying paper for the experimental configurations.
