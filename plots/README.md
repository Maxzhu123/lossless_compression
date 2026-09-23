# Paper plots

The shared styling and rendering helpers are copied from
`/home/maccyz/Documents/bitsparse/plots`. Keep new plotting scripts here and
use these helpers so the figures have a consistent appearance.

Generated PDFs belong in `plots/plots/`, separate from
the plotting code. Plotting scripts do not update `paper/figures/`. Copy finished
PDFs there manually when they are ready for the paper.

## Shared format

- Times-style serif text with STIX mathematics; base text is 12 pt, labels
  13 pt, and ticks/legends 11 pt.
- Default size: 5.5 × 3.4 inches. Wide size: 8 × 4.5 inches; use
  `WIDE_FONT_SCALE` (1.5) when a wide figure needs larger text at page scale.
- White background, thin light-gray grid, no top/right spines, open markers,
  and consistent legend spacing.
- Save vector PDFs with tight bounds and embedded Type 42 fonts. Raster
  elements use 300 dpi on export; interactive display uses 150 dpi.
- Series colors and markers are keyed by name in `plot_lib.CONFIG_STYLES`:
  `Dense`, `LCT`, `LCT-buffer`, `DFloat11`, `SplitZip`, and `ZipNN`.
  `LCT-buffer` denotes the shared-arena configuration. Keep the same name
  across figures. Ordered groups such as layers use `sample_group_colors`.
- Put units in axis labels and convert measurements explicitly. Do not
  combine timings or memory measurements with different definitions.

## Script structure

Separate reading results, building a figure, and saving it. For example:

```python
from pathlib import Path
import matplotlib.pyplot as plt
from plot_lib import plot_series, format_axes, finish_plot
from plot_tables import render


def build(series):
    # series: (label, x_values, y_values) tuples loaded from real results.
    fig, ax = plt.subplots()
    plot_series(ax, series)
    format_axes(ax, xlabel="Elements", ylabel="Encode time / ms")
    finish_plot(ax)
    return fig, ax


def save(series):
    render(
        {"encode_time.pdf": series}, build,
        output_dir=Path(__file__).resolve().parent / "plots",
        show=False,
    )
```

Run scripts directly with `python plots/plot_name.py` from the repository root.
Use direct sibling imports within `plots/`. Run figure creation and saving within `plot_style()` when
not using `render()`; the context restores global Matplotlib settings afterward.
For continuous axes with fractional values, pass `xformat=None` or a suitable
format to `format_axes` (its default uses integer ticks).

`plot_tables.parse_series` also accepts whitespace-separated measurement
tables with one column per named series. Write missing entries as `-`, not
blank cells. `parse_grouped` handles tables with several metrics per series.
For histograms, use the same style context with Matplotlib's `stairs` or
other appropriate artists; the style does not require converting data to lines.

Weight-distribution plots should load the saved CPU tensors from
`artefacts/weight_distribution_results.pt`, rather than loading the model
or rerunning data collection. No BitSparse measurements or generated figures
are included in this directory.

## Codec throughput

Run `python plots/plot_codec_throughput.py` to generate
`plots/plots/codec_throughput.pdf`. The script retains the supplied mean timings
and plots encoding and decoding throughput in separate panels, excluding LZ4
and Cascaded. Input sizes are interpreted as MiB (1024 = 1 GiB); throughput is
`(size_mib / 1024) / (time_ms / 1000)` for both operations. The plot starts at
16 MiB on a logarithmic size axis, with the same linear throughput scale across panels. Existing series
reuse the shared colors and markers; nvCOMP styles are also defined centrally.

## Weight distributions

`plot_weight_distributions.py` exports only `plots/plots/weight_distributions.pdf`
(vector). Selection, counts, extrema, and displayed mass are computed from the
saved histograms on demand and returned by `plot_distributions()` in memory:

```sh
python plots/plot_weight_distributions.py
```

It selects **parameter groups with strictly more than 500 million values**,
aggregated over layers using the grouping in `llm_analysis/weight_distribution.py`.
It uses a density histogram, shared axes, and a logarithmic density scale.
The common horizontal range contains at least 99.9999% of every selected group;
normalization still includes all values, including those outside the view.
Raw counts remain unchanged. For display, a Gaussian kernel smooths histogram
mass (standard deviation 3 bins for weights, 5 bins for activations). The curves
are renormalized to preserve mass; exact-zero counts are removed before
smoothing and restored afterward. Smoothing respects observed support, including
nonnegative activations, and introduces no pseudocounts. Set `smoothing_bins=0`
in either script to display the original histogram steps. The log-axis limits
are retained from the unsmoothed data.

The default source is `artefacts/weight_distribution_results.pt`, falling back to
`artefacts/weight_distribution_large_results.pt` if the full results are absent.
Edit `results` inside `main()` to select a particular saved file. No model weights are
loaded when plotting.

For a machine without enough free RAM to load the full model, the companion
collector reads the local safetensors checkpoint in CPU chunks:

```sh
python plots/collect_weight_histograms.py
```

It uses a metadata-only model to identify the same parameter groups, collects
every value in the selected groups (no sampling), and saves a separate
`artefacts/weight_distribution_large_results.pt` with 0.001-wide bins and a
provenance sidecar. The six-group result is deliberately separate from a full
all-group analysis. The collector never downloads weights or allocates CUDA
tensors. Set `threads` inside `main()` to control CPU parallelism (default 4).

Suggested paper caption: *Weight distributions of the six Nemotron-H-8B
parameter groups containing more than 500M values. Histograms aggregate weights
across layers and use a shared range covering at least 99.9999% of each group.
Densities are normalized by the full group size and shown on a logarithmic scale.*

## Weight exponents

Collect exact eight-bit BF16 exponent-field counts on the CPU, then plot them:

```sh
python plots/collect_weight_histograms.py
python plots/plot_weight_exponents.py
```

Set `exponents_only = True` in the collector's `main()` before running it.
The counts are saved separately in
`artefacts/weight_exponent_distribution_results.pt`. The plot exports
only `plots/plots/weight_exponent_distributions.pdf`, with the same six
groups, ordering, and colors as the raw-weight figure. It uses probabilities
on a shared linear scale. Exponents below -20 are combined in a separate
hatched bar; all weights remain in the normalization. `plot_exponents()` returns full
exponent entropy, the number of most-frequent symbols needed to reach 99%,
exact-zero/subnormal fractions, and the displayed tail mass. The 99% symbol
set need not be contiguous. These statistics are computed from the saved counts
on demand and kept in memory, with no JSON plot output. Collection and plotting never use CUDA.

## Activation distributions

Collect fresh activation histograms and exact BF16 exponent counts, then render
the same PDF layouts used for weights:

```sh
python llm_analysis/activation_distribution.py
python plots/plot_activation_distributions.py
python plots/plot_activation_exponents.py
```

The collector uses the local checkpoint at `artefacts/Nemotron-H-8B-Base-8K`
and `llm_analysis/sample_text.txt`: four batches of one 1,024-token sequence,
with no KV cache. On CUDA it runs BF16 inference and saves both
`artefacts/activation_distribution_results.pt` and
`artefacts/activation_exponent_distribution_results.pt`. The float32 CPU path
saves value histograms only. Exponent counts are collected from the actual BF16
bits during the same forward passes, not inferred from value histograms.

“Largest” means categories with **more than 500M observed activation values**,
aggregated across captured layers and batches. These are observation counts,
not parameter counts or the size of one simultaneously live tensor. In the
default run this selects FFN pre-activation, FFN hidden, Mamba projected,
Mamba hidden, and logits. Both plots use the same order and colors.

The scripts have editable settings inside `main()` and export only:

- `plots/plots/activation_distributions.pdf`: shared symmetric value axes,
  logarithmic density, and at least 99.9999% displayed mass per group. The
  collector uses 0.1-wide bins over ±1,000, with overflow buckets for more
  extreme values; normalization includes all observations. Exact zeros remain
  in the histogram.
- `plots/plots/activation_exponent_distributions.pdf`: exact exponent-field
  probabilities, with bias removed for normal exponents. The hatched `<-20`
  bar includes exact zeros, subnormals, and smaller normal exponents.

As with weights, plotting loads only saved CPU counts, shares `plot_style()`,
and returns group counts, extrema, coverage, and exponent statistics in memory.
Plotting neither runs the model nor updates `paper/figures/`.

## Muon momentum evolution

Run `python nanogpt/record_momentum_distribution.py` to record the first eight
nanoGPT optimizer steps using the current `qwen_time_mem.py` settings. Exact
per-buffer BF16 exponent counts and run metadata are saved under
`artefacts/muon_momentum/<run>/`.

Run `python plots/plot_momentum_distribution.py` to plot the latest saved run,
or set `CSV_PATH` in that script to choose one. It exports
`plots/plots/momentum_evolution.pdf` using the shared styles. Each buffer's
histogram is normalized separately, then averaged with equal weight. All steps
appear on one logarithmic-probability plot; exact zeros are shown separately
at the left and remain part of the probability normalization.

For the matching nanoGPT weight plot, run
`python nanogpt/record_weight_distribution.py`, then
`python plots/plot_nanogpt_weight_exponents.py`. Counts and metadata are saved
under `artefacts/nanogpt_weights/<run>/`, and the figure is exported to
`plots/plots/nanogpt_weight_evolution.pdf`. It averages the normalized histograms
of all BF16 weight matrices equally, including the embedding and LM head;
FP32 biases and normalization gains are excluded.

`python plots/collect_nanogpt_weight_exponents.py` adds saved-checkpoint counts
for steps 300, 1500, and 3300 from `nanogpt/logs/2026-07-04_00-06-23` to
`artefacts/nanogpt_weight_checkpoints/2026-07-04_00-06-23/`. The weight plot
labels all curves as `Step <step>`. Checkpoint FP32 matrices are converted to BF16, matching their forward
compute representation. These are different training runs, not a continuous
trajectory. The momentum plot uses only the recorded early steps because these
checkpoints do not contain optimizer state.

New `train_gpt_lct.py` checkpoints contain `model`, `step`, and
`muon_momentum`. Momentum is saved as dense CPU BF16 tensors keyed by parameter
name, regardless of its training storage format; uninitialized step-zero buffers
are `None`. These files do not include AdamW state or RNG state for a full resume.
The weight collector accepts both these checkpoints and the older flat model files.

To add later momentum curves, set `CHECKPOINT_DIR` and `STEPS` in
`collect_nanogpt_momentum_exponents.py` and run it. Counts are saved under
`artefacts/muon_momentum_checkpoints/<run>/`. Set `CHECKPOINT_CSV` in
`plot_momentum_distribution.py` to that run's `exponents.csv` and rerun the plot.
Older checkpoints without momentum are rejected rather than reconstructed.
