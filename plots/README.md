# Paper plots

The shared styling and rendering helpers are copied from
`/home/maccyz/Documents/bitsparse/plots`. Keep new plotting scripts here and
use these helpers so the figures have a consistent appearance.

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
from plots.plot_lib import plot_series, format_axes, finish_plot
from plots.plot_tables import render


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
        output_dir=Path(__file__).resolve().parent,
        show=False,
    )
```

Use `python -m plots.plot_name` from the repository root for scripts with
package imports. Run figure creation and saving within `plot_style()` when
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

## Weight distributions

`plot_weight_distributions.py` exports `weight_distributions.pdf` (vector),
`weight_distributions.png` (300 dpi), and `weight_distributions.json` (source,
selection, counts, extrema, and displayed mass):

```sh
python -m plots.plot_weight_distributions
```

It selects **parameter groups with strictly more than 500 million values**,
aggregated over layers using the grouping in `llm_analysis/weight_distribution.py`.
It uses a density histogram, shared axes, and a logarithmic density scale.
The common horizontal range contains at least 99.9999% of every selected group;
normalization still includes all values, including those outside the view.
Zero-count bins are left empty rather than replaced with pseudocounts.

The default source is `artefacts/weight_distribution_results.pt`, falling back to
`artefacts/weight_distribution_large_results.pt` if the full results are absent.
Use `--results PATH` to select a particular saved file. No model weights are
loaded when plotting.

For a machine without enough free RAM to load the full model, the companion
collector reads the local safetensors checkpoint in CPU chunks:

```sh
python -m plots.collect_weight_histograms
```

It uses a metadata-only model to identify the same parameter groups, collects
every value in the selected groups (no sampling), and saves a separate
`artefacts/weight_distribution_large_results.pt` with 0.001-wide bins and a
provenance sidecar. The six-group result is deliberately separate from a full
all-group analysis. The collector never downloads weights or allocates CUDA
tensors. `--threads` controls CPU parallelism (default 4).

Suggested paper caption: *Weight distributions of the six Nemotron-H-8B
parameter groups containing more than 500M values. Histograms aggregate weights
across layers and use a shared range covering at least 99.9999% of each group.
Densities are normalized by the full group size and shown on a logarithmic scale.*
