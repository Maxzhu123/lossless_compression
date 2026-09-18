"""Paper figure of Nemotron weight groups with more than 500M parameters.

Reads saved CPU histograms only. Export PDF/PNG plus a selection/provenance JSON.
Run from the repository root: python -m plots.plot_weight_distributions
"""
import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, LogLocator, NullFormatter
import numpy as np
import torch

if __package__:
    from .plot_lib import plot_style
else:
    from plot_lib import plot_style

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / 'artefacts/weight_distribution_results.pt'
STREAMED_RESULTS = ROOT / 'artefacts/weight_distribution_large_results.pt'
COLORS = ('#0072B2', '#D55E00', '#009E73', '#CC79A7', '#333333', '#56B4E9')


def load_groups(path, min_elements):
    data = torch.load(path, map_location='cpu', weights_only=True)
    if data.get('format_version') != 2:
        raise ValueError('Expected weight histogram format version 2')
    edges = data['edges'].double().numpy()
    if not np.isfinite(edges).all() or not (np.diff(edges) > 0).all():
        raise ValueError('Histogram edges must be finite and increasing')
    groups = []
    for label in data['categories']:
        counts = data['histograms'][label].numpy()
        if len(counts) != len(edges)+1 or (counts < 0).any():
            raise ValueError(f'Invalid histogram for {label}')
        total = int(counts.sum())  # Includes both overflow buckets.
        if total > min_elements:
            groups.append((label, counts, total))
    if not groups:
        raise ValueError(f'No weight groups contain more than {min_elements:,} values')
    return data, edges, groups


def shared_limit(edges, groups, coverage):
    # Pick bin boundaries so the displayed window contains whole histogram bins.
    boundaries = np.maximum(np.abs(edges[:-1]), np.abs(edges[1:]))
    order = np.argsort(boundaries)
    limit = 0.0
    for label, counts, total in groups:
        cumulative = np.cumsum(counts[1:-1][order], dtype=np.int64)
        if cumulative[-1] < coverage*total:
            raise ValueError(f'{label}: overflow exceeds display tolerance; collect wider histograms')
        index = np.searchsorted(cumulative, coverage*total)
        limit = max(limit, float(boundaries[order[index]]))
    return limit


def plot_distributions(path, output, min_elements=500_000_000, coverage=0.999999):
    if not 0 < coverage <= 1:
        raise ValueError('coverage must be in (0,1]')
    data, edges, groups = load_groups(path, min_elements)
    limit = shared_limit(edges, groups, coverage)
    widths = np.diff(edges)
    centers = (edges[:-1]+edges[1:])/2
    visible = (edges[:-1] >= -limit) & (edges[1:] <= limit)
    densities = [counts[1:-1]/(total*widths) for _, counts, total in groups]
    positive = np.concatenate([d[visible & (d > 0)] for d in densities])
    ymin = 10 ** math.floor(math.log10(float(positive.min())))
    ymax = 10 ** math.ceil(math.log10(float(positive.max())))
    columns, rows = 2, math.ceil(len(groups)/2)
    summary = {'source': str(path.resolve()), 'model_name': data['model_name'],
               'selection': f'group count > {min_elements}', 'normalization': 'count / (all group values * bin width)',
               'xlim': [-limit, limit], 'coverage_target': coverage,
               'y_scale': 'logarithmic', 'groups': []}
    output.parent.mkdir(parents=True, exist_ok=True)
    with plot_style(wide=True, font_scale=1.1,
                    overrides={'figure.figsize': (8, 2.3*rows+0.4), 'axes.grid': False}):
        fig, axes = plt.subplots(rows, columns, sharex=True, sharey=True, squeeze=False)
        for index, ((label, counts, total), density) in enumerate(zip(groups, densities)):
            ax = axes.flat[index]
            color = COLORS[index % len(COLORS)]
            # Zero-count bins remain gaps on the log axis; do not add pseudocounts.
            values = np.where(density > 0, density, np.nan)
            ax.stairs(values, edges, color=color, linewidth=1.5)
            ax.set_yscale('log')
            ax.set_ylim(ymin, ymax)
            ax.set_xlim(-limit, limit)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=5, symmetric=True))
            ax.yaxis.set_major_locator(LogLocator(base=10, numticks=5))
            ax.yaxis.set_minor_formatter(NullFormatter())
            ax.grid(axis='y', which='major', color='#D9D9D9', linewidth=0.5)
            ax.axvline(0, color='#999999', linewidth=0.6, linestyle=':', zorder=0)
            ax.set_title(f'({chr(97+index)}) {label}', loc='left', fontsize=12.5, pad=10)
            if index % columns == 0:
                ax.set_ylabel('Probability density')
            if index // columns == rows-1:
                ax.set_xlabel('Weight value')
            displayed = int(counts[1:-1][visible].sum())
            assert displayed/total >= coverage-1e-12
            summary['groups'].append({'label': label, 'numel': total,
                'displayed_fraction': displayed/total,
                'histogram_overflow': int(counts[0]+counts[-1]),
                'minimum': float(data['extrema'][label][0]),
                'maximum': float(data['extrema'][label][1])})
        for index in range(len(groups), rows*columns):
            axes.flat[index].set_visible(False)
        fig.tight_layout(pad=0.7, h_pad=1.5, w_pad=1.1)
        fig.savefig(output.with_suffix('.pdf'))
        fig.savefig(output.with_suffix('.png'), dpi=300)
        plt.close(fig)
    output.with_suffix('.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))
    print(f'Saved {output.with_suffix(".pdf")}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent/'weight_distributions')
    parser.add_argument('--min-elements', type=int, default=500_000_000)
    parser.add_argument('--coverage', type=float, default=0.999999)
    args = parser.parse_args()
    results = args.results or (DEFAULT_RESULTS if DEFAULT_RESULTS.exists() else STREAMED_RESULTS)
    if not results.exists():
        parser.error(f'No saved histograms at {results}; run python -m plots.collect_weight_histograms first')
    plot_distributions(results, args.output, args.min_elements, args.coverage)


if __name__ == '__main__':
    main()
