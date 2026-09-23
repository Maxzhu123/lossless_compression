"""Plot exact BF16 exponent probabilities for the largest Nemotron weight groups."""
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import torch

from plot_lib import plot_style
from plot_weight_distributions import COLORS

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'artefacts/weight_exponent_distribution_results.pt'


def plot_exponents(results=RESULTS, output=Path(__file__).resolve().parent/'plots/weight_exponent_distributions',
                   min_elements=500_000_000, *, group_kind='weights'):
    data = torch.load(results, map_location='cpu', weights_only=True)
    if data.get('format_version') != 1:
        raise ValueError('Expected exponent histogram format 1')
    groups, lower, upper = [], -20, 2
    for label in data['categories']:
        counts = data['exponent_counts'][label].numpy()
        if counts.shape != (256,) or (counts < 0).any():
            raise ValueError(f'Invalid exponent counts for {label}')
        total = int(counts.sum())
        if total <= min_elements:
            continue
        if counts[255]:
            raise ValueError(f'{label} contains nonfinite values; add a separate special-value panel')
        normal = counts[1:255]
        normal_total = int(normal.sum())
        if normal_total == 0:
            raise ValueError(f'{label} has no normal {group_kind}')
        upper = max(upper, int(np.flatnonzero(normal)[-1])-126)
        groups.append((label, counts, total))
    if not groups:
        raise ValueError('No groups exceed the element threshold')
    exponents = np.arange(-126, 128)
    normal_ticks = list(range(lower, upper+1, 5))
    tail_x = lower-5
    visible = (exponents >= lower) & (exponents <= upper)
    columns, rows = 2, math.ceil(len(groups)/2)
    ymax = max(float(counts.max()/total) for _, counts, total in groups)
    ymax = math.ceil(ymax/0.05)*0.05
    with plot_style(wide=True, font_scale=1.1,
                    overrides={'figure.figsize': (8, 2.3*rows+0.4), 'axes.grid': False}):
        fig, axes = plt.subplots(rows, columns, sharex=True, sharey=True, squeeze=False)
        for index, (label, counts, total) in enumerate(groups):
            ax = axes.flat[index]
            color = COLORS[index % len(COLORS)]
            probabilities = counts/total
            ax.bar(exponents[visible], probabilities[1:255][visible], width=0.82, color=color, linewidth=0)
            left_mass = float(probabilities[:lower+127].sum())
            ax.bar(tail_x, left_mass, width=0.82, color=color, hatch='//', edgecolor='#333333', linewidth=0.5)
            ax.axvline(lower-2.5, color='#AAAAAA', linewidth=0.6, linestyle=':')
            ax.set_xlim(tail_x-2, upper+1)
            ax.set_ylim(0, ymax)
            ax.set_xticks([tail_x]+normal_ticks, [f'<{lower}']+[str(value) for value in normal_ticks])
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.grid(axis='y', color='#D9D9D9', linewidth=0.5)
            ax.set_title(f'({chr(97+index)}) {label}', loc='left', fontsize=12.5, pad=10)
            if index % columns == 0:
                ax.set_ylabel('Probability')
            if index + columns >= len(groups):
                ax.tick_params(axis='x', labelbottom=True)
                ax.set_xlabel('Exponent (bias removed)')
            assert np.isclose(left_mass+probabilities[1:255][visible].sum(), 1)
        for index in range(len(groups), rows*columns):
            axes.flat[index].set_visible(False)
        fig.tight_layout(pad=0.7, h_pad=1.5, w_pad=1.1)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output.with_suffix('.pdf'))
        plt.close(fig)
    print(f'Saved {output.with_suffix(".pdf")}')


def main():
    # Edit these settings before running.
    results = RESULTS
    output = Path(__file__).resolve().parent / 'plots/weight_exponent_distributions'
    min_elements = 500_000_000
    plot_exponents(results, output, min_elements)


if __name__ == '__main__':
    main()
