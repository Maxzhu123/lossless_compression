"""Plot exact BF16 exponent probabilities for the largest Nemotron weight groups."""
import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import torch

if __package__:
    from .plot_lib import plot_style
    from .plot_weight_distributions import COLORS
else:
    from plot_lib import plot_style
    from plot_weight_distributions import COLORS

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'artefacts/weight_exponent_distribution_results.pt'


def shortest_interval(counts, coverage=0.99):
    """Shortest contiguous normal-exponent interval reaching the requested mass."""
    target = coverage*counts.sum()
    cumulative = np.concatenate(([0], np.cumsum(counts, dtype=np.int64)))
    candidates = []
    for left in range(len(counts)):
        right = int(np.searchsorted(cumulative, cumulative[left]+target, side='left'))
        if right <= len(counts):
            candidates.append((right-left, left-126, right-1-126))
    return min(candidates)


def plot_exponents(results=RESULTS, output=Path(__file__).resolve().parent/'plots/weight_exponent_distributions',
                   min_elements=500_000_000):
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
            raise ValueError(f'{label} has no normal weights')
        upper = max(upper, int(np.flatnonzero(normal)[-1])-126)
        groups.append((label, counts, total))
    if not groups:
        raise ValueError('No groups exceed the parameter threshold')
    exponents = np.arange(-126, 128)
    normal_ticks = list(range(lower, upper+1, 5))
    tail_x = lower-5
    visible = (exponents >= lower) & (exponents <= upper)
    columns, rows = 2, math.ceil(len(groups)/2)
    ymax = max(float(counts.max()/total) for _, counts, total in groups)
    ymax = math.ceil(ymax/0.05)*0.05
    summary = {'source': str(results.resolve()), 'model_name': data['model_name'],
               'normal_exponent_xlim': [lower, upper], 'probability_denominator': 'all weights in each group',
               'left_bin': 'all exponent fields below the displayed normal range; includes field 0', 'groups': []}
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
            if index // columns == rows-1:
                ax.set_xlabel('Exponent (bias removed)')
            positive = probabilities[probabilities > 0]
            entropy = float(-(positive*np.log2(positive)).sum())
            width, lo, hi = shortest_interval(counts[1:255])
            mass99 = int(counts[lo+127:hi+128].sum())/int(counts[1:255].sum())
            assert mass99 >= 0.99-1e-12
            normal_visible = int(counts[1:255][visible].sum())/int(counts[1:255].sum())
            assert np.isclose(left_mass+probabilities[1:255][visible].sum(), 1)
            top_bins = int(np.searchsorted(np.cumsum(np.sort(counts)[::-1]), .99*total))+1
            summary['groups'].append({'label': label, 'numel': total,
                'entropy_bits': entropy, 'symbols_for_99pct': top_bins,
                'left_aggregate_fraction': left_mass, 'normal_99pct_width': width, 'normal_99pct_interval': [lo, hi],
                'normal_interval_mass': mass99, 'mode_exponent': int(np.argmax(counts[1:255]))-126,
                'zero_fraction': data['zero_counts'][label]/total,
                'subnormal_fraction': (int(counts[0])-data['zero_counts'][label])/total,
                'visible_normal_fraction': normal_visible})
        for index in range(len(groups), rows*columns):
            axes.flat[index].set_visible(False)
        fig.tight_layout(pad=0.7, h_pad=1.5, w_pad=1.1)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output.with_suffix('.pdf'))
        plt.close(fig)
    print(f'Saved {output.with_suffix(".pdf")}')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=RESULTS)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent/'plots/weight_exponent_distributions')
    args = parser.parse_args()
    plot_exponents(args.results, args.output)


if __name__ == '__main__':
    main()
