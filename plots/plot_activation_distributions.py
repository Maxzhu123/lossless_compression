"""Plot the largest activation groups in the same format as weight distributions.

Run from the repository root: python plots/plot_activation_distributions.py
"""
from pathlib import Path

from plot_weight_distributions import plot_distributions

ROOT = Path(__file__).resolve().parents[1]


def main():
    # Edit these settings before running.
    results = ROOT / 'artefacts/activation_distribution_results.pt'
    output = Path(__file__).resolve().parent / 'plots/activation_distributions'
    min_elements = 500_000_000  # Aggregated observations over all layers and batches.
    coverage = 0.999999
    return plot_distributions(
        results, output, min_elements, coverage,
        group_kind='activation', value_label='Activation value',
    )


if __name__ == '__main__':
    main()
