"""Plot exact BF16 exponents for the same largest activation groups.

Run from the repository root: python plots/plot_activation_exponents.py
"""
from pathlib import Path

from plot_weight_exponents import plot_exponents

ROOT = Path(__file__).resolve().parents[1]


def main():
    # Edit these settings before running.
    results = ROOT / 'artefacts/activation_exponent_distribution_results.pt'
    output = Path(__file__).resolve().parent / 'plots/activation_exponent_distributions'
    min_elements = 500_000_000
    return plot_exponents(results, output, min_elements, group_kind='activations')


if __name__ == '__main__':
    main()
