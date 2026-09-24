"""Plot nanoGPT Muon momentum exponent history across recorded training steps."""
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

from plot_lib import (sample_group_colors, format_axes, WIDE_FONT_SCALE,
                      LEGEND_HANDLE_LENGTH, LEGEND_HANDLE_TEXT_PAD)
from plot_tables import render

CSV_PATH = None
STEPS = (1, 4, 8, 300, 1500, 3300)
CHECKPOINT_CSV = (Path(__file__).resolve().parents[1] / "artefacts/muon_momentum_checkpoints"
                  / "2026-09-23_12-04-32_428546_lct/exponents.csv")
RESULTS_DIR = Path(__file__).resolve().parents[1] / "artefacts" / "muon_momentum"
OUTPUT_DIR = Path(__file__).resolve().parent / "plots"


def load(path):
    path = Path(path)
    histograms, totals, zero_counts = {}, {}, {}
    with path.open(newline="") as file:
        for row in csv.DictReader(file):
            if not row["parameter"].endswith((".mlp.fc.weight", ".mlp.proj.weight")):
                continue
            key = (int(row["step"]), row["parameter"])
            histograms.setdefault(key, np.zeros(256, dtype=np.int64))[int(row["exponent"]) + 127] = int(row["count"])
            totals[key] = int(row["elements"])
            zero_counts[key] = zero_counts.get(key, 0) + int(row["zero_count"])
    if not histograms:
        raise ValueError(f"No feedforward weight or momentum histograms in {path}")
    steps = sorted({step for step, _ in histograms})
    means, zeros = [], []
    for step in steps:
        keys = [key for key in histograms if key[0] == step]
        assert all(histograms[key].sum() == totals[key] for key in keys)
        # Equal weight per tensor, not per element or matrix size.
        means.append(np.mean([histograms[key] / totals[key] for key in keys], axis=0))
        zeros.append(np.mean([zero_counts[key] / totals[key] for key in keys]))
    means, zeros = np.array(means), np.array(zeros)
    assert np.allclose(means.sum(axis=1), 1)
    assert np.all(means[:, 255] == 0) and np.allclose(means[:, 0], zeros)
    return steps, means, zeros


def build(data, ylabel="Mean probability per feedforward buffer", labels=None, min_exponent=None):
    steps, means, zeros = data
    exponents = np.arange(-126, 128)
    probabilities = means[:, 1:255]
    if min_exponent is not None:
        visible = exponents >= min_exponent
        exponents, probabilities = exponents[visible], probabilities[:, visible]
    occupied = np.any(probabilities > 0, axis=0)
    low, high = exponents[occupied][[0, -1]]
    zero_x = low - 5

    fig, ax = plt.subplots(figsize=(5.5, 5.2))
    for i, (step, color) in enumerate(zip(steps, sample_group_colors(len(steps)))):
        values = np.where(probabilities[i] > 0, probabilities[i], np.nan)
        zero_percent = 100 * zeros[i]
        zero_label = "~0%" if 0 < zero_percent < 0.01 else f"{zero_percent:.3g}%"
        ax.plot(exponents, values, color=color,
                linestyle="-" if i % 2 == 0 else "--",
                label=f"{labels[i] if labels is not None else f'Step {step}'}  ({zero_label} zero)")
        if zeros[i] > 0:
            ax.plot(zero_x, zeros[i], marker="o" if i % 2 == 0 else "s",
                    color=color, markerfacecolor="white", linestyle="none")
    ax.axvline(low - 2.5, color="#AAAAAA", linestyle=":", linewidth=.6)
    ax.set_yscale("log")
    positive = np.concatenate((probabilities[probabilities > 0], zeros[zeros > 0]))
    ax.set_ylim(positive.min() / 2, positive.max() * 1.5)
    ax.set_xlim(zero_x - 2, high + 1)
    ticks = MaxNLocator(nbins=4, integer=True).tick_values(low, high)
    ticks = ticks[(ticks >= low) & (ticks <= high)]
    ax.set_xticks([zero_x, *ticks], ["Zero", *[str(int(t)) for t in ticks]])
    format_axes(ax, xlabel="Exponent",
                ylabel="Mean probability", xformat=None)
    ax.grid(axis="x", visible=False)
    handles, legend_labels = ax.get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, 0),
               handlelength=LEGEND_HANDLE_LENGTH,
               handletextpad=LEGEND_HANDLE_TEXT_PAD,
               columnspacing=0.8, fontsize=16)
    fig.subplots_adjust(left=0.19, right=0.98, top=0.98, bottom=0.36)
    return fig, ax


def plot(path, filename="nanogpt_momentum_exponent_history.pdf", ylabel="Mean probability per feedforward buffer", checkpoint_path=None):
    data = load(path)
    if checkpoint_path is not None:
        later = load(checkpoint_path)
        data = (data[0] + later[0], np.concatenate((data[1], later[1])),
                np.concatenate((data[2], later[2])))
    indices = [data[0].index(step) for step in STEPS]
    data = (list(STEPS), data[1][indices], data[2][indices])
    render({filename: data}, lambda data: build(data, ylabel),
           output_dir=OUTPUT_DIR, font_scale=WIDE_FONT_SCALE, show=False)
    print(f"Saved {OUTPUT_DIR / filename}")


if __name__ == "__main__":
    plot(CSV_PATH or sorted(RESULTS_DIR.glob("*/exponents.csv"))[-1], checkpoint_path=CHECKPOINT_CSV)
