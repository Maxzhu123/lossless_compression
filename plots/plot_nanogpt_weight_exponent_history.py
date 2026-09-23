"""Plot nanoGPT weight exponent history across recorded steps and saved checkpoints."""
from pathlib import Path

import numpy as np

from plot_nanogpt_momentum_exponent_history import load, build, OUTPUT_DIR
from plot_tables import render

CSV_PATH = None
EARLY_STEPS = (1, 4, 8)
RESULTS_DIR = Path(__file__).resolve().parents[1] / "artefacts" / "nanogpt_weights"
CHECKPOINT_CSV = (Path(__file__).resolve().parents[1] / "artefacts" /
                  "nanogpt_weight_checkpoints/2026-07-04_00-06-23/exponents.csv")


def build_weights(data):
    early, later = data
    indices = [early[0].index(step) for step in EARLY_STEPS]
    steps = list(EARLY_STEPS) + later[0]
    means = np.concatenate((early[1][indices], later[1]))
    zeros = np.concatenate((early[2][indices], later[2]))
    fig, ax = build((steps, means, zeros), ylabel="Mean probability per weight matrix", min_exponent=-20)
    ax.set_ylim(bottom=1e-6)
    return fig, ax


if __name__ == "__main__":
    path = CSV_PATH or sorted(RESULTS_DIR.glob("*/exponents.csv"))[-1]
    render({"nanogpt_weight_exponent_history.pdf": (load(path), load(CHECKPOINT_CSV))},
           build_weights, output_dir=OUTPUT_DIR, wide=True, show=False)
    print(f"Saved {OUTPUT_DIR / 'nanogpt_weight_exponent_history.pdf'}")
