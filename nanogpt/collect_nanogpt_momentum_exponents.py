"""Collect actual saved Muon momentum from selected LCT training checkpoints."""
from pathlib import Path

from collect_nanogpt_weight_exponents import collect

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = ROOT / "nanogpt/logs/2026-09-23_12-04-32_428546_lct"
STEPS = (300, 1500, 3300)


if __name__ == "__main__":
    if CHECKPOINT_DIR is None:
        raise ValueError("Set CHECKPOINT_DIR to an LCT training run with saved Muon momentum")
    checkpoint_dir = Path(CHECKPOINT_DIR)
    collect(checkpoint_dir, STEPS,
            ROOT / "artefacts/muon_momentum_checkpoints" / checkpoint_dir.name,
            kind="momentum")
