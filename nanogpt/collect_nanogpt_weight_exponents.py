"""Collect BF16-view weight exponents from selected saved nanoGPT checkpoints."""
import csv
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = ROOT / "nanogpt/logs/2026-07-04_00-06-23"
STEPS = (300, 1500, 3300)
OUTPUT_DIR = ROOT / "artefacts/nanogpt_weight_checkpoints" / CHECKPOINT_DIR.name


def collect(checkpoint_dir=CHECKPOINT_DIR, steps=STEPS, output_dir=OUTPUT_DIR, kind="weights"):
    torch.set_num_threads(4)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "exponents.csv"
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("step", "parameter", "exponent", "count", "elements", "zero_count"))
        for step in steps:
            checkpoint = Path(checkpoint_dir) / f"{step}.pt"
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            if kind == "momentum":
                if "muon_momentum" not in state:
                    raise ValueError(f"{checkpoint} has no saved Muon momentum")
                state = state["muon_momentum"]
            else:
                state = state.get("model", state)
            matrices = 0
            for name, weight in state.items():
                if weight is None:  # Momentum has not been initialized at step zero.
                    continue
                if not name.endswith((".mlp.fc.weight", ".mlp.proj.weight")) or weight.ndim != 2:
                    continue
                # The old trainer stores FP32 matrices but casts them to BF16 for forward.
                bits = weight.to(torch.bfloat16).contiguous().view(torch.int16).flatten()
                counts = torch.bincount(((bits >> 7) & 255).long(), minlength=256).tolist()
                zeros = int(((bits & 0x7fff) == 0).sum())
                assert sum(counts) == weight.numel() and counts[255] == 0
                for byte, count in enumerate(counts):
                    writer.writerow((step, name, byte - 127, count, weight.numel(),
                                     zeros if byte == 0 else 0))
                matrices += 1
                del bits
            file.flush()
            print(f"Recorded checkpoint {step}: {matrices} matrices", flush=True)
            del state
    print(f"Saved {path}", flush=True)


if __name__ == "__main__":
    collect()
