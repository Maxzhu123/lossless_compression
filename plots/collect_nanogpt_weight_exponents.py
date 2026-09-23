"""Collect BF16-view weight exponents from selected saved nanoGPT checkpoints."""
import csv
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = ROOT / "nanogpt/logs/2026-07-04_00-06-23"
STEPS = (300, 1500, 3300)
OUTPUT_DIR = ROOT / "artefacts/nanogpt_weight_checkpoints" / CHECKPOINT_DIR.name


def main():
    torch.set_num_threads(4)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / "exponents.csv"
    sources = []
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("step", "parameter", "exponent", "count", "elements", "zero_count"))
        for step in STEPS:
            checkpoint = CHECKPOINT_DIR / f"{step}.pt"
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            matrices = 0
            for name, weight in state.items():
                if not name.endswith(".weight") or weight.ndim != 2:
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
            file.flush()
            sources.append(dict(step=step, checkpoint=str(checkpoint), matrices=matrices))
            print(f"Recorded checkpoint {step}: {matrices} matrices", flush=True)
            del state, weight, bits
    metadata = dict(sources=sources, dtype="BF16 view of checkpoint weights",
                    selection="All 2D weight matrices, including embedding and LM head",
                    counts="Exact counts of all elements; no sampling",
                    comparison="Later checkpoints are a different training run from the early-step recordings")
    (OUTPUT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved {path}", flush=True)


if __name__ == "__main__":
    main()
