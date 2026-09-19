"""
sparsity_analysis.py

Measure the counting sparsity (fraction of zero activations) of the ReLU²
feed-forward activations in a nanogpt checkpoint.

Ported from optimizer/modded-nanogpt/try_gpt.py: the ActivationSparsityTracker
and its wiring, without the distributed data generator or the sequence-length
sweep.
"""

import csv
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from dataloader import data_generator
from nanogpt import GPT
from eval_gpt import get_state_dict, infer_model_config


LOG_DIR = Path("logs/2026-07-04_00-06-23")
RESULTS_PATH = LOG_DIR / "sparsity.csv"
DATA_PATTERN = "data/fineweb10B/fineweb_val_*.bin"
DATA_ROOT = Path.cwd()
SEQ_LEN = 4096
SEQUENCES_PER_BATCH = 4
BATCH_SIZE = SEQUENCES_PER_BATCH * SEQ_LEN
EVAL_STEPS = 64
WARMUP_STEPS = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class ActivationSparsityTracker:
    """Accumulate the nonzero fraction of a layer's activation across batches."""

    def __init__(self, num_layers: int):
        self.num_layers = num_layers
        self.sparsity: dict[int, Tensor] = {}
        self.least_sparse: dict[int, Tensor] = {}
        self.counts: dict[int, int] = {}
        self.batch_sparsity: dict[int, Tensor] = {}
        self.least_sparse_batch_avg: Tensor | None = None

    @torch.no_grad()
    def update(self, layer_num: int, x: Tensor) -> None:
        ratio = ((x != 0).sum() / x.numel()).detach()
        if layer_num not in self.sparsity:
            self.sparsity[layer_num] = ratio
            self.least_sparse[layer_num] = ratio
            self.counts[layer_num] = 1
            self._update_batch(layer_num, ratio)
            return

        count = self.counts[layer_num]
        self.sparsity[layer_num] = self.sparsity[layer_num] + (ratio - self.sparsity[layer_num]) / (count + 1)
        self.least_sparse[layer_num] = torch.maximum(self.least_sparse[layer_num], ratio)
        self.counts[layer_num] = count + 1
        self._update_batch(layer_num, ratio)

    def _update_batch(self, layer_num: int, ratio: Tensor) -> None:
        self.batch_sparsity[layer_num] = ratio
        if len(self.batch_sparsity) != self.num_layers:
            return

        batch_avg = torch.stack([self.batch_sparsity[i] for i in range(self.num_layers)]).mean()
        if self.least_sparse_batch_avg is None:
            self.least_sparse_batch_avg = batch_avg
        else:
            self.least_sparse_batch_avg = torch.maximum(self.least_sparse_batch_avg, batch_avg)
        self.batch_sparsity.clear()

    def reset(self) -> None:
        self.sparsity.clear()
        self.least_sparse.clear()
        self.counts.clear()
        self.batch_sparsity.clear()
        self.least_sparse_batch_avg = None

    def as_floats(self) -> dict[int, float]:
        return {layer_num: value.item() for layer_num, value in sorted(self.sparsity.items())}

    def least_sparse_as_floats(self) -> dict[int, float]:
        return {layer_num: value.item() for layer_num, value in sorted(self.least_sparse.items())}

    def least_sparse_batch_avg_as_float(self) -> float:
        if self.least_sparse_batch_avg is None:
            return float("nan")
        return self.least_sparse_batch_avg.item()


def _make_hook(tracker: ActivationSparsityTracker, layer_num: int) -> Callable[..., None]:
    """Report the ReLU² activation sparsity produced by ``MLP.fc``."""

    def hook(module: nn.Module, inputs: tuple[Tensor, ...], output: Tensor) -> None:
        # ``MLP._forward_basic`` computes ``relu_(fc(x)).square()`` in place on
        # the tensor returned here, so recompute the same activation to measure
        # the sparsity of what the FFN actually stores.
        activation = output.relu().square()
        tracker.update(layer_num, activation)

    return hook


def attach_sparsity_tracker(model: GPT) -> tuple[ActivationSparsityTracker, list[Any]]:
    """Hook every block's MLP and return the tracker plus its removable handles."""
    tracker = ActivationSparsityTracker(len(model.blocks))
    handles = [
        block.mlp.fc.register_forward_hook(_make_hook(tracker, layer_num))
        for layer_num, block in enumerate(model.blocks)
    ]
    return tracker, handles


@torch.no_grad()
def measure_sparsity(
    model: GPT,
    tracker: ActivationSparsityTracker,
    seq_len: int,
    sequences_per_batch: int,
    eval_steps: int,
) -> None:
    """Run the model over FineWeb validation data and accumulate sparsity stats."""
    batch_size = sequences_per_batch * seq_len
    loader = data_generator(
        DATA_PATTERN,
        batch_size,
        seq_len=seq_len,
        device=DEVICE,
        data_root=DATA_ROOT,
    )

    model.eval()
    for _ in range(WARMUP_STEPS):
        inputs, targets = next(loader)
        model(inputs, targets)

    tracker.reset()
    for _ in range(eval_steps):
        inputs, targets = next(loader)
        model(inputs, targets)


def analyze_checkpoint(checkpoint_path: Path) -> dict[str, float | int | str]:
    print(f"checkpoint: {checkpoint_path}")

    state_dict = get_state_dict(checkpoint_path)
    vocab_size, num_layers, model_dim = infer_model_config(state_dict)
    model = GPT(
        vocab_size=vocab_size,
        num_layers=num_layers,
        model_dim=model_dim,
        cfg={"bitsparse": False, "pack_sbit": False, "checkpoint": False},
    )
    model.load_state_dict(state_dict)
    model.to(DEVICE)

    tracker, handles = attach_sparsity_tracker(model)
    measure_sparsity(model, tracker, SEQ_LEN, SEQUENCES_PER_BATCH, EVAL_STEPS)

    result: dict[str, float | int | str] = {
        "checkpoint": str(checkpoint_path),
        "step": int(checkpoint_path.stem),
        "seq_len": SEQ_LEN,
        "least_sparse_batch_avg": tracker.least_sparse_batch_avg_as_float(),
    }
    least_sparse = tracker.least_sparse_as_floats()
    for layer_num, sparsity in tracker.as_floats().items():
        result[f"layer_{layer_num}_avg_sparsity"] = sparsity
        result[f"layer_{layer_num}_least_sparse"] = least_sparse[layer_num]

    for handle in handles:
        handle.remove()
    del model
    if torch.device(DEVICE).type == "cuda":
        torch.cuda.empty_cache()

    avg = tracker.as_floats()
    least_sparse_batch_avg = float(result["least_sparse_batch_avg"])  # type: ignore[arg-type]
    print(
        f"  avg sparsity: {sum(avg.values()) / max(len(avg), 1):.4f}, "
        f"least sparse batch avg: {least_sparse_batch_avg:.4f}"
    )
    return result


def main() -> None:
    checkpoints = sorted(LOG_DIR.glob("*.pt"), key=lambda path: int(path.stem))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint files found in {LOG_DIR}")

    print(f"data: {DATA_ROOT / DATA_PATTERN}")
    print(f"checkpoints: {len(checkpoints)}")

    with RESULTS_PATH.open("w", newline="") as file:
        writer = None
        for checkpoint_path in checkpoints:
            result = analyze_checkpoint(checkpoint_path)
            if writer is None:
                writer = csv.DictWriter(file, fieldnames=result.keys())
                writer.writeheader()
            writer.writerow(result)
            file.flush()
    print(f"results: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
