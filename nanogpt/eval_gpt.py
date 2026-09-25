"""
try_gpt.py
Load a checkpoint produced by train_gpt_simple.py and evaluate it on a slice of
the FineWeb validation data.
"""

import csv
import time
from pathlib import Path
import torch
from torch import Tensor

from dataloader import data_generator
from nanogpt import GPT


LOG_DIR = Path("logs/2026-07-04_00-06-23")
DATA_PATTERN = "data/fineweb10B/fineweb_val_*.bin"
DATA_ROOT = Path.cwd()
SEQUENCES_PER_BATCH = 4
EVAL_STEPS = 16
WARMUP_STEPS = 4


def setup_hooks(model: GPT) -> None:
    """Simulate an optimiser that releases gradients immediately, without updates."""
    def hook(parameter: Tensor) -> None:
        parameter.grad = None

    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.register_post_accumulate_grad_hook(hook)


def get_state_dict(checkpoint_path: Path) -> dict[str, Tensor]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        checkpoint = checkpoint["model"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    if not isinstance(checkpoint, dict) or not all(torch.is_tensor(value) for value in checkpoint.values()):
        raise TypeError(f"{checkpoint_path} does not look like a model state_dict")

    return {key.removeprefix("_orig_mod."): value for key, value in checkpoint.items()}


def infer_model_config(state_dict: dict[str, Tensor]) -> tuple[int, int, int]:
    vocab_size, model_dim = state_dict["embed.weight"].shape
    block_ids = {
        int(key.split(".", 2)[1])
        for key in state_dict
        if key.startswith("blocks.")
    }
    if len(block_ids) == 0:
        raise ValueError("Could not infer num_layers from checkpoint state_dict")
    return vocab_size, max(block_ids) + 1, model_dim


def evaluate(
    model: GPT,
    seq_len: int,
    sequences_per_batch: int,
    eval_steps: int,
) -> tuple[float, float, int]:

    batch_size = sequences_per_batch * seq_len
    eval_tokens = batch_size * eval_steps
    loader = data_generator(
        DATA_PATTERN,
        batch_size,
        seq_len=seq_len,
        device="cuda",
        data_root=DATA_ROOT,
    )

    model.eval()
    for _ in range(WARMUP_STEPS):
        inputs, targets = next(loader)
        l = model(inputs, targets)
        l.backward()
        del l
        model.zero_grad()
    torch.cuda.synchronize()

    loss_sum = torch.zeros((), device="cuda")
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(eval_steps):
        model.zero_grad()
        inputs, targets = next(loader)
        loss_step = model(inputs, targets)
        loss_step.backward()
        loss_sum += loss_step.detach()
    torch.cuda.synchronize()
    peak_memory = torch.cuda.max_memory_allocated() // 1024**2

    elapsed = time.perf_counter() - t0

    loss = loss_sum.item() / eval_tokens
    return loss, elapsed / eval_steps, peak_memory


def evaluate_checkpoint(
    checkpoint_path: Path,
    seq_len: int,
    sequences_per_batch: int,
    eval_steps: int,
    cfg_dict: dict[str, bool],
) -> dict[str, float | int]:
    print(f"checkpoint: {checkpoint_path}")

    state_dict = get_state_dict(checkpoint_path)
    vocab_size, num_layers, model_dim = infer_model_config(state_dict)
    model = GPT(
        vocab_size=vocab_size,
        num_layers=num_layers,
        model_dim=model_dim, cfg=cfg_dict,
    )
    model.load_state_dict(state_dict)
    model.cuda()
    setup_hooks(model)

    loss, avg_time, peak_memory = evaluate(
        model,
        seq_len,
        sequences_per_batch,
        eval_steps,
    )

    result: dict[str, float | int] = {
        "step": int(checkpoint_path.stem),
        "seq_len": seq_len,
        "loss": loss,
        "avg_time": avg_time,
        "peak_memory": peak_memory,
    }

    print(f"loss: {loss:.5f}, seq_len: {seq_len}, avg_time: {avg_time:.4f}s, peak_memory: {peak_memory} MB")
    del model
    torch.cuda.empty_cache()
    return result


def main() -> None:
    checkpoint = Path("~/Documents/bitsparse/nanogpt/logs/2026-07-04_00-06-23/3300.pt").expanduser()
    print(f"data: {DATA_ROOT / DATA_PATTERN}")

    # configs = [ {"bitsparse": True, "pack_sbit": False, "checkpoint": False},
    #             {"bitsparse": True, "pack_sbit": True, "checkpoint": False},
    #             {"bitsparse": False, "pack_sbit": False, "checkpoint": True},
    #             ]
    configs=[{"bitsparse": False, "pack_sbit": False, "checkpoint": False}]
    for cfg_dict in configs:
        RESULTS_PATH = LOG_DIR / f"bitsparse_{cfg_dict['bitsparse']}_sbit_{cfg_dict['pack_sbit']}_ckpt_{cfg_dict['checkpoint']}.csv"
        results_file_exists = RESULTS_PATH.exists()
        print(f"results: {RESULTS_PATH}")
        #
        sequence_lengths = [256, 512, 1024, 2048, 4096, 8192, 12000, 16000]
        with open(RESULTS_PATH, "a", newline="") as file:    # RESULTS_PATH.open("a", newline="")
            writer = None
            for seq_len in sequence_lengths:
                result = evaluate_checkpoint(
                    checkpoint,
                    seq_len,
                    SEQUENCES_PER_BATCH,
                    EVAL_STEPS, cfg_dict
                )
                if writer is None:
                    writer = csv.DictWriter(file, fieldnames=result.keys())
                    if not results_file_exists:
                        writer.writeheader()
                writer.writerow(result)
                file.flush()


if __name__ == "__main__":
    main()
