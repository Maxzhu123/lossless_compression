"""Collect activations and save CPU histogram tensors for offline plotting."""
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from torch import Tensor, nn
from tqdm import trange
from transformers import AutoTokenizer
from transformers.models.nemotron_h.modeling_nemotron_h import (
    NemotronHForCausalLM,
    NemotronHAttention,
    NemotronHMLP,
    NemotronHMamba2Mixer,
)

from histogram import tensor_histogram


MODEL_NAME = "nvidia/Nemotron-H-8B-Base-8K"
ARTEFACTS_PATH = Path(__file__).resolve().parents[1] / "artefacts"
MODEL_PATH = ARTEFACTS_PATH / "Nemotron-H-8B-Base-8K"
SAMPLE_TEXT_PATH = Path(__file__).parent / "sample_text.txt"
RESULTS_PATH = ARTEFACTS_PATH / "activation_distribution_results.pt"
EXPONENT_RESULTS_PATH = ARTEFACTS_PATH / "activation_exponent_distribution_results.pt"
RESULTS_FORMAT_VERSION = 2  # Shared histogram schema used by weight_distribution.py.
NUM_BATCHES = 4
SEQUENCE_LENGTH = 1024
SEQUENCES_PER_BATCH = 1
BIN_WIDTH = 0.1
LIMIT = 1000.0  # More extreme values remain in the two overflow buckets.


@dataclass(frozen=True)
class ActivationCategory:
    """A semantic activation capture point relative to a parent module."""

    parent_type: type[nn.Module]
    module_path: str = ""
    capture: Literal["input", "output"] = "output"

    def __post_init__(self) -> None:
        if self.capture not in ("input", "output"):
            raise ValueError("capture must be 'input' or 'output'")

    def resolve(self, parent: nn.Module) -> nn.Module:
        return parent.get_submodule(self.module_path) if self.module_path else parent


ACTIVATION_CATEGORIES: dict[str, ActivationCategory] = {
    "Token embeddings": ActivationCategory(
        NemotronHForCausalLM,
        "model.embeddings",
    ),
    "FFN input": ActivationCategory(NemotronHMLP, capture="input"),
    "FFN pre-activation": ActivationCategory(NemotronHMLP, "up_proj"),
    # down_proj receives act_fn(up_proj(x)), so its input is the FFN hidden state.
    "FFN hidden": ActivationCategory(NemotronHMLP, "down_proj", "input"),
    "FFN output": ActivationCategory(NemotronHMLP),
    "Attention input": ActivationCategory(NemotronHAttention, capture="input"),
    "Attention query": ActivationCategory(NemotronHAttention, "q_proj"),
    "Attention key": ActivationCategory(NemotronHAttention, "k_proj"),
    "Attention value": ActivationCategory(NemotronHAttention, "v_proj"),
    # o_proj receives the concatenated per-head attention result.
    "Attention context": ActivationCategory(NemotronHAttention, "o_proj", "input"),
    "Attention output": ActivationCategory(NemotronHAttention),
    "Mamba input": ActivationCategory(NemotronHMamba2Mixer, capture="input"),
    "Mamba projected": ActivationCategory(NemotronHMamba2Mixer, "in_proj"),
    # out_proj receives the gated, normalized SSM result.
    "Mamba hidden": ActivationCategory(NemotronHMamba2Mixer, "out_proj", "input"),
    "Mamba output": ActivationCategory(NemotronHMamba2Mixer),
    "Final hidden": ActivationCategory(
        NemotronHForCausalLM,
        "model.norm_f",
    ),
    "Logits": ActivationCategory(NemotronHForCausalLM, "lm_head"),
}


def _first_tensor(value: Any) -> Tensor | None:
    """Find the first tensor in a module output, including tuple outputs."""
    if torch.is_tensor(value):
        return value
    if isinstance(value, Mapping):
        for item in value.values():
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    elif isinstance(value, (tuple, list)):
        for item in value:
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    return None


@torch.inference_mode()
def activation_distribution(
    model: nn.Module,
    batches: Iterable[Mapping[str, Tensor] | Tensor],
    categories: Mapping[str, ActivationCategory],
    *,
    num_batches: int,
    bin_width: float = BIN_WIDTH,
    limit: float = LIMIT,
    exponent_counts: dict[str, Tensor] | None = None,
) -> tuple[
    dict[str, Tensor],
    Tensor,
    dict[str, tuple[Tensor, Tensor]],
    dict[str, Tensor],
]:
    """Aggregate histograms and zeros; optionally fill exact BF16 exponent counts."""
    if num_batches <= 0:
        raise ValueError("num_batches must be positive")
    if not categories:
        raise ValueError("categories must not be empty")

    counts: dict[str, Tensor | None] = {label: None for label in categories}
    zero_counts: dict[str, Tensor | None] = {label: None for label in categories}
    minima: dict[str, Tensor | None] = {label: None for label in categories}
    maxima: dict[str, Tensor | None] = {label: None for label in categories}
    edges: Tensor | None = None
    modules = list(model.modules())

    category_modules: dict[str, list[nn.Module]] = {}
    for label, category in categories.items():
        targets = []
        seen: set[int] = set()
        for parent in modules:
            if not isinstance(parent, category.parent_type):
                continue
            target = category.resolve(parent)
            if id(target) not in seen:
                targets.append(target)
                seen.add(id(target))
        category_modules[label] = targets

    unmatched_labels = [
        label for label, targets in category_modules.items() if not targets
    ]
    if unmatched_labels:
        raise ValueError(f"Model contains no capture points for {unmatched_labels}")

    def record_activation(label: str, module: nn.Module, value: Any) -> None:
        nonlocal edges
        tensor = _first_tensor(value)
        if tensor is None:
            raise TypeError(
                f"Expected {type(module).__name__} activation to contain a tensor"
            )
        batch_counts, batch_edges, batch_minimum, batch_maximum = tensor_histogram(
            [tensor],
            bin_width=bin_width,
            limit=limit,
        )
        batch_zero_count = tensor.eq(0).sum(dtype=torch.int64)
        if exponent_counts is not None:
            if tensor.dtype != torch.bfloat16:
                raise ValueError("Exact BF16 exponent collection requires BF16 activations")
            bits = tensor.detach().contiguous().view(torch.int16).reshape(-1)
            exponent_histogram = torch.zeros(256, dtype=torch.int64, device=tensor.device)
            for start in range(0, bits.numel(), 1024 ** 2):
                fields = ((bits[start:start + 1024 ** 2] >> 7) & 255).long()
                exponent_histogram += torch.bincount(fields, minlength=256)
            exponent_counts[label] = exponent_counts.get(label, 0) + exponent_histogram
        counts[label] = (
            batch_counts
            if counts[label] is None
            else counts[label] + batch_counts
        )
        zero_counts[label] = (
            batch_zero_count
            if zero_counts[label] is None
            else zero_counts[label] + batch_zero_count
        )
        minima[label] = (
            batch_minimum
            if minima[label] is None
            else torch.minimum(minima[label], batch_minimum)
        )
        maxima[label] = (
            batch_maximum
            if maxima[label] is None
            else torch.maximum(maxima[label], batch_maximum)
        )
        edges = batch_edges

    def output_hook(label: str):
        def capture_output(
            module: nn.Module,
            inputs: tuple[Any, ...],
            output: Any,
        ) -> None:
            record_activation(label, module, output)

        return capture_output

    def input_hook(label: str):
        def capture_input(
            module: nn.Module,
            inputs: tuple[Any, ...],
            kwargs: dict[str, Any],
        ) -> None:
            tensor = _first_tensor(inputs)
            record_activation(label, module, tensor if tensor is not None else kwargs)

        return capture_input

    handles = []
    for label, category in categories.items():
        for module in category_modules[label]:
            if category.capture == "input":
                handles.append(
                    module.register_forward_pre_hook(input_hook(label), with_kwargs=True)
                )
            else:
                handles.append(module.register_forward_hook(output_hook(label)))

    model.eval()
    try:
        iterator = iter(batches)
        for _ in trange(num_batches, desc="Collecting activations", unit="batch"):
            batch = next(iterator)
            if isinstance(batch, Mapping):
                model(**batch, use_cache=False)
            else:
                model(batch, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()

    if (
        edges is None
        or any(value is None for value in counts.values())
        or any(value is None for value in zero_counts.values())
        or any(value is None for value in minima.values())
        or any(value is None for value in maxima.values())
    ):
        raise RuntimeError("One or more activation hooks did not capture any outputs")

    histograms = {label: value for label, value in counts.items() if value is not None}
    exact_zero_counts = {
        label: value for label, value in zero_counts.items() if value is not None
    }
    extrema = {
        label: (minima[label], maxima[label])
        for label in categories
        if minima[label] is not None and maxima[label] is not None
    }
    return histograms, edges, extrema, exact_zero_counts


def save_activation_results(
    path: Path,
    histograms: Mapping[str, Tensor],
    edges: Tensor,
    extrema: Mapping[str, tuple[Tensor, Tensor]],
    zero_counts: Mapping[str, Tensor],
    categories: Mapping[str, ActivationCategory],
    *,
    model_name: str,
    num_batches: int,
    sequence_length: int,
    sequences_per_batch: int,
    bin_width: float,
    limit: float,
    model_dtype: str | None = None,
) -> None:
    """Save the weight-compatible histogram schema plus activation-specific metadata."""
    if any(set(values) != set(histograms) for values in (zero_counts, extrema, categories)):
        raise ValueError("histograms, zero_counts, extrema, and categories must have matching labels")
    result = {
        "format_version": RESULTS_FORMAT_VERSION,
        "model_name": model_name,
        "model_dtype": model_dtype,
        "analysis_device": str(edges.device),
        "num_batches": num_batches,
        "sequence_length": sequence_length,
        "sequences_per_batch": sequences_per_batch,
        "bin_width": bin_width,
        "limit": limit,
        "categories": list(histograms),
        "capture_points": {
            label: {
                "parent_type": category.parent_type.__name__,
                "module_path": category.module_path,
                "capture": category.capture,
            }
            for label, category in categories.items()
        },
        "histograms": {
            label: counts.detach().cpu()
            for label, counts in histograms.items()
        },
        "zero_counts": {
            label: count.detach().cpu()
            for label, count in zero_counts.items()
        },
        "edges": edges.detach().cpu(),
        "extrema": {
            label: (minimum.detach().cpu(), maximum.detach().cpu())
            for label, (minimum, maximum) in extrema.items()
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, path)


def token_batches(
    tokenizer,
    text: str,
    *,
    num_batches: int,
    sequence_length: int,
    sequences_per_batch: int,
    device: torch.device,
) -> Iterable[dict[str, Tensor]]:
    """Tokenize text and yield fixed-size, non-overlapping input batches."""
    tokens = tokenizer(text, return_tensors="pt", add_special_tokens=False)[
        "input_ids"
    ].squeeze(0)
    batch_tokens = sequence_length * sequences_per_batch
    required_tokens = num_batches * batch_tokens
    if tokens.numel() < required_tokens:
        raise ValueError(
            f"Sample text contains {tokens.numel()} tokens; "
            f"{required_tokens} are required"
        )

    for start in range(0, required_tokens, batch_tokens):
        input_ids = tokens[start : start + batch_tokens].reshape(
            sequences_per_batch, sequence_length
        )
        yield {
            "input_ids": input_ids.to(device),
            "attention_mask": torch.ones_like(input_ids, device=device),
        }


def main() -> None:
    if not (MODEL_PATH / "config.json").is_file():
        raise FileNotFoundError(f"Local Nemotron checkpoint not found at {MODEL_PATH}")
    if not SAMPLE_TEXT_PATH.is_file():
        raise FileNotFoundError(
            f"Provide sample text at {SAMPLE_TEXT_PATH} before collecting activations"
        )
    text = SAMPLE_TEXT_PATH.read_text(encoding="utf-8")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    model = NemotronHForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=dtype,
        local_files_only=True,
    ).to(device)

    batches = token_batches(
        tokenizer,
        text,
        num_batches=NUM_BATCHES,
        sequence_length=SEQUENCE_LENGTH,
        sequences_per_batch=SEQUENCES_PER_BATCH,
        device=device,
    )
    exponent_counts = {} if dtype == torch.bfloat16 else None
    histograms, edges, extrema, zero_counts = activation_distribution(
        model,
        batches,
        ACTIVATION_CATEGORIES,
        num_batches=NUM_BATCHES,
        exponent_counts=exponent_counts,
    )
    for label, (minimum, maximum) in extrema.items():
        counts = histograms[label]
        overflow_fraction = float((counts[0] + counts[-1]) / counts.sum())
        zero_fraction = float(zero_counts[label] / counts.sum())
        print(
            f"{label}: min={minimum.item():.6g}, max={maximum.item():.6g}, "
            f"outside fit range={overflow_fraction:.3%}, exact zeros={zero_fraction:.3%}"
        )
    save_activation_results(
        RESULTS_PATH,
        histograms,
        edges,
        extrema,
        zero_counts,
        ACTIVATION_CATEGORIES,
        model_name=MODEL_NAME,
        num_batches=NUM_BATCHES,
        sequence_length=SEQUENCE_LENGTH,
        sequences_per_batch=SEQUENCES_PER_BATCH,
        bin_width=BIN_WIDTH,
        limit=LIMIT,
        model_dtype=str(dtype),
    )
    print(f"Saved activation results to {RESULTS_PATH}")
    if dtype == torch.bfloat16:
        for label, counts in exponent_counts.items():
            assert int(counts.sum()) == int(histograms[label].sum()), label
        torch.save({
            "format_version": 1, "model_name": MODEL_NAME,
            "model_dtype": str(dtype), "analysis_device": str(device),
            "num_batches": NUM_BATCHES, "sequence_length": SEQUENCE_LENGTH,
            "sequences_per_batch": SEQUENCES_PER_BATCH,
            "categories": list(histograms),
            "exponent_counts": {label: counts.cpu() for label, counts in exponent_counts.items()},
            "zero_counts": {label: int(count) for label, count in zero_counts.items()},
            "extrema": {label: (lo.cpu(), hi.cpu()) for label, (lo, hi) in extrema.items()},
            "exponent_definition": "raw BF16 field; normal exponent = field - 127",
        }, EXPONENT_RESULTS_PATH)
        print(f"Saved activation exponent results to {EXPONENT_RESULTS_PATH}")


if __name__ == "__main__":
    main()
