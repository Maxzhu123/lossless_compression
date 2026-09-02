from pathlib import Path

import torch
from torch import Tensor, nn
from transformers.models.nemotron_h.modeling_nemotron_h import (
    NemotronHForCausalLM,
    NemotronHMamba2Mixer,
    NemotronHRMSNorm,
)

from histogram import tensor_histogram


MODEL_NAME = "nvidia/Nemotron-H-8B-Base-8K"
RESULTS_PATH = Path(__file__).parent / "weight_distribution_results.pt"
RESULTS_FORMAT_VERSION = 2
BIN_WIDTH = 0.01
LIMIT = 50.0


def group_parameters(model: nn.Module) -> dict[str, list[Tensor]]:
    """Group parameters by the Nemotron component that owns them."""
    groups: dict[str, list[Tensor]] = {}
    seen: set[int] = set()

    def add(label: str, parameter: Tensor) -> None:
        if id(parameter) in seen:
            return
        seen.add(id(parameter))
        groups.setdefault(label, []).append(parameter)

    for module_name, module in model.named_modules():
        suffix = module_name.rsplit(".", 1)[-1]
        for name, parameter in module.named_parameters(recurse=False):
            if isinstance(module, nn.Embedding):
                label = "Embedding weights"
            elif isinstance(module, nn.Linear):
                if module_name == "lm_head":
                    label = "LM head weights"
                elif name == "bias":
                    label = "Biases"
                else:
                    label = {
                        "up_proj": "FFN input weights",
                        "down_proj": "FFN output weights",
                        "q_proj": "Attention query weights",
                        "k_proj": "Attention key weights",
                        "v_proj": "Attention value weights",
                        "o_proj": "Attention output weights",
                        "in_proj": "Mamba input weights",
                        "out_proj": "Mamba output weights",
                    }.get(suffix, "Other")
            elif isinstance(module, NemotronHMamba2Mixer):
                label = "Mamba state parameters"
            elif module_name.endswith(".mixer.conv1d"):
                label = "Mamba convolution weights"
            elif isinstance(module, NemotronHRMSNorm) or "RMSNorm" in type(module).__name__:
                label = "Norm weights"
            elif name == "bias":
                label = "Biases"
            else:
                label = "Other"
            add(label, parameter)

    label_order = (
        "Embedding weights",
        "FFN input weights",
        "FFN output weights",
        "Attention query weights",
        "Attention key weights",
        "Attention value weights",
        "Attention output weights",
        "Mamba input weights",
        "Mamba output weights",
        "Mamba convolution weights",
        "LM head weights",
        "Norm weights",
        "Mamba state parameters",
        "Biases",
        "Other",
    )
    return {label: groups[label] for label in label_order if label in groups}


@torch.no_grad()
def weight_distribution(
    model: nn.Module,
    *,
    bin_width: float = BIN_WIDTH,
    limit: float = LIMIT,
) -> tuple[dict[str, Tensor], Tensor, dict[str, tuple[Tensor, Tensor]]]:
    """Calculate histograms and extrema for Nemotron parameter groups."""
    groups = group_parameters(model)
    if not groups:
        raise ValueError("Model contains no parameters")

    histograms: dict[str, Tensor] = {}
    extrema: dict[str, tuple[Tensor, Tensor]] = {}
    edges: Tensor | None = None
    for label, tensors in groups.items():
        counts, group_edges, minimum, maximum = tensor_histogram(
            tensors,
            bin_width=bin_width,
            limit=limit,
        )
        histograms[label] = counts
        extrema[label] = (minimum, maximum)
        edges = group_edges

    if edges is None:
        raise RuntimeError("No parameter histograms were produced")
    return histograms, edges, extrema


def save_weight_results(
    path: Path,
    histograms: dict[str, Tensor],
    edges: Tensor,
    extrema: dict[str, tuple[Tensor, Tensor]],
    *,
    model_name: str,
    bin_width: float,
    limit: float,
) -> None:
    """Save weight histograms, extrema, and run metadata as CPU tensors."""
    result = {
        "format_version": RESULTS_FORMAT_VERSION,
        "model_name": model_name,
        "bin_width": bin_width,
        "limit": limit,
        "categories": list(histograms),
        "histograms": {
            label: counts.detach().cpu()
            for label, counts in histograms.items()
        },
        "edges": edges.detach().cpu(),
        "extrema": {
            label: (minimum.detach().cpu(), maximum.detach().cpu())
            for label, (minimum, maximum) in extrema.items()
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, path)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = NemotronHForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=dtype,
    ).to(device)

    histograms, edges, extrema = weight_distribution(model)
    for label, (minimum, maximum) in extrema.items():
        print(f"{label}: min={minimum.item():.6g}, max={maximum.item():.6g}")
    save_weight_results(
        RESULTS_PATH,
        histograms,
        edges,
        extrema,
        model_name=MODEL_NAME,
        bin_width=BIN_WIDTH,
        limit=LIMIT,
    )
    print(f"Saved weight results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
