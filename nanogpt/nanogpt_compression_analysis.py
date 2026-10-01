"""Compare LCT configurations on saved nanoGPT weights or Muon momentum.

Run from the repository root::

    python nanogpt/nanogpt_compression_analysis.py

Edit the settings below to select layers, weight types, and configurations; no CLI arguments.

TENSOR_SOURCE selects 'weights' or 'muon_momentum'. Momentum requires a
checkpoint from train_gpt_lct.py with initialized Muon state (after a training
step). WEIGHT_TYPES selects the corresponding parameter names in either mode;
Muon covers block matrices, not embeddings, the LM head, biases, or norms.
Adam moments are not saved by this trainer and cannot be analysed here.

Accepts plain state dictionaries (older trainer) and checkpoints with a model
state dictionary under 'model' or 'state_dict'. Loads the checkpoint on CPU and
compresses one selected tensor on CUDA at a time; no model construction needed.
Older trainers save FP32 weights that are cast to BF16 during the forward pass.
CAST_TO_BF16 enables that conversion here. Ratios and bitwise verification then
refer to the BF16 representation, NOT lossless compression of the original FP32
checkpoint. Source dtypes and conversion flags are recorded in the results.
Only the entries in CONFIGURATIONS are tested, using their specified parameters.
No codebook parameters are fitted to the weights or presets added automatically.

Reported bytes are CompressedTensor.memory_size(): owned tensor allocations,
including padding, metadata, and private overflow storage. Shared codebook
tables, allocator caching, and temporary encode/decode workspace are excluded.
Ratios are original/compressed (larger is better); savings may be negative.
Summary rows include arithmetic means over tensors as well as aggregate ratios
computed from summed bytes, separately for each weight type and all selected weights.
CSV overflow rate is the percentage of codec streams requiring fallback storage,
including padded streams. Console output contains only the arithmetic mean
compression ratio for each configuration, with equal weight per selected tensor.
Every result requires a bitwise-exact decode. CUDA and Triton are required.
"""

from collections.abc import Mapping
import csv
import json
from pathlib import Path
import re
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

from LCT.comp_format import DistType, Distribution, NoiseLevel
from LCT.compress import compress, decompress
from LCT.codec.runtime import geometry


# Edit these settings before running. Compression always uses CUDA.
CHECKPOINT_PATH = ROOT / "nanogpt/logs/selection/3350.pt"
TENSOR_SOURCE = "muon_momentum"  # "weights" or "muon_momentum"
OUTPUT_PATH = ROOT / "artefacts/nanogpt_compression" / CHECKPOINT_PATH.parent.name / CHECKPOINT_PATH.stem / TENSOR_SOURCE
CPU_THREADS = 4
CAST_TO_BF16 = True  # False: require selected tensors to already be BF16.

# Each entry is (unique result label, Distribution). Add/remove entries as needed.
# param: scale (Gaussian: std); mean: location; zero_prob: exact-zero probability.
# Layouts: CLEAN, MEDIUM, HIGH, SPARSE. LCT rounds param/mean to multiples of 0.25
# (param >= 0.25) and zero_prob to two decimals; metadata records effective values.
CONFIGURATIONS = [
    ("Power-Law", Distribution(DistType.EMPIRICAL, zero_prob=0.02)),
    ("gaussian", Distribution(DistType.GAUSSIAN, zero_prob=0.02)),
    ("laplace", Distribution(DistType.LAPLACE, zero_prob=0.02)),
    ("gamma", Distribution(DistType.GAMMA, zero_prob=0.02)),
]

LAYERS = None  # None: all matching blocks; e.g. [0, 3, 11] for this 12-layer model.
WEIGHT_TYPES = ["up_proj", "down_proj"]

# Valid selectors, mapped to checkpoint suffixes. Types can be mixed across layers.
LAYER_WEIGHT_TYPES = {
    "up_proj": "mlp.fc.weight",
    "down_proj": "mlp.proj.weight",
    "q_proj": "attn.q.weight",
    "k_proj": "attn.k.weight",
    "v_proj": "attn.v.weight",
    "o_proj": "attn.proj.weight",
    "up_proj.bias": "mlp.fc.bias",
    "down_proj.bias": "mlp.proj.bias",
    "q_proj.bias": "attn.q.bias",
    "k_proj.bias": "attn.k.bias",
    "v_proj.bias": "attn.v.bias",
    "o_proj.bias": "attn.proj.bias",
    "norm1": "norm1.gains",
    "norm2": "norm2.gains",
}
# Global tensors have no layer index and are included when selected, regardless of LAYERS.
GLOBAL_WEIGHT_TYPES = {
    "embeddings": "embed.weight",
    "lm_head": "proj.weight",
    "lm_head.bias": "proj.bias",
    "embed_norm": "norm1.gains",
    "final_norm": "norm2.gains",
}
LAYER_KEY = re.compile(r"^blocks\.(\d+)\.(.+)$")


def load_checkpoint(path, tensor_source="weights"):
    """Load weights or Muon state, normalizing compiled/DDP parameter names."""
    if tensor_source not in ("weights", "muon_momentum"):
        raise ValueError("TENSOR_SOURCE must be 'weights' or 'muon_momentum'")
    path = Path(path)
    if path.stat().st_size == 0:
        raise ValueError(f"Checkpoint is empty: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Expected a state dictionary or a checkpoint containing one")
    if tensor_source == "muon_momentum":
        if "muon_momentum" not in checkpoint:
            raise ValueError("Checkpoint has no muon_momentum; use a checkpoint saved by train_gpt_lct.py")
        state = checkpoint["muon_momentum"]
    else:
        state = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    if not isinstance(state, Mapping):
        raise ValueError(f"Checkpoint {tensor_source} must be a state dictionary")
    normalized = {}
    for name, tensor in state.items():
        if not isinstance(name, str):
            continue
        if not isinstance(tensor, torch.Tensor):
            if tensor_source != "muon_momentum":
                continue
            if tensor is not None:
                raise ValueError(f"Invalid Muon momentum for {name}: expected a tensor or None")
        while name.startswith(("_orig_mod.", "module.")):
            name = name.split(".", 1)[1]
        if name in normalized:
            raise ValueError(f"Duplicate parameter after removing wrapper prefixes: {name}")
        normalized[name] = tensor
    if not normalized:
        raise ValueError(f"Checkpoint contains no {tensor_source} entries")
    step = checkpoint.get("step") if state is not checkpoint else None
    return normalized, int(step) if step is not None else None


def checkpoint_entries(state, layers=None, weight_types=("up_proj", "down_proj")):
    """Select parameters from this repo's nanoGPT naming scheme, not buffers."""
    selected_types = set(weight_types)
    valid_types = LAYER_WEIGHT_TYPES.keys() | GLOBAL_WEIGHT_TYPES.keys()
    if not selected_types or selected_types - valid_types:
        raise ValueError(f"WEIGHT_TYPES must be a nonempty selection from {sorted(valid_types)}")
    entries = []
    layer_types = {suffix: name for name, suffix in LAYER_WEIGHT_TYPES.items()}
    global_types = {suffix: name for name, suffix in GLOBAL_WEIGHT_TYPES.items()}
    available_layers = set()
    for key, tensor in state.items():
        match = LAYER_KEY.fullmatch(key)
        if match:
            layer, projection = int(match[1]), layer_types.get(match[2])
            available_layers.add(layer)
            if projection in selected_types and (layers is None or layer in layers):
                entries.append((layer, projection, key, tensor))
        elif global_types.get(key) in selected_types:
            entries.append((None, global_types[key], key, tensor))
    if layers is not None and set(layers) - available_layers:
        raise ValueError(f"Layer indices absent from checkpoint: {sorted(set(layers) - available_layers)}")
    if not entries:
        raise ValueError("No matching tensors found for LAYERS and WEIGHT_TYPES")
    missing_types = selected_types - {entry[1] for entry in entries}
    if missing_types:
        raise ValueError(f"Weight types absent from selected layers: {sorted(missing_types)}")
    if layers is not None and selected_types & LAYER_WEIGHT_TYPES.keys():
        missing_layers = set(layers) - {entry[0] for entry in entries}
        if missing_layers:
            raise ValueError(f"Requested layers have no selected weight types: {sorted(missing_layers)}")
    return sorted(entries, key=lambda entry: (-1 if entry[0] is None else entry[0], entry[1], entry[2]))


@torch.no_grad()
def measure(tensor, distribution):
    """Measure complete storage and verify original BF16 bits, including zeros."""
    if tensor.dtype != torch.bfloat16 or tensor.device.type != "cuda":
        raise ValueError("LCT comparison requires BF16 tensors on CUDA")
    packed = compress(tensor, distribution=distribution, allow_raw=False)
    try:
        restored = decompress(packed)
        if restored.shape != tensor.shape or restored.dtype != tensor.dtype:
            raise AssertionError("LCT round trip changed shape or dtype")
        if not torch.equal(tensor.view(torch.int16), restored.view(torch.int16)):
            raise AssertionError("LCT round trip changed BF16 bits")
        block_symbols, lanes, _, _ = geometry(distribution)
        streams = (packed.storage_numel // block_symbols) * lanes
        overflow_streams = int(packed.fallback_count.item())
        return packed.memory_size(), streams, overflow_streams
    finally:
        packed.free()


def size_metrics(original, compressed):
    return {
        "original_bytes": original,
        "compressed_bytes": compressed,
        "original_over_compressed": original / compressed,
        "savings_percent": 100 * (1 - compressed / original),
    }


def run(checkpoint_path, output, layers, configs, weight_types=("up_proj", "down_proj"),
        cast_to_bf16=True, tensor_source="weights"):
    if not configs:
        raise ValueError("CONFIGURATIONS must contain at least one entry")
    labels = [label for label, _ in configs]
    if any(not isinstance(label, str) or not label.strip() for label in labels):
        raise ValueError("Each configuration needs a non-empty string label")
    if len(set(labels)) != len(labels):
        raise ValueError("Configuration labels must be unique")
    if any(not isinstance(dist, Distribution) for _, dist in configs):
        raise TypeError("Each configuration must contain a Distribution")
    if not torch.cuda.is_available():
        raise RuntimeError("LCT requires an available CUDA device")
    # Use the current CUDA device for both weights and LCT's Huffman tables.
    device = torch.device("cuda", torch.cuda.current_device())
    state, step = load_checkpoint(checkpoint_path, tensor_source)
    entries = checkpoint_entries(state, layers, weight_types)
    del state
    for _, _, key, source in entries:
        if source is None:
            raise ValueError(f"Muon momentum for {key} is uninitialized; use a checkpoint after a training step")
        if not source.is_floating_point() or source.numel() == 0:
            raise ValueError(f"Expected nonempty floating-point tensor for {key}")
        if source.dtype != torch.bfloat16 and not cast_to_bf16:
            raise ValueError(f"{key} is {source.dtype}; set CAST_TO_BF16=True to analyse its BF16 representation")
    output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "checkpoint_path": str(Path(checkpoint_path).resolve()), "step": step,
        "tensor_source": tensor_source,
        "layers": sorted({e[0] for e in entries if e[0] is not None}),
        "cast_to_bf16": cast_to_bf16,
        "source_dtypes": {e[2]: str(e[3].dtype) for e in entries},
        "ratio_and_verification_reference": f"BF16 representation of selected {tensor_source}",
        "global_tensors": [e[2] for e in entries if e[0] is None],
        "weight_types": sorted({e[1] for e in entries}),
        "device": str(device), "gpu": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__, "dtype": "bfloat16",
        "buffer": "private", "allow_raw": False,
        "storage_measurement": "CompressedTensor.memory_size; excludes shared tables and workspace",
        "overflow_definition": "100 * fallback streams / total codec streams (including padding)",
        "configurations": {
            label: {"family": dist.family.value, "noise_level": dist.noise_level.name,
                    "param": dist.param, "mean": dist.mean, "zero_prob": dist.zero_prob}
            for label, dist in configs
        },
        "complete": False,
    }
    metadata_path = output / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    totals = {}
    fields = ["tensor_source", "layer", "weight_type", "tensor", "shape", "configuration", "elements",
              "source_dtype", "converted_to_bf16", "checkpoint_tensor_bytes",
              "original_bytes", "compressed_bytes", "original_over_compressed",
              "savings_percent", "streams", "overflow_streams", "overflow_percent", "bitwise_equal"]
    with (output / "per_tensor.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for layer, projection, key, source in entries:
            tensor = source.to(device=device, dtype=torch.bfloat16).contiguous()
            for label, distribution in configs:
                compressed, streams, overflow_streams = measure(tensor, distribution)
                overflow_percent = 100 * overflow_streams / streams
                metrics = size_metrics(tensor.nbytes, compressed)
                writer.writerow({"tensor_source": tensor_source,
                                 "layer": layer, "weight_type": projection, "tensor": key,
                                 "shape": json.dumps(list(tensor.shape)), "configuration": label,
                                 "elements": tensor.numel(), **metrics, "streams": streams,
                                 "source_dtype": str(source.dtype),
                                 "converted_to_bf16": source.dtype != torch.bfloat16,
                                 "checkpoint_tensor_bytes": source.nbytes,
                                 "overflow_streams": overflow_streams,
                                 "overflow_percent": overflow_percent, "bitwise_equal": True})
                file.flush()
                for group in (projection, "all_selected"):
                    total = totals.setdefault((label, group), [0, 0, 0, 0.0, 0.0, 0, 0, 0.0])
                    total[0] += tensor.nbytes
                    total[1] += compressed
                    total[2] += 1
                    total[3] += metrics["original_over_compressed"]
                    total[4] += metrics["savings_percent"]
                    total[5] += streams
                    total[6] += overflow_streams
                    total[7] += overflow_percent
            del tensor
    with (output / "summary.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=[
            "tensor_source", "configuration", "group", "tensor_count", *size_metrics(1, 1),
            "mean_original_bytes", "mean_compressed_bytes",
            "mean_original_over_compressed", "mean_savings_percent",
            "streams", "overflow_streams", "overflow_percent", "mean_overflow_percent",
        ])
        writer.writeheader()
        for (label, group), total in totals.items():
            original, compressed, count, ratio_sum, savings_sum, streams, overflow_streams, overflow_sum = total
            writer.writerow({"tensor_source": tensor_source,
                             "configuration": label, "group": group, "tensor_count": count,
                             **size_metrics(original, compressed),
                             "mean_original_bytes": original / count,
                             "mean_compressed_bytes": compressed / count,
                             "mean_original_over_compressed": ratio_sum / count,
                             "mean_savings_percent": savings_sum / count,
                             "streams": streams, "overflow_streams": overflow_streams,
                             "overflow_percent": 100 * overflow_streams / streams,
                             "mean_overflow_percent": overflow_sum / count})
            if group == "all_selected":
                print(f"{label:20s} mean compression ratio {ratio_sum / count:.4f}", flush=True)
    metadata["complete"] = True
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")


def main():
    torch.set_num_threads(CPU_THREADS)
    run(CHECKPOINT_PATH, OUTPUT_PATH, LAYERS, CONFIGURATIONS, WEIGHT_TYPES, CAST_TO_BF16,
        tensor_source=TENSOR_SOURCE)


if __name__ == "__main__":
    main()
