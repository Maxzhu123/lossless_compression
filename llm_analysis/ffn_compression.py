"""Compare LCT configurations on selected local Nemotron-H parameters.

Run from the repository root::

    python llm_analysis/ffn_compression.py

Edit the settings below to select layers, weight types, and configurations; no CLI arguments.

Each selected tensor is compressed separately, without casting or normalising
the checkpoint's BF16 values. Only one tensor is loaded onto CUDA at a time.
Only the entries in CONFIGURATIONS are tested, using their specified parameters.
No codebook parameters are fitted to the weights or presets added automatically.

Reported bytes are CompressedTensor.memory_size(): owned tensor allocations,
including padding, metadata, and private overflow storage. Shared codebook
tables, allocator caching, and temporary encode/decode workspace are excluded.
Ratios are original/compressed (larger is better); savings may be negative.
Summary rows include arithmetic means over tensors as well as aggregate ratios
computed from summed bytes, separately for each weight type and all selected weights.
Overflow rate is the percentage of codec streams requiring fallback storage,
including padded streams. The printed rate is averaged equally over tensors.
Every result requires a bitwise-exact decode. CUDA and Triton are required.
"""

import csv
import json
from pathlib import Path
import re
import sys

import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

from LCT.comp_format import DistType, Distribution, NoiseLevel
from LCT.compress import compress, decompress
from LCT.codec.runtime import geometry


# Edit these settings before running. Compression always uses CUDA.
MODEL_PATH = ROOT / "artefacts/Nemotron-H-8B-Base-8K"
OUTPUT_PATH = ROOT / "artefacts/nemotron_h_ffn_compression"
CPU_THREADS = 4

# Each entry is (unique result label, Distribution). Add/remove entries as needed.
# param: scale (Gaussian: std); mean: location; zero_prob: exact-zero probability.
# Layouts: CLEAN, MEDIUM, HIGH, SPARSE. LCT rounds param/mean to multiples of 0.25
# (param >= 0.25) and zero_prob to two decimals; metadata records effective values.
CONFIGURATIONS = [
    ("empirical", Distribution(DistType.EMPIRICAL)),
    ("gaussian", Distribution(DistType.GAUSSIAN)),
    ("laplace", Distribution(DistType.LAPLACE)),
    ("gamma", Distribution(DistType.GAMMA)),
]

LAYERS = None  # Numbered layers only; None: all matching layers, e.g. [7, 18] for attention.
WEIGHT_TYPES = ["in_proj", "out_proj"] # ["q_proj", "k_proj", "v_proj", "o_proj"]

# Valid selectors, mapped to checkpoint suffixes. Types can be mixed across layers.
LAYER_WEIGHT_TYPES = {
    "up_proj": "mixer.up_proj.weight",         # FFN
    "down_proj": "mixer.down_proj.weight",
    "q_proj": "mixer.q_proj.weight",           # Attention
    "k_proj": "mixer.k_proj.weight",
    "v_proj": "mixer.v_proj.weight",
    "o_proj": "mixer.o_proj.weight",
    "in_proj": "mixer.in_proj.weight",         # Mamba
    "out_proj": "mixer.out_proj.weight",
    "conv1d": "mixer.conv1d.weight",
    "conv1d.bias": "mixer.conv1d.bias",
    "A_log": "mixer.A_log",
    "D": "mixer.D",
    "dt_bias": "mixer.dt_bias",
    "mixer_norm": "mixer.norm.weight",
    "norm": "norm.weight",                    # Every numbered layer
}
# Global tensors have no layer index and are included when selected, regardless of LAYERS.
GLOBAL_WEIGHT_TYPES = {
    "embeddings": "embeddings.weight",
    "norm_f": "norm_f.weight",
    "lm_head": "lm_head.weight",
}
LAYER_KEY = re.compile(r"^layers\.(\d+)\.(.+)$")


def checkpoint_entries(model_path, layers=None, weight_types=("up_proj", "down_proj")):
    """Select named parameter types, accepting old and new checkpoint prefixes."""
    selected_types = set(weight_types)
    valid_types = LAYER_WEIGHT_TYPES.keys() | GLOBAL_WEIGHT_TYPES.keys()
    if not selected_types or selected_types - valid_types:
        raise ValueError(f"WEIGHT_TYPES must be a nonempty selection from {sorted(valid_types)}")
    config = json.loads((model_path / "config.json").read_text())
    if config.get("model_type") != "nemotron_h":
        raise ValueError(f"Expected a Nemotron-H checkpoint at {model_path}")
    index_path = model_path / "model.safetensors.index.json"
    if index_path.is_file():
        weight_map = json.loads(index_path.read_text())["weight_map"]
    else:
        with safe_open(model_path / "model.safetensors", framework="pt", device="cpu") as handle:
            weight_map = {key: "model.safetensors" for key in handle.keys()}
    entries = []
    layer_types = {suffix: name for name, suffix in LAYER_WEIGHT_TYPES.items()}
    global_types = {suffix: name for name, suffix in GLOBAL_WEIGHT_TYPES.items()}
    available_layers = set()
    for key, shard in weight_map.items():
        suffix = key.removeprefix("backbone.").removeprefix("model.")
        match = LAYER_KEY.fullmatch(suffix)
        if match:
            layer, projection = int(match[1]), layer_types.get(match[2])
            available_layers.add(layer)
            if projection in selected_types and (layers is None or layer in layers):
                entries.append((layer, projection, key, shard))
        elif global_types.get(suffix) in selected_types:
            entries.append((None, global_types[suffix], key, shard))
    if layers is not None and set(layers) - available_layers:
        raise ValueError(f"Layer indices absent from checkpoint: {sorted(set(layers) - available_layers)}")
    if not entries:
        raise ValueError("No matching weights found for LAYERS and WEIGHT_TYPES")
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
        raise ValueError("LCT comparison requires original BF16 weights on CUDA")
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


def run(model_path, output, layers, configs, weight_types=("up_proj", "down_proj")):
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
    entries = checkpoint_entries(model_path, layers, weight_types)
    output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "model_path": str(model_path.resolve()), "layers": sorted({e[0] for e in entries if e[0] is not None}),
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
    fields = ["layer", "weight_type", "tensor", "shape", "configuration", "elements",
              "original_bytes", "compressed_bytes", "original_over_compressed",
              "savings_percent", "streams", "overflow_streams", "overflow_percent", "bitwise_equal"]
    with (output / "per_tensor.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for layer, projection, key, shard in entries:
            with safe_open(model_path / shard, framework="pt", device="cpu") as handle:
                source = handle.get_tensor(key)
                if source.dtype != torch.bfloat16 or source.numel() == 0:
                    raise ValueError(f"Expected a nonempty BF16 tensor for {key}, got {source.dtype}, {source.shape}")
                tensor = source.to(device=device).contiguous()
                del source
            for label, distribution in configs:
                compressed, streams, overflow_streams = measure(tensor, distribution)
                overflow_percent = 100 * overflow_streams / streams
                metrics = size_metrics(tensor.nbytes, compressed)
                writer.writerow({"layer": layer, "weight_type": projection, "tensor": key,
                                 "shape": json.dumps(list(tensor.shape)), "configuration": label,
                                 "elements": tensor.numel(), **metrics, "streams": streams,
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
            "configuration", "group", "tensor_count", *size_metrics(1, 1),
            "mean_original_bytes", "mean_compressed_bytes",
            "mean_original_over_compressed", "mean_savings_percent",
            "streams", "overflow_streams", "overflow_percent", "mean_overflow_percent",
        ])
        writer.writeheader()
        for (label, group), total in totals.items():
            original, compressed, count, ratio_sum, savings_sum, streams, overflow_streams, overflow_sum = total
            writer.writerow({"configuration": label, "group": group, "tensor_count": count,
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
    run(MODEL_PATH, OUTPUT_PATH, LAYERS, CONFIGURATIONS, WEIGHT_TYPES)


if __name__ == "__main__":
    main()
