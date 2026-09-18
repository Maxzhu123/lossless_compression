"""Stream selected local checkpoint groups into CPU histograms; never load all weights."""
from collections import defaultdict
import json
import math
from pathlib import Path
import sys

import torch
from safetensors import safe_open
from transformers.models.nemotron_h.configuration_nemotron_h import NemotronHConfig

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'llm_analysis'))
from weight_distribution import MODEL_PATH, MODEL_NAME, NemotronHForCausalLM, group_parameters, save_weight_results
from histogram import tensor_histogram

DEFAULT_OUTPUT = ROOT / 'artefacts/weight_distribution_large_results.pt'
EXPONENT_OUTPUT = ROOT / 'artefacts/weight_exponent_distribution_results.pt'


def exponent_histogram(chunk):
    """Count the stored eight-bit BF16 exponent field, without floating arithmetic."""
    if chunk.dtype != torch.bfloat16 or chunk.device.type != 'cpu':
        raise ValueError('Expected CPU BF16 weights')
    bits = chunk.contiguous().view(torch.int16).reshape(-1)
    exponents = ((bits >> 7) & 255).to(torch.int64)
    counts = torch.bincount(exponents, minlength=256)
    zeros = int(((bits & 0x7fff) == 0).sum())
    return counts, zeros


@torch.no_grad()
def collect(output=DEFAULT_OUTPUT, min_elements=500_000_000, chunk_elements=4_194_304,
            bin_width=0.001, limit=50.0, exponents_only=False):
    if chunk_elements <= 0:
        raise ValueError('chunk_elements must be positive')
    config = NemotronHConfig.from_pretrained(MODEL_PATH, local_files_only=True)
    # Architecture metadata only: parameters have shapes but no allocated storage.
    with torch.device('meta'):
        model = NemotronHForCausalLM(config)
    names = {id(p): name for name, p in model.named_parameters()}
    groups = group_parameters(model)
    index = json.loads((MODEL_PATH / 'model.safetensors.index.json').read_text())['weight_map']
    selected = {}
    for label, parameters in groups.items():
        count = sum(p.numel() for p in parameters)
        if count <= min_elements:
            continue
        entries = []
        for parameter in parameters:
            name = names[id(parameter)]
            # This checkpoint predates Transformers' backbone -> model rename.
            key = name if name in index else name.replace('model.', 'backbone.', 1)
            if key not in index:
                raise KeyError(f'Checkpoint is missing {name}')
            entries.append((key, tuple(parameter.shape)))
        selected[label] = (count, entries)
    del model, groups, names
    if not selected:
        raise ValueError('No parameter groups exceed the threshold')
    histograms, extrema = {}, {}
    provenance = {}
    zero_counts = {}
    edges = None
    for label, (count, entries) in selected.items():
        print(f'{label}: collecting {count:,} weights from {len(entries)} tensors', flush=True)
        counts = minimum = maximum = None
        zero_count = 0
        by_shard = defaultdict(list)
        for key, shape in entries:
            by_shard[index[key]].append((key, shape))
        for shard, tensors in by_shard.items():
            with safe_open(MODEL_PATH / shard, framework='pt', device='cpu') as handle:
                for key, expected_shape in tensors:
                    view = handle.get_slice(key)
                    shape = tuple(view.get_shape())
                    if shape != expected_shape:
                        raise ValueError(f'Shape mismatch for {key}: {shape} != {expected_shape}')
                    row_size = math.prod(shape[1:])
                    rows = max(1, chunk_elements // row_size)
                    for start in range(0, shape[0], rows):
                        chunk = view[start:start + rows]
                        if exponents_only:
                            partial, zeros = exponent_histogram(chunk)
                            zero_count += zeros
                            lo, hi = chunk.amin(), chunk.amax()
                        else:
                            partial, edges, lo, hi = tensor_histogram([chunk], bin_width=bin_width, limit=limit)
                        counts = partial if counts is None else counts + partial
                        minimum = lo.clone() if minimum is None else torch.minimum(minimum, lo)
                        maximum = hi.clone() if maximum is None else torch.maximum(maximum, hi)
                        del chunk
        assert int(counts.sum()) == count, label
        histograms[label] = counts
        extrema[label] = (minimum, maximum)
        zero_counts[label] = zero_count
        provenance[label] = {'numel': count, 'tensors': [key for key, _ in entries]}
        print(f'  complete: range [{float(minimum):.5g}, {float(maximum):.5g}]', flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    if exponents_only:
        torch.save({'format_version': 1, 'model_name': MODEL_NAME, 'analysis_device': 'cpu',
                    'categories': list(histograms), 'exponent_counts': histograms,
                    'zero_counts': zero_counts, 'extrema': extrema,
                    'exponent_definition': 'raw BF16 field; normal exponent = field - 127'}, output)
    else:
        save_weight_results(output, histograms, edges, extrema, model_name=MODEL_NAME,
                            bin_width=bin_width, limit=limit, model_dtype=str(torch.bfloat16))
    metadata = {'checkpoint': str(MODEL_PATH), 'selection': f'group numel > {min_elements}',
                'collection': 'CPU streaming, all values (no sampling)', 'chunk_elements': chunk_elements,
                'groups': provenance}
    output.with_suffix('.metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(f'Saved {output}', flush=True)


def main():
    # Edit these settings before running.
    exponents_only = False
    output = EXPONENT_OUTPUT if exponents_only else DEFAULT_OUTPUT
    min_elements = 500_000_000
    threads = 4
    chunk_elements = 4_194_304
    bin_width = 0.001
    limit = 50.0

    if threads < 1:
        raise ValueError('threads must be positive')
    torch.set_num_threads(threads)
    collect(output, min_elements, chunk_elements, bin_width, limit, exponents_only)


if __name__ == '__main__':
    main()
