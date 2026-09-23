"""Re-tokenize the local nanoGPT FineWeb documents with the Qwen tokenizer."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import tiktoken
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / 'nanogpt/data/fineweb10B'
DATA_DIR = ROOT / 'artefacts/qwen/fineweb'
MODEL_ID = 'Qwen/Qwen3-4B'
TOKENS_PER_SHARD = 1_000_000


def documents(files):
    """Preserve document boundaries and UTF-8 across source shard boundaries."""
    gpt2 = tiktoken.get_encoding('gpt2')
    pending = []
    for path in files:
        header = np.fromfile(path, dtype='<i4', count=256)
        if header[0] != 20240520 or header[1] != 1:
            raise ValueError(f'Not a nanoGPT GPT-2 FineWeb shard: {path}')
        count = int(header[2])
        if path.stat().st_size != 1024 + 2 * count:
            raise ValueError(f'Invalid shard length: {path}')
        tokens = np.memmap(path, dtype='<u2', mode='r', offset=1024, shape=(count,))
        for start in range(0, count, 1_000_000):
            block = tokens[start:start+1_000_000]
            previous = 0
            for end in np.flatnonzero(block == gpt2.eot_token):
                pending.extend(block[previous:end].tolist())
                if pending:
                    yield gpt2.decode(pending)
                    pending.clear()
                previous = int(end) + 1
            pending.extend(block[previous:].tolist())
    if pending:
        yield gpt2.decode(pending)


def prepare(source_dir=SOURCE_DIR, output_dir=DATA_DIR, model_id=MODEL_ID,
            train_tokens=50_000_000, val_tokens=524_288):
    source_dir, output_dir = Path(source_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if list(output_dir.glob('*.bin')) or (output_dir / 'manifest.json').exists():
        raise FileExistsError(f'{output_dir} already contains data; choose a new output directory')
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    eot = tokenizer.convert_tokens_to_ids('<|endoftext|>')
    if eot is None or eot == tokenizer.unk_token_id:
        raise ValueError('The tokenizer has no <|endoftext|> document separator')
    manifest = dict(model_id=model_id, dtype='uint32', tokenizer_sha256=hashlib.sha256(
        tokenizer.backend_tokenizer.to_str().encode()).hexdigest(), eot_token_id=eot,
        source_tokenizer='gpt2', source_dir=str(source_dir.resolve()), splits={})
    for split, limit in [('val', val_tokens), ('train', train_tokens)]:
        files = sorted(source_dir.glob(f'fineweb_{split}_*.bin'))
        if not files:
            raise FileNotFoundError(f'No {split} shards in {source_dir}')
        pending, total, shard, docs = [], 0, 0, 0
        def write(values):
            nonlocal shard
            name = f'{split}_{shard:06d}.bin'
            np.asarray(values, dtype='<u4').tofile(output_dir / name)
            shard += 1
        for text in documents(files):
            ids = [eot] + tokenizer.encode(text, add_special_tokens=False)
            if limit is not None:
                ids = ids[:max(0, limit-total)]
            pending.extend(ids); total += len(ids); docs += 1
            while len(pending) >= TOKENS_PER_SHARD:
                write(pending[:TOKENS_PER_SHARD]);del pending[:TOKENS_PER_SHARD]
            if limit is not None and total >= limit:
                break
        if pending:write(pending)
        manifest['splits'][split] = dict(tokens=total, documents=docs, shards=shard,
                                        source_files=[str(p.resolve()) for p in files], limit=limit)
        print(f'{split}: {total:,} Qwen tokens, {docs:,} documents, {shard} shards', flush=True)
    tokenizer.save_pretrained(output_dir / 'tokenizer')
    (output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, default=SOURCE_DIR)
    parser.add_argument('--output-dir', type=Path, default=DATA_DIR)
    parser.add_argument('--train-tokens', type=int, default=50_000_000, help='0 means all source tokens')
    parser.add_argument('--val-tokens', type=int, default=524_288, help='0 means all source tokens')
    args = parser.parse_args()
    prepare(args.source_dir, args.output_dir, train_tokens=args.train_tokens or None,
            val_tokens=args.val_tokens or None)
