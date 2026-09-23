"""Memory-mapped uint32 Qwen token streams; never consumes GPT-2 token IDs."""
import json
from pathlib import Path

import numpy as np
import torch


class TokenStream:
    def __init__(self, directory, split, model_id):
        directory = Path(directory)
        metadata = json.loads((directory / 'manifest.json').read_text())
        if metadata['model_id'] != model_id or metadata['dtype'] != 'uint32':
            raise ValueError('Dataset/tokenizer mismatch; run qwen.prepare_data for this model')
        self.files = sorted(directory.glob(f'{split}_*.bin'))
        if not self.files or metadata['splits'][split]['tokens'] < 2:
            raise ValueError(f'No usable {split} data in {directory}')
        self.index = self.position = 0
        self.tokens = np.memmap(self.files[0], dtype='<u4', mode='r')
        self.previous = self.read(1)

    def read(self, count):
        pieces = []
        while count:
            take = min(count, len(self.tokens)-self.position)
            pieces.append(np.array(self.tokens[self.position:self.position+take], copy=True))
            self.position += take;count -= take
            if self.position == len(self.tokens):
                self.index = (self.index+1) % len(self.files)
                self.tokens = np.memmap(self.files[self.index], dtype='<u4', mode='r')
                self.position = 0
        return np.concatenate(pieces)

    def batch(self, batch_size, sequence_length, device='cuda'):
        window = np.concatenate((self.previous, self.read(batch_size*sequence_length)))
        self.previous = window[-1:]
        x = torch.from_numpy(window[:-1].astype(np.int64)).reshape(batch_size, sequence_length)
        y = torch.from_numpy(window[1:].astype(np.int64)).reshape(batch_size, sequence_length)
        if str(device).startswith('cuda'):
            x, y = x.pin_memory(), y.pin_memory()
        return x.to(device, non_blocking=True), y.to(device, non_blocking=True)
