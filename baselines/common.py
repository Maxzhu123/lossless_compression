"""Shared BF16 experiment interface (compression does not preserve autograd)."""
from dataclasses import dataclass, fields, is_dataclass
from typing import Any

import torch


def validate_tensor(tensor, *, cuda=False):
    if not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.bfloat16:
        raise TypeError("baselines require a torch.bfloat16 tensor")
    if tensor.layout != torch.strided:
        raise ValueError("baselines require a dense strided tensor")
    if tensor.device.type not in {"cpu", "cuda"}:
        raise ValueError("baselines support CPU and CUDA devices only")
    if cuda and tensor.device.type != "cuda":
        raise ValueError("this baseline requires a CUDA tensor")
    return tensor.detach().contiguous().reshape(-1)


def storage_bytes(value):
    """Count unique tensor storages and byte buffers, excluding Python objects."""
    seen = set()

    def visit(obj):
        if isinstance(obj, torch.Tensor):
            storage = obj.untyped_storage()
            key = (str(obj.device), storage.data_ptr())
            if key in seen:
                return 0
            seen.add(key)
            return storage.nbytes()
        if isinstance(obj, (bytes, bytearray)):
            if id(obj) in seen:
                return 0
            seen.add(id(obj))
            return len(obj)
        if is_dataclass(obj):
            return sum(visit(getattr(obj, f.name)) for f in fields(obj))
        if isinstance(obj, dict):
            return sum(visit(v) for v in obj.values())
        if isinstance(obj, (tuple, list)):
            return sum(visit(v) for v in obj)
        return 0

    return visit(value)


@dataclass
class CompressedTensor:
    method: str
    payload: Any
    shape: tuple[int, ...]
    device: torch.device
    compressed_bytes: int
    original_bytes: int

    def memory_size(self):
        """Retained buffer bytes, including padding/scratch; not peak memory."""
        return storage_bytes(self.payload)

    @property
    def storage_ratio(self):
        """Compressed / original bytes (smaller is better); empty is undefined."""
        return self.compressed_bytes / self.original_bytes if self.original_bytes else float("nan")


def check_method(data, method):
    if not isinstance(data, CompressedTensor) or data.method != method:
        raise TypeError(f"expected a {method} CompressedTensor")
