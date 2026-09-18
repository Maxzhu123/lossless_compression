"""Lossless CPU ZipNN with explicit CUDA staging when given CUDA tensors."""
from .common import CompressedTensor, check_method, validate_tensor


class ZipNN:
    def __init__(self, *, threads=0):
        if not isinstance(threads, int) or threads < 0:
            raise ValueError("threads must be a nonnegative integer (0 selects automatically)")
        try:
            from ._vendor.zipnn.zipnn import ZipNN as UpstreamZipNN
        except ImportError as exc:
            raise ImportError(
                "ZipNN needs its local C extension and numpy/torch/safetensors. "
                "Build with: python baselines/_vendor/zipnn/build.py build_ext --inplace"
            ) from exc
        self._codec = UpstreamZipNN(method="HUFFMAN", input_format="torch", threads=threads)

    def compress(self, tensor):
        flat = validate_tensor(tensor)
        # The upstream C bit-reordering stage mutates its input buffer.
        payload = bytes(self._codec.compress(flat.to("cpu", copy=True))) if flat.numel() else b""
        return CompressedTensor("zipnn", payload, tuple(tensor.shape), tensor.device,
                                len(payload), tensor.nbytes)

    def decompress(self, data):
        check_method(data, "zipnn")
        if not data.original_bytes:
            import torch
            return torch.empty(data.shape, dtype=torch.bfloat16, device=data.device)
        return self._codec.decompress(data.payload).reshape(data.shape).to(data.device)


def compress(tensor, *, threads=0):
    return ZipNN(threads=threads).compress(tensor)


def decompress(data):
    return ZipNN().decompress(data)
