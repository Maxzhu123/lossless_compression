"""Public lossless BF16 compression and decompression API."""

import torch

from .code_storage import CompressedTensor, Distribution
from .codec.runtime import compress_dense, decode as _decode
from .tensor_buffer import TensorBuffer


def compress(
    data: torch.Tensor,
    distribution: Distribution = Distribution(),
    buffer: TensorBuffer | None = None,
) -> CompressedTensor:
    """Losslessly encode the exponent byte of a CUDA bfloat16 tensor."""
    return compress_dense(data, distribution, buffer)


def decompress(data: CompressedTensor) -> torch.Tensor:
    """Decode a tensor produced by :func:`compress`."""
    return _decode(data)
