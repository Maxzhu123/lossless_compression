"""Public lossless BF16 compression and decompression API."""

import torch

from .code_storage import CompressedTensor, Distribution
from .codec.runtime import compress_dense, decode as _decode
from .ops.pointwise import pointwise_compressed_dense
from .ops.registry import ADD, MULTIPLY
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


def compressed_add(
    data: CompressedTensor,
    other: torch.Tensor,
    *,
    dense_output: bool = True,
    buffer: TensorBuffer | None = None,
    distribution=None,
) -> torch.Tensor | CompressedTensor:
    """Add dense BF16 values to a compressed tensor."""
    return pointwise_compressed_dense(
        data, other, ADD, dense_output=dense_output,
        buffer=buffer, distribution=distribution,
    )


def compressed_multiply(
    data: CompressedTensor,
    other: torch.Tensor,
    *,
    dense_output: bool = True,
    buffer: TensorBuffer | None = None,
    distribution=None,
) -> torch.Tensor | CompressedTensor:
    """Multiply compressed values by dense BF16 values."""
    return pointwise_compressed_dense(
        data, other, MULTIPLY, dense_output=dense_output,
        buffer=buffer, distribution=distribution,
    )
