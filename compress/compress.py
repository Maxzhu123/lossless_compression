"""Public lossless BF16 compression and decompression API."""

import torch

from .code_storage import CompressedTensor, Distribution
from .codec.runtime import (
    compress_dense,
    decode as _decode,
    decode_matrix_dense,
)
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


def decode_matrix(weight: CompressedTensor) -> torch.Tensor:
    """Decode a compressed matrix into contiguous logical ``[N, K]`` order."""
    if len(weight.shape) != 2:
        raise ValueError("decode_matrix expects a two-dimensional tensor")
    return decode_matrix_dense(weight) if weight.layout_shape else _decode(weight)


def compressed_linear(
    activations: torch.Tensor, weight: CompressedTensor,
) -> torch.Tensor:
    """Compute ``activations @ weight.T`` after one matrix-aware decode."""
    if activations.ndim != 2 or activations.shape[1] != weight.shape[1]:
        raise ValueError("activation and weight inner dimensions must match")
    return activations.contiguous() @ decode_matrix(weight).T


def compressed_matmul(
    activations: torch.Tensor, weight: CompressedTensor,
) -> torch.Tensor:
    """Compute ``activations @ weight`` after one matrix-aware decode."""
    if activations.ndim != 2 or activations.shape[1] != weight.shape[0]:
        raise ValueError("activation and weight inner dimensions must match")
    return activations.contiguous() @ decode_matrix(weight)


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
