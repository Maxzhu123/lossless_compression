"""Public lossless BF16 compression and decompression API."""

import torch

from .code_storage import CompressedTensor, Distribution, StorageLayout
from .codec.runtime import compress_dense, decode_matrix_dense
from .ops.pointwise import pointwise_compressed_dense
from .ops.registry import ADD, MULTIPLY, SCALAR_MUL_ADD
from .tensor_buffer import TensorBuffer


def compress(
    data: torch.Tensor,
    distribution: Distribution | None = None,
    buffer: TensorBuffer | None = None,
) -> CompressedTensor:
    """Losslessly encode the exponent byte of a CUDA bfloat16 tensor."""
    if distribution is None:
        distribution = Distribution()
    return compress_dense(data, distribution, buffer)


def decompress(data: CompressedTensor) -> torch.Tensor:
    """Decode a tensor produced by :func:`compress`."""
    if data.layout == StorageLayout.BLOCKED:
        return decode_matrix_dense(data)
    return data.data.reshape(data.shape)


def compressed_linear(
    activations: torch.Tensor, weight: CompressedTensor,
) -> torch.Tensor:
    """Compute ``activations @ weight.T`` after one matrix-aware decode."""
    if activations.ndim != 2 or activations.shape[1] != weight.shape[1]:
        raise ValueError("activation and weight inner dimensions must match")
    return activations.contiguous() @ decompress(weight).T


def compressed_matmul(
    activations: torch.Tensor, weight: CompressedTensor,
) -> torch.Tensor:
    """Compute ``activations @ weight`` after one matrix-aware decode."""
    if activations.ndim != 2 or activations.shape[1] != weight.shape[0]:
        raise ValueError("activation and weight inner dimensions must match")
    return activations.contiguous() @ decompress(weight)


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


def compressed_scalar_mul_add(
    data: CompressedTensor,
    alpha: float,
    other: torch.Tensor,
    *,
    dense_output: bool = True,
    buffer: TensorBuffer | None = None,
    distribution=None,
) -> torch.Tensor | CompressedTensor:
    """Compute ``alpha * data + other`` as a fused pointwise operation."""
    return pointwise_compressed_dense(
        data, other, SCALAR_MUL_ADD, alpha=alpha,
        dense_output=dense_output,
        buffer=buffer, distribution=distribution,
    )
