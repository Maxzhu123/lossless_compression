"""Public lossless BF16 compression and decompression API."""
from typing import TYPE_CHECKING
import torch

from .comp_format import Distribution, StorageLayout
from .codec.runtime import compress_dense, decode_dense
from .codec.pointwise import (
    ADD, MULTIPLY, SCALAR_MUL_ADD,
    pointwise_compressed_dense, pointwise_scale_add_compressed,
)
from .tensor_buffer import TensorBuffer

if TYPE_CHECKING:
    from torch import Tensor
    from .comp_tensor import CompressedTensor

def compress(
    data: Tensor,
    distribution: Distribution | None = None,
    buffer: TensorBuffer | None = None,
) -> CompressedTensor:
    """ Losslessly encode the exponent byte of a CUDA bfloat16 tensor."""
    if distribution is None:
        distribution = Distribution()
    return compress_dense(data, distribution, buffer)


def decompress(data: CompressedTensor) -> Tensor:
    """ Decode a tensor produced by :func:`compress`."""
    if data.layout == StorageLayout.COMPRESSED:
        return decode_dense(data)
    return data.data.reshape(data.shape)


def A_compBT(A: Tensor, B_comp: CompressedTensor) -> Tensor:
    """ Compute A @ B_comp.T."""
    if A.ndim != 2 or A.shape[1] != B_comp.shape[1]:
        raise ValueError("activation and weight inner dimensions must match")
    return A.contiguous() @ decompress(B_comp).T


def A_compB(A: Tensor, B_comp: CompressedTensor) -> Tensor:
    """ Compute A @ B_comp."""
    if A.ndim != 2 or A.shape[1] != B_comp.shape[0]:
        raise ValueError("activation and weight inner dimensions must match")
    return A.contiguous() @ decompress(B_comp)


def compA_add_B(
    A_comp: CompressedTensor, B: Tensor,
    *,
    dense_output: bool = True, buffer: TensorBuffer | None = None, distribution=None,
) -> Tensor | CompressedTensor:
    """ Compute A + B where A is compressed."""
    return pointwise_compressed_dense(
        A_comp, B, ADD, dense_output=dense_output,
        buffer=buffer, distribution=distribution,
    )


def compA_mul_B(
    A_comp: CompressedTensor, B: Tensor,
    *,
    dense_output: bool = True, buffer: TensorBuffer | None = None, distribution=None,
) -> Tensor | CompressedTensor:
    """ Compute A * B elementwise, where A is sparse."""
    return pointwise_compressed_dense(
        A_comp, B, MULTIPLY, dense_output=dense_output,
        buffer=buffer, distribution=distribution,
    )


def a_compA_add_B(
    A_comp: CompressedTensor, a: Tensor, B: Tensor,
    *,
    dense_output: bool = True, buffer: TensorBuffer | None = None, distribution=None,
) -> Tensor | CompressedTensor:
    """ Compute ``a * A_comp + B`` with a fused pointwise operation.
        .
    """
    if not isinstance(a, torch.Tensor):
        raise TypeError("alpha must be a torch.Tensor")
    return pointwise_compressed_dense(
        A_comp, B, SCALAR_MUL_ADD, alpha=a,
        dense_output=dense_output,
        buffer=buffer, distribution=distribution,
    )


def a_compA_add_compB(
    A_comp: CompressedTensor, a: Tensor, B_comp: CompressedTensor,
    *,
    dense_output: bool = True, buffer: TensorBuffer | None = None, distribution=None,
) -> Tensor | CompressedTensor:
    """Compute ``a * A_comp + B_comp`` with both operands compressed.

    This is the fused sparse-update entry point. It uses the one-pass fused
    matrix kernel, which decodes both compressed operands directly and supports
    private and buffered fallback storage.
    """
    if not isinstance(a, torch.Tensor):
        raise TypeError("alpha must be a torch.Tensor")
    return pointwise_scale_add_compressed(
        A_comp, B_comp, a,
        dense_output=dense_output, buffer=buffer, distribution=distribution,
    )
