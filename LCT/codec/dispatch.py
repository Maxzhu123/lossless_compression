"""Typed codec entry points, dispatched on the input tensor's GPU."""

import torch

from ..comp_format import Distribution
from ..comp_tensor import CompressedTensor
from ..tensor_buffer import TensorBuffer
from .backends import backend_for


def compress_dense(
    data: torch.Tensor,
    distribution: Distribution,
    buffer: TensorBuffer | None = None,
    *,
    allow_raw: bool = False,
) -> CompressedTensor:
    """Encode BF16 values using the selected backend's complete encode pipeline."""
    backend = backend_for(data.device)
    return backend.encode(data, distribution, buffer, allow_raw=allow_raw)


def compress_components(
    source_values: torch.Tensor,
    sign_mantissa: torch.Tensor,
    size: int,
    distribution: Distribution,
    buffer: TensorBuffer | None,
    shape: tuple[int, ...],
    *,
    precomputed: bool,
    center: torch.Tensor | None = None,
    logical_numel: int,
) -> CompressedTensor:
    """Dispatch component encoding, including recompression after fused ops."""
    backend = backend_for(source_values.device)
    return backend.encode_components(
        source_values, sign_mantissa, size, distribution, buffer, shape,
        precomputed=precomputed, center=center, logical_numel=logical_numel,
    )


def decode_dense(data: CompressedTensor) -> torch.Tensor:
    """Decode using the backend selected for the user's GPU."""
    backend = backend_for(data.data.device)
    return backend.decode(data)
