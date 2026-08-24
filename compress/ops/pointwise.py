"""Public pointwise operations for compressed tensors."""

import torch

from ..code_storage import CompressedTensor
from ..compress import (
    _compress_components,
    _decode,
    _decode_add_components,
    _uses_raw_source,
    compress,
)
from ..tensor_buffer import TensorBuffer


def compressed_add(
    data: CompressedTensor,
    other: torch.Tensor,
    *,
    output: str = "dense",
    buffer: TensorBuffer | None = None,
    distribution=None,
) -> torch.Tensor | CompressedTensor:
    """Add while decoding and return either dense or compressed output."""
    if output == "dense":
        return _decode(data, other)
    if output != "compressed":
        raise ValueError("output must be 'dense' or 'compressed'")

    distribution = distribution or data.distribution
    if _uses_raw_source(data.size, distribution):
        return compress(_decode(data, other), distribution, buffer)

    exponents, sign_mantissa = _decode_add_components(data, other)
    return _compress_components(
        exponents, sign_mantissa, data.size, distribution, buffer, data.shape,
        precomputed=True,
    )
