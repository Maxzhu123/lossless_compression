"""Registry-driven pointwise operations for compressed tensors."""

import torch
import triton

from ..code_storage import CompressedTensor
from ..codec.runtime import compress_components, compress_dense, geometry
from ..huffman_tables import FIRST_MASK, get_distribution_tables
from ..kernels.pointwise import (
    COMPRESSED_OUTPUT,
    DENSE_OUTPUT,
    binary_compressed_dense_kernel,
    binary_fallback_kernel,
)
from ..tensor_buffer import TensorBuffer
from .registry import ADD, MULTIPLY, PointwiseOp


def _launch_binary(data, other, operation, output_policy):
    other = other.contiguous().view(-1)
    _, decode_table, rare_length = get_distribution_tables(data.distribution)
    block_size, lanes, steps, fixed_words = geometry(data.distribution)
    blocks = triton.cdiv(data.size, block_size)
    if output_policy == DENSE_OUTPUT:
        output = torch.empty(
            data.size, dtype=torch.bfloat16, device=data.data.device
        )
        auxiliary = output
    else:
        output = torch.empty(
            data.size, dtype=torch.uint8, device=data.data.device
        )
        auxiliary = torch.empty_like(output)

    binary_compressed_dense_kernel[(blocks,)](
        data.data, data.sign_mantissa, other, output, auxiliary, decode_table,
        data.size, blocks * lanes, data.center,
        OP=operation.triton_fn, OUTPUT_POLICY=output_policy,
        FIRST_MASK=FIRST_MASK, RARE_LENGTH=rare_length,
        BLOCK=block_size, N_LANES=lanes, N_STEPS=steps,
        FIXED_WORDS=fixed_words,
    )
    if data.fallback_descriptor is not None:
        metadata = data.fallback_buffer.view(torch.int32)
        binary_fallback_kernel[(triton.cdiv(blocks * lanes, 64),)](
            metadata, data.fallback_buffer, metadata, data.fallback_buffer, 0,
            metadata, data.fallback_descriptor, data.fallback_count,
            data.sign_mantissa, other, output, auxiliary, data.size,
            OP=operation.triton_fn, OUTPUT_POLICY=output_policy,
            BUFFERED=True, TILE=64, BLOCK=block_size,
            N_LANES=lanes, N_STEPS=steps,
        )
    elif data.offsets.numel():
        binary_fallback_kernel[(triton.cdiv(data.offsets.numel(), 64),)](
            data.offsets, data.fallback_starts, data.fallback_offsets,
            data.fallback_buffer, data.fallback_base, data.offsets,
            data.offsets, data.fallback_count, data.sign_mantissa,
            other, output, auxiliary, data.size,
            OP=operation.triton_fn, OUTPUT_POLICY=output_policy,
            BUFFERED=False, TILE=64, BLOCK=block_size,
            N_LANES=lanes, N_STEPS=steps,
        )
    return output, auxiliary


def binary_pointwise(
    data: CompressedTensor,
    other: torch.Tensor,
    operation: PointwiseOp,
    *,
    output: str = "dense",
    buffer: TensorBuffer | None = None,
    distribution=None,
) -> torch.Tensor | CompressedTensor:
    """Apply a registered compressed+dense binary operation."""
    if operation.arity != 2:
        raise ValueError(f"{operation.name} is not a binary operation")
    if tuple(other.shape) != data.shape:
        raise ValueError(f"{operation.name} expects tensors with the same shape")
    if output not in {"dense", "compressed"}:
        raise ValueError("output must be 'dense' or 'compressed'")

    if data.offsets is None and data.fallback_descriptor is None:
        result = operation.torch_fn(data.data.reshape(data.shape), other)
        if output == "dense":
            return result
        return compress_dense(result, distribution or data.distribution, buffer)

    policy = DENSE_OUTPUT if output == "dense" else COMPRESSED_OUTPUT
    values, auxiliary = _launch_binary(data, other, operation, policy)
    if output == "dense":
        return values.reshape(data.shape)
    return compress_components(
        auxiliary, values, data.size, distribution or data.distribution,
        buffer, data.shape, precomputed=True,
    )


def compressed_add(
    data: CompressedTensor,
    other: torch.Tensor,
    *,
    output: str = "dense",
    buffer: TensorBuffer | None = None,
    distribution=None,
) -> torch.Tensor | CompressedTensor:
    """Add dense BF16 values to a compressed tensor."""
    return binary_pointwise(
        data, other, ADD, output=output,
        buffer=buffer, distribution=distribution,
    )


def compressed_multiply(
    data: CompressedTensor,
    other: torch.Tensor,
    *,
    output: str = "dense",
    buffer: TensorBuffer | None = None,
    distribution=None,
) -> torch.Tensor | CompressedTensor:
    """Multiply compressed values by dense BF16 values."""
    return binary_pointwise(
        data, other, MULTIPLY, output=output,
        buffer=buffer, distribution=distribution,
    )
