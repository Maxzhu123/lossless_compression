"""Registry-driven pointwise operations for compressed tensors."""

from dataclasses import replace

import torch
import triton

from ..code_storage import CompressedTensor, StorageLayout
from ..codec.runtime import compress_components, compress_dense, geometry
from ..huffman_tables import FIRST_BITS, FIRST_MASK, get_distribution_tables
from ..trition_kernels import _shift_decoding_table_kernel
from ..kernels.pointwise import (
    COMPRESSED_OUTPUT,
    DENSE_OUTPUT,
    pointwise_compressed_dense_matrix_fallback_kernel,
    pointwise_compressed_dense_matrix_kernel,
)
from ..tensor_buffer import TensorBuffer
from .registry import PointwiseOp


def _launch_pointwise_compressed_dense(
    data, other, operation, output_policy, alpha=None,
):
    """Launch the fixed-stream operation, then correct sparse fallback tails."""
    other = other.contiguous().view(-1)
    if alpha is None:
        alpha = torch.tensor([1.0], dtype=torch.float32, device=data.data.device)
    elif not isinstance(alpha, torch.Tensor):
        alpha = torch.tensor([float(alpha)], dtype=torch.float32, device=data.data.device)
    _, decode_table, rare_length = get_distribution_tables(data.distribution)
    shifted_decode = torch.empty(
        1 << FIRST_BITS, dtype=torch.int32, device=data.data.device,
    )
    _shift_decoding_table_kernel[(1,)](
        decode_table, data.center, shifted_decode,
        TABLE_SIZE=1 << FIRST_BITS, BLOCK=1 << FIRST_BITS,
    )
    block_size, lanes, steps, fixed_words = geometry(data.distribution)
    blocks = triton.cdiv(data.size, block_size)
    matrix_n, matrix_k = data.layout_shape
    k_tile_blocks = data.storage_shape[1]
    if output_policy == DENSE_OUTPUT:
        output = torch.empty(
            data.logical_numel,
            dtype=torch.bfloat16, device=data.data.device,
        )
        auxiliary = output
    else:
        output = torch.empty(
            data.size, dtype=torch.uint8, device=data.data.device
        )
        auxiliary = torch.empty_like(output)

    main_args = (
        data.data, data.sign_mantissa, other, output, auxiliary,
        shifted_decode, data.size, blocks * lanes, data.center, alpha,
    )
    main_meta = dict(
        OP=operation.triton_fn, OUTPUT_POLICY=output_policy,
        FIRST_MASK=FIRST_MASK, RARE_LENGTH=rare_length,
        BLOCK=block_size, N_LANES=lanes, N_STEPS=steps,
        FIXED_WORDS=fixed_words,
    )
    pointwise_compressed_dense_matrix_kernel[(blocks,)](
        *main_args, MATRIX_N=matrix_n, MATRIX_K=matrix_k,
        MATRIX_NUMEL=data.logical_numel,
        K_TILE_BLOCKS=k_tile_blocks, **main_meta,
    )
    if data.fallback_descriptor is not None:
        metadata = data.fallback_buffer.view(torch.int32)
        fallback_args = (
            metadata, data.fallback_buffer, metadata, data.fallback_buffer, 0,
            metadata, data.fallback_descriptor, data.fallback_count,
            data.sign_mantissa, other, output, auxiliary, data.size, alpha,
        )
        fallback_meta = dict(
            OP=operation.triton_fn, OUTPUT_POLICY=output_policy,
            BUFFERED=True, TILE=64, BLOCK=block_size,
            N_LANES=lanes, N_STEPS=steps,
        )
        fallback_grid = (triton.cdiv(blocks * lanes, 64),)
        pointwise_compressed_dense_matrix_fallback_kernel[fallback_grid](
            *fallback_args, MATRIX_N=matrix_n, MATRIX_K=matrix_k,
            MATRIX_NUMEL=data.logical_numel,
            K_TILE_BLOCKS=k_tile_blocks, **fallback_meta,
        )
    elif data.offsets.numel():
        fallback_args = (
            data.offsets, data.fallback_starts, data.fallback_offsets,
            data.fallback_buffer, data.fallback_base, data.offsets,
            data.offsets, data.fallback_count, data.sign_mantissa,
            other, output, auxiliary, data.size, alpha,
        )
        fallback_meta = dict(
            OP=operation.triton_fn, OUTPUT_POLICY=output_policy,
            BUFFERED=False, TILE=64, BLOCK=block_size,
            N_LANES=lanes, N_STEPS=steps,
        )
        fallback_grid = (triton.cdiv(data.offsets.numel(), 64),)
        pointwise_compressed_dense_matrix_fallback_kernel[fallback_grid](
            *fallback_args, MATRIX_N=matrix_n, MATRIX_K=matrix_k,
            MATRIX_NUMEL=data.logical_numel,
            K_TILE_BLOCKS=k_tile_blocks, **fallback_meta,
        )
    return output, auxiliary


def pointwise_compressed_dense(
    data: CompressedTensor,
    other: torch.Tensor,
    operation: PointwiseOp,
    *,
    alpha=None,
    dense_output: bool = True,
    buffer: TensorBuffer | None = None,
    distribution=None,
) -> torch.Tensor | CompressedTensor:
    """Apply a registered compressed+dense operation.

    Dense results return directly; component results reuse the shared encoder.
    """
    if operation.arity != 2:
        raise ValueError(f"{operation.name} is not a binary operation")
    if tuple(other.shape) != data.shape:
        raise ValueError(f"{operation.name} expects tensors with the same shape")

    result_distribution = distribution or data.distribution
    if (
        data.layout == StorageLayout.BLOCKED
        and not dense_output
        and geometry(result_distribution) != geometry(data.distribution)
    ):
        raise ValueError(
            "matrix compressed output must preserve the input stream geometry"
        )

    if data.offsets is None and data.fallback_descriptor is None:
        if operation.name == "scalar_mul_add":
            if alpha is None:
                alpha = 1.0
            result = operation.torch_fn(data.data.reshape(data.shape), other, alpha)
        else:
            result = operation.torch_fn(data.data.reshape(data.shape), other)
        if dense_output:
            return result
        return compress_dense(result, result_distribution, buffer)

    policy = DENSE_OUTPUT if dense_output else COMPRESSED_OUTPUT
    values, auxiliary = _launch_pointwise_compressed_dense(
        data, other, operation, policy, alpha=alpha,
    )
    if dense_output:
        return values.reshape(data.shape)
    matrix_n, matrix_k = data.layout_shape
    k_tile_blocks = data.storage_shape[1]
    result = compress_components(
        auxiliary, values, data.size, result_distribution,
        buffer, data.shape, precomputed=True,
        matrix_shape=(matrix_n, matrix_k, data.logical_numel, k_tile_blocks),
    )
    if data.layout == StorageLayout.BLOCKED:
        result = replace(
            result, layout=data.layout, layout_shape=data.layout_shape,
            storage_shape=data.storage_shape,
        )
    return result
