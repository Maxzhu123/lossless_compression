"""Registry-driven pointwise operations for compressed tensors."""

from dataclasses import replace

import torch
import triton

from ..code_storage import CompressedTensor, StorageLayout
from ..codec.runtime import compress_components, compress_dense, decode_matrix_dense, geometry
from ..huffman_tables import FIRST_BITS, FIRST_MASK, get_distribution_tables
from ..trition_kernels import _shift_decoding_table_kernel
from ..kernels.pointwise import (
    COMPRESSED_OUTPUT,
    DENSE_OUTPUT,
    pointwise_compressed_dense_matrix_fallback_kernel,
    pointwise_compressed_dense_matrix_kernel,
)
from ..kernels.pointwise_scalar import (
    pointwise_scalar_mul_add_dense_matrix_fallback_kernel,
    pointwise_scalar_mul_add_dense_matrix_kernel,
)
from ..kernels.pointwise_scalar_dual import (
    pointwise_scalar_mul_add_compressed_compressed_fallback_kernel,
    pointwise_scalar_mul_add_compressed_compressed_kernel,
)
from ..tensor_buffer import TensorBuffer
from .registry import PointwiseOp, SCALAR_MUL_ADD


def _launch_pointwise_compressed_dense(data, other, operation, output_policy):
    """Launch the fixed-stream operation, then correct sparse fallback tails."""
    other = other.contiguous().view(-1)
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
        shifted_decode, data.size, blocks * lanes, data.center,
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
            data.sign_mantissa, other, output, auxiliary, data.size,
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
            other, output, auxiliary, data.size,
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


def _launch_scalar_mul_add_compressed_dense(
    data, other, alpha, output_policy,
):
    """Launch the dedicated fused scalar multiply-add pointwise kernels."""
    other = other.contiguous().view(-1)
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
        OUTPUT_POLICY=output_policy,
        FIRST_MASK=FIRST_MASK, RARE_LENGTH=rare_length,
        BLOCK=block_size, N_LANES=lanes, N_STEPS=steps,
        FIXED_WORDS=fixed_words,
    )
    pointwise_scalar_mul_add_dense_matrix_kernel[(blocks,)](
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
            OUTPUT_POLICY=output_policy,
            BUFFERED=True, TILE=64, BLOCK=block_size,
            N_LANES=lanes, N_STEPS=steps,
        )
        fallback_grid = (triton.cdiv(blocks * lanes, 64),)
        pointwise_scalar_mul_add_dense_matrix_fallback_kernel[fallback_grid](
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
            OUTPUT_POLICY=output_policy,
            BUFFERED=False, TILE=64, BLOCK=block_size,
            N_LANES=lanes, N_STEPS=steps,
        )
        fallback_grid = (triton.cdiv(data.offsets.numel(), 64),)
        pointwise_scalar_mul_add_dense_matrix_fallback_kernel[fallback_grid](
            *fallback_args, MATRIX_N=matrix_n, MATRIX_K=matrix_k,
            MATRIX_NUMEL=data.logical_numel,
            K_TILE_BLOCKS=k_tile_blocks, **fallback_meta,
        )
    return output, auxiliary



def _launch_scalar_mul_add_compressed_compressed(
    data, other, alpha, output_policy,
    *,
    a_fallback_buffer=None, a_fallback_base=0,
    a_fallback_starts=None, a_fallback_offsets=None,
    b_fallback_buffer=None, b_fallback_base=0,
    b_fallback_starts=None, b_fallback_offsets=None,
    job_streams=None, job_sides=None, job_count=0,
):
    """Launch the fused two-compressed scalar multiply-add kernel."""
    _, a_decode_table, rare_length_a = get_distribution_tables(data.distribution)
    _, b_decode_table, rare_length_b = get_distribution_tables(other.distribution)
    a_shifted_decode = torch.empty(
        1 << FIRST_BITS, dtype=torch.int32, device=data.data.device,
    )
    b_shifted_decode = torch.empty(
        1 << FIRST_BITS, dtype=torch.int32, device=data.data.device,
    )
    _shift_decoding_table_kernel[(1,)](
        a_decode_table, data.center, a_shifted_decode,
        TABLE_SIZE=1 << FIRST_BITS, BLOCK=1 << FIRST_BITS,
    )
    _shift_decoding_table_kernel[(1,)](
        b_decode_table, other.center, b_shifted_decode,
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
        data.data, data.sign_mantissa,
        other.data, other.sign_mantissa,
        output, auxiliary, a_shifted_decode, b_shifted_decode,
        data.size, blocks * lanes, data.center, other.center, alpha,
    )
    main_meta = dict(
        OUTPUT_POLICY=output_policy,
        FIRST_MASK=FIRST_MASK, RARE_LENGTH=rare_length_a,
        BLOCK=block_size, N_LANES=lanes, N_STEPS=steps,
        FIXED_WORDS=fixed_words,
    )
    # Both operands must use the same geometry; the b decode table may have a
    # different RARE_LENGTH if distributions differ.  The host fallback covers
    # that case, so this launcher is only called when geometries match.
    if rare_length_a != rare_length_b:
        raise ValueError("two-compressed path requires matching RARE_LENGTH")
    pointwise_scalar_mul_add_compressed_compressed_kernel[(blocks,)](
        *main_args, MATRIX_N=matrix_n, MATRIX_K=matrix_k,
        MATRIX_NUMEL=data.logical_numel,
        K_TILE_BLOCKS=k_tile_blocks, **main_meta,
    )
    if job_count:
        fallback_grid = (triton.cdiv(job_count, 64),)
        pointwise_scalar_mul_add_compressed_compressed_fallback_kernel[fallback_grid](
            job_streams, job_sides,
            data.data, data.sign_mantissa, a_shifted_decode, data.center,
            other.data, other.sign_mantissa, b_shifted_decode, other.center,
            a_fallback_buffer, a_fallback_base,
            a_fallback_starts, a_fallback_offsets,
            b_fallback_buffer, b_fallback_base,
            b_fallback_starts, b_fallback_offsets,
            output, auxiliary, alpha,
            data.size, blocks * lanes, job_count,
            MATRIX_N=matrix_n, MATRIX_K=matrix_k,
            MATRIX_NUMEL=data.logical_numel,
            K_TILE_BLOCKS=k_tile_blocks, TILE=64, **main_meta,
        )
    return output, auxiliary


def pointwise_scale_add_compressed(
    data: CompressedTensor,
    other: CompressedTensor,
    alpha: torch.Tensor,
    *,
    dense_output: bool = True,
    buffer: TensorBuffer | None = None,
    distribution=None,
) -> torch.Tensor | CompressedTensor:
    """Apply ``alpha * data + other`` where both operands are compressed.

    Uses the dedicated two-compressed kernel for private blocked tensors,
    including tensors with fallback/overflow streams.  Buffered fallback and
    non-blocked operands fall back to decoding ``other`` to dense and using the
    existing scalar multiply-add kernel.
    """
    same_layout = data.layout == StorageLayout.BLOCKED and other.layout == StorageLayout.BLOCKED
    private_storage = (
        data.offsets is not None
        and other.offsets is not None
    )
    if same_layout and private_storage and geometry(data.distribution) == geometry(other.distribution):
        block_size, lanes, steps, _ = geometry(data.distribution)
        blocks = triton.cdiv(data.size, block_size)
        streams = blocks * lanes

        def fallback_lookup(compressed):
            starts = torch.full(
                (streams,), steps, dtype=torch.int32, device=data.data.device,
            )
            offsets = torch.zeros(
                (streams,), dtype=torch.int32, device=data.data.device,
            )
            if compressed.offsets is not None and compressed.offsets.numel():
                ids = compressed.offsets.long()
                starts.index_put_((ids,), compressed.fallback_starts.to(torch.int32))
                offsets.index_put_((ids,), compressed.fallback_offsets.to(torch.int32))
            return starts, offsets

        a_count = data.offsets.numel() if data.offsets is not None else 0
        b_count = other.offsets.numel() if other.offsets is not None else 0
        if a_count or b_count:
            a_starts, a_offsets = fallback_lookup(data)
            b_starts, b_offsets = fallback_lookup(other)
            job_streams = torch.cat(
                [data.offsets.to(torch.int32), other.offsets.to(torch.int32)]
            )
            job_sides = torch.cat(
                [
                    torch.zeros(a_count, dtype=torch.int32, device=data.data.device),
                    torch.ones(b_count, dtype=torch.int32, device=data.data.device),
                ]
            )
            job_count = a_count + b_count
        else:
            a_starts = b_starts = a_offsets = b_offsets = None
            job_streams = None
            job_sides = None
            job_count = 0

        result_distribution = distribution or data.distribution
        policy = DENSE_OUTPUT if dense_output else COMPRESSED_OUTPUT
        values, auxiliary = _launch_scalar_mul_add_compressed_compressed(
            data, other, alpha, policy,
            a_fallback_buffer=data.fallback_buffer,
            a_fallback_base=data.fallback_base,
            a_fallback_starts=a_starts,
            a_fallback_offsets=a_offsets,
            b_fallback_buffer=other.fallback_buffer,
            b_fallback_base=other.fallback_base,
            b_fallback_starts=b_starts,
            b_fallback_offsets=b_offsets,
            job_streams=job_streams,
            job_sides=job_sides,
            job_count=job_count,
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

    if other.layout == StorageLayout.RAW:
        other_dense = other.data.reshape(other.shape)
    else:
        other_dense = decode_matrix_dense(other)
    return pointwise_compressed_dense(
        data, other_dense, SCALAR_MUL_ADD, alpha=alpha,
        dense_output=dense_output, buffer=buffer, distribution=distribution,
    )


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
    if operation.name == "scalar_mul_add" and alpha is None:
        alpha = torch.tensor(
            [1.0], dtype=torch.float32, device=data.data.device,
        )
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
            result = operation.torch_fn(data.data.reshape(data.shape), other, alpha)
        else:
            result = operation.torch_fn(data.data.reshape(data.shape), other)
        if dense_output:
            return result
        return compress_dense(result, result_distribution, buffer)

    policy = DENSE_OUTPUT if dense_output else COMPRESSED_OUTPUT
    if operation.name == "scalar_mul_add":
        values, auxiliary = _launch_scalar_mul_add_compressed_dense(
            data, other, alpha, policy,
        )
    else:
        values, auxiliary = _launch_pointwise_compressed_dense(
            data, other, operation, policy,
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
