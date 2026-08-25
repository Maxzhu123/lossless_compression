"""Matrix-oriented compression and fused linear algebra operations."""

from dataclasses import replace
from functools import lru_cache

import torch
import torch.nn.functional as F
import triton

from ..code_storage import CompressedTensor, Distribution, DistType, StorageLayout
from ..codec.runtime import compress_dense, geometry
from ..huffman_tables import FIRST_MASK, get_distribution_tables
from ..kernels.matmul import (
    MATRIX_TILE, compressed_linear_kernel, decode_matrix_rhs_kernel,
)
from ..tensor_buffer import TensorBuffer


@lru_cache(maxsize=None)
def _multiprocessor_count(device: torch.device) -> int:
    """Cache the hardware parallelism used to choose the matrix backend."""
    return torch.cuda.get_device_properties(device).multi_processor_count


def _tile_weight(
    weight: torch.Tensor, tile_k: int,
) -> tuple[torch.Tensor, tuple[int, ...]]:
    """Pad and reorder ``[N, K]`` weights into codec-aligned ``[K, N]`` tiles."""
    n, k = weight.shape
    n_tiles = triton.cdiv(n, MATRIX_TILE)
    k_tiles = triton.cdiv(k, tile_k)
    padded = F.pad(weight, (0, k_tiles * tile_k - k, 0, n_tiles * MATRIX_TILE - n))
    tiled = padded.T.reshape(
        k_tiles, tile_k, n_tiles, MATRIX_TILE,
    ).permute(2, 0, 1, 3).contiguous()
    return tiled, tuple(tiled.shape)


def compress_matrix(
    weight: torch.Tensor,
    distribution: Distribution = Distribution(DistType.GAUSSIAN),
    buffer: TensorBuffer | None = None,
) -> CompressedTensor:
    """Compress a 2D BF16 weight using the existing Huffman codec in GEMM tile order."""
    if weight.ndim != 2:
        raise ValueError("compress_matrix expects a 2D weight")
    _, _, steps, _ = geometry(distribution)
    tiled, storage_shape = _tile_weight(weight, steps)
    encoded = compress_dense(
        tiled, distribution, buffer, preserve_stream_map=True,
    )
    return replace(
        encoded, shape=tuple(weight.shape), layout=StorageLayout.MATRIX_TILED,
        storage_shape=storage_shape,
    )


def compressed_linear(
    activations: torch.Tensor, weight: CompressedTensor,
) -> torch.Tensor:
    """Compute ``activations @ decompress(weight).T`` using the suitable backend."""
    if weight.layout != StorageLayout.MATRIX_TILED:
        raise ValueError("compressed_linear expects a matrix-tiled compressed weight")
    if activations.ndim != 2 or activations.shape[1] != weight.shape[1]:
        raise ValueError("activation and weight inner dimensions must match")
    activations = activations.contiguous()
    n_tiles, k_tiles, _, _ = weight.storage_shape
    multiprocessors = _multiprocessor_count(weight.data.device)
    # Fusion helps only while the direct decoder lacks enough blocks to fill the GPU
    # and all activation rows fit in one tensor-core output tile.
    if activations.shape[0] <= 16 and n_tiles * k_tiles < multiprocessors:
        return _compressed_linear_fused(activations, weight)
    return _compressed_linear_staged(activations, weight)


def _compressed_linear_fused(
    activations: torch.Tensor, weight: CompressedTensor,
) -> torch.Tensor:
    """Decode weights inside split-K tensor-core programs for tiny workloads."""
    m, k = activations.shape
    n = weight.shape[0]
    n_tiles, k_tiles, steps, _ = weight.storage_shape
    _, decode_table, rare_length = get_distribution_tables(weight.distribution)
    _, _, _, fixed_words = geometry(weight.distribution)
    streams = n_tiles * k_tiles * MATRIX_TILE
    output = torch.zeros((m, n), dtype=torch.float32, device=activations.device)
    buffered = weight.fallback_descriptor is not None
    descriptor = weight.fallback_descriptor if buffered else weight.data
    compressed_linear_kernel[
        lambda meta: (
            triton.cdiv(m, meta["BLOCK_M"]),
            triton.cdiv(n, meta["BLOCK_N"]), meta["SPLIT_K"],
        )
    ](
        activations, weight.data, weight.sign_mantissa, decode_table,
        weight.fallback_buffer, descriptor, weight.fallback_count,
        weight.matrix_fallback_starts, weight.matrix_fallback_offsets,
        output, weight.center,
        M=m, N=n, K=k, K_TILE_BLOCKS=k_tiles,
        MATRIX_STEPS=steps,
        N_STREAMS=streams, FIXED_WORDS=fixed_words,
        FALLBACK_BASE=weight.fallback_base, BUFFERED=buffered,
        FIRST_MASK=FIRST_MASK, RARE_LENGTH=rare_length,
    )
    return output.to(torch.bfloat16)


def _decode_matrix_rhs(weight: CompressedTensor) -> torch.Tensor:
    """Decode tiled weights directly into the contiguous RHS layout used by GEMM."""
    n, k = weight.shape
    n_tiles, k_tiles, steps, _ = weight.storage_shape
    _, decode_table, rare_length = get_distribution_tables(weight.distribution)
    _, _, _, fixed_words = geometry(weight.distribution)
    streams = n_tiles * k_tiles * MATRIX_TILE
    output = torch.empty((k, n), dtype=torch.bfloat16, device=weight.data.device)
    buffered = weight.fallback_descriptor is not None
    descriptor = weight.fallback_descriptor if buffered else weight.data
    decode_matrix_rhs_kernel[(n_tiles * k_tiles,)](
        weight.data, weight.sign_mantissa, decode_table,
        weight.fallback_buffer, descriptor, weight.fallback_count,
        weight.matrix_fallback_starts, weight.matrix_fallback_offsets,
        output, weight.center, N=n, K=k, K_TILE_BLOCKS=k_tiles,
        MATRIX_STEPS=steps, N_STREAMS=streams, FIXED_WORDS=fixed_words,
        FALLBACK_BASE=weight.fallback_base, BUFFERED=buffered,
        FIRST_MASK=FIRST_MASK, RARE_LENGTH=rare_length,
    )
    return output


def _compressed_linear_staged(
    activations: torch.Tensor, weight: CompressedTensor,
) -> torch.Tensor:
    """Decode weights once into GEMM order and delegate multiplication to PyTorch."""
    return activations @ _decode_matrix_rhs(weight)
