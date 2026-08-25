"""Matrix-oriented compression and multiplication operations."""

from dataclasses import replace

import torch
import triton

from ..code_storage import CompressedTensor, Distribution, DistType, StorageLayout
from ..codec.runtime import compress_components, estimate_center, geometry
from ..huffman_tables import FIRST_MASK, get_distribution_tables
from ..kernels.matmul import (
    MATRIX_TILE, decode_matrix_kernel, split_matrix_components_kernel,
)
from ..tensor_buffer import TensorBuffer


def _storage_shape(weight: torch.Tensor, tile_n: int) -> tuple[int, ...]:
    """Return padded native-order codec geometry without materializing padding."""
    n, k = weight.shape
    n_tiles = triton.cdiv(n, tile_n)
    k_tiles = triton.cdiv(k, MATRIX_TILE)
    return n_tiles, k_tiles, tile_n, MATRIX_TILE


def compress_matrix(
    weight: torch.Tensor,
    distribution: Distribution = Distribution(DistType.GAUSSIAN),
    buffer: TensorBuffer | None = None,
) -> CompressedTensor:
    """Compress a 2D BF16 weight using the existing codec in matrix-tile order."""
    if weight.ndim != 2:
        raise ValueError("compress_matrix expects a 2D weight")
    weight = weight.contiguous()
    n, k = weight.shape
    _, _, steps, _ = geometry(distribution)
    storage_shape = _storage_shape(weight, steps)
    n_tiles, k_tiles, _, _ = storage_shape
    storage_size = n_tiles * k_tiles * steps * MATRIX_TILE
    exponents = torch.empty(
        storage_size, dtype=torch.uint8, device=weight.device,
    )
    sign_mantissa = torch.empty(
        storage_size, dtype=torch.uint8, device=weight.device,
    )
    split_matrix_components_kernel[(n_tiles * k_tiles,)](
        weight.view(torch.int16), exponents, sign_mantissa,
        N=n, K=k, K_TILE_BLOCKS=k_tiles, N_STEPS=steps,
    )
    center = estimate_center(
        weight.view(torch.int16), weight.numel(), precomputed=False,
    )
    encoded = compress_components(
        exponents, sign_mantissa, storage_size,
        distribution, buffer, storage_shape, precomputed=True,
        preserve_stream_map=True, center=center,
    )
    return replace(
        encoded, shape=tuple(weight.shape), layout=StorageLayout.MATRIX_TILED,
        storage_shape=storage_shape,
    )


def decode_matrix(weight: CompressedTensor) -> torch.Tensor:
    """Decode tiled weights directly into contiguous logical ``[N, K]`` order."""
    if weight.layout != StorageLayout.MATRIX_TILED:
        raise ValueError("decode_matrix expects a matrix-tiled compressed weight")
    n, k = weight.shape
    n_tiles, k_tiles, steps, _ = weight.storage_shape
    _, decode_table, rare_length = get_distribution_tables(weight.distribution)
    _, _, _, fixed_words = geometry(weight.distribution)
    streams = n_tiles * k_tiles * MATRIX_TILE
    output = torch.empty((n, k), dtype=torch.bfloat16, device=weight.data.device)
    buffered = weight.fallback_descriptor is not None
    descriptor = weight.fallback_descriptor if buffered else weight.data
    decode_matrix_kernel[(n_tiles * k_tiles,)](
        weight.data, weight.sign_mantissa, decode_table,
        weight.fallback_buffer, descriptor, weight.fallback_count,
        weight.matrix_fallback_starts, weight.matrix_fallback_offsets,
        output, weight.center, N=n, K=k, K_TILE_BLOCKS=k_tiles,
        MATRIX_STEPS=steps, N_STREAMS=streams, FIXED_WORDS=fixed_words,
        FALLBACK_BASE=weight.fallback_base, BUFFERED=buffered,
        FIRST_MASK=FIRST_MASK, RARE_LENGTH=rare_length,
    )
    return output


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
