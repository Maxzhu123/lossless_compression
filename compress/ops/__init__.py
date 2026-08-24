"""Operations consuming compressed tensors."""

from .pointwise import (
    compressed_add,
    compressed_multiply,
    pointwise_compressed_dense,
)
from .registry import POINTWISE_OPS

__all__ = [
    "POINTWISE_OPS",
    "pointwise_compressed_dense",
    "compressed_add",
    "compressed_multiply",
]
