"""Typed host operations implemented by each codec backend."""

from dataclasses import dataclass
from typing import Protocol

import torch

from ..comp_format import Distribution
from ..comp_tensor import CompressedTensor
from ..tensor_buffer import TensorBuffer


class DecodeFn(Protocol):
    def __call__(self, data: CompressedTensor) -> torch.Tensor: ...


class EncodeFn(Protocol):
    def __call__(
        self,
        data: torch.Tensor,
        distribution: Distribution,
        buffer: TensorBuffer | None = None,
        *,
        allow_raw: bool = False,
    ) -> CompressedTensor: ...


class EncodeComponentsFn(Protocol):
    def __call__(
        self,
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
    ) -> CompressedTensor: ...


@dataclass(frozen=True)
class KernelBackend:
    encode: EncodeFn
    encode_components: EncodeComponentsFn
    decode: DecodeFn
