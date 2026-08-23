from dataclasses import dataclass
from enum import Enum, IntEnum

import torch


class CompressionLayout(IntEnum):
    CLEAN = 0  # Default fixed-stream Huffman layout.
    MEDIUM = 2  # Larger fixed payload for medium-noise data.
    HIGH = 3  # Smaller blocks with extra payload for high-noise data.
    RAW = -1  # Bypass compression and retain the source tensor.


class DistributionFamily(Enum):
    STANDARD = "standard"
    GAUSSIAN = "gaussian"
    LAPLACE = "laplace"


@dataclass(frozen=True)
class Distribution:
    """Codec distribution selector.

    ``family`` is a :class:`DistributionFamily` enum member.  The optional
    ``param`` is the Gaussian standard deviation or Laplace scale.  Parameters
    are rounded to the nearest 0.25 so the table cache stays small.
    """

    family: DistributionFamily
    param: float | None = None

    def __post_init__(self):
        if not isinstance(self.family, DistributionFamily):
            raise TypeError("family must be a DistributionFamily member")
        if self.family == DistributionFamily.STANDARD:
            if self.param is not None:
                raise ValueError("standard distribution takes no parameter")
        else:
            if self.param is None:
                default = (
                    2.0
                    if self.family == DistributionFamily.GAUSSIAN
                    else 1.5
                )
                object.__setattr__(self, "param", default)
            else:
                param = float(self.param)
                if param <= 0.0:
                    raise ValueError("distribution parameter must be positive")
                param = max(0.25, round(param / 0.25) * 0.25)
                object.__setattr__(self, "param", float(param))

    @classmethod
    def standard(cls) -> "Distribution":
        return cls(DistributionFamily.STANDARD)

    @classmethod
    def gaussian(cls, std: float = 2.0) -> "Distribution":
        return cls(DistributionFamily.GAUSSIAN, std)

    @classmethod
    def laplace(cls, scale: float = 1.5) -> "Distribution":
        return cls(DistributionFamily.LAPLACE, scale)

    def __str__(self) -> str:
        if self.family == DistributionFamily.STANDARD:
            return self.family.value
        label = "std" if self.family == DistributionFamily.GAUSSIAN else "scale"
        return f"{self.family.value}/{label}={self.param}"


@dataclass(frozen=True)
class CompressedTensor:
    dtype: torch.dtype  # Original input dtype (always bfloat16 for this codec).
    data: torch.Tensor  # Fixed encoded exponent payload or raw source data.
    size: int  # Logical number of input elements.
    sign_mantissa: torch.Tensor | None = None  # Raw sign + 7-bit mantissa.
    offsets: torch.Tensor | None = None  # Overflowing stream IDs.
    fallback_starts: torch.Tensor | None = None  # First fallback step per stream.
    fallback_offsets: torch.Tensor | None = None  # Offsets into fallback_data.
    fallback_data: torch.Tensor | None = None  # Compact raw exponent suffixes.
    fallback_buffer: torch.Tensor | None = None  # Shared TensorBuffer storage for fallback data.
    fallback_base: int = 0  # Byte offset of this tensor's region in fallback_buffer.
    fallback_count: torch.Tensor | None = None  # Device scalar with the number of fallback streams.
    fallback_used: torch.Tensor | None = None  # Device scalar with actual fallback bytes used.
    layout: CompressionLayout = CompressionLayout.CLEAN  # Encoding geometry.
    distribution: Distribution = Distribution(DistributionFamily.STANDARD)  # Huffman table selector.
    center: torch.Tensor | int = 0  # CUDA center scalar used by encode/decode.
    shape: tuple[int, ...] = ()  # Original tensor shape.
