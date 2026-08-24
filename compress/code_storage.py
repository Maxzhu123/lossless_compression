from dataclasses import dataclass, fields
from enum import Enum, auto
import torch


class NoiseLevel(Enum):
    CLEAN = auto()   # Default fixed-stream Huffman layout.
    MEDIUM = auto()  # Larger fixed payload for medium-noise data.
    HIGH = auto()    # Smaller blocks with extra payload for high-noise data.


class DistType(Enum):
    EMPIRICAL = "empirical"
    GAUSSIAN = "gaussian"
    LAPLACE = "laplace"


@dataclass(frozen=True)
class Distribution:
    """Codec distribution selector.

    ``family`` is a :class:`DistType` enum member. ``param`` is the source
    value scale: empirical-distribution scale, Gaussian standard deviation, or
    Laplace scale. Parameters are rounded to the nearest 0.25 so the table
    cache stays small.
    """

    family: DistType = DistType.EMPIRICAL
    param: float | None = None
    noise_level: NoiseLevel = NoiseLevel.CLEAN

    def __post_init__(self):
        if not isinstance(self.family, DistType):
            raise TypeError("family must be a DistType member")
        if not isinstance(self.noise_level, NoiseLevel):
            raise TypeError("noise_level must be a NoiseLevel member")
        if self.param is None:
            default = (
                0.5
                if self.family == DistType.EMPIRICAL
                else 2.0
                if self.family == DistType.GAUSSIAN
                else 1.5
            )
            object.__setattr__(self, "param", default)
        else:
            param = float(self.param)
            if param <= 0.0:
                raise ValueError("distribution parameter must be positive")
            param = max(0.25, round(param / 0.25) * 0.25)
            object.__setattr__(self, "param", float(param))


    def __str__(self) -> str:
        label = "std" if self.family == DistType.GAUSSIAN else "scale"
        return f"{self.family.value}/{label}={self.param}/{self.noise_level.name.lower()}"


@dataclass(frozen=True)
class CompressedTensor:
    data: torch.Tensor  # Fixed exponent payload or source fallback.
    size: int  # Logical number of input elements.
    sign_mantissa: torch.Tensor | None = None  # Raw sign + 7-bit mantissa.
    offsets: torch.Tensor | None = None  # Overflowing stream IDs.
    fallback_starts: torch.Tensor | None = None  # First fallback step per stream.
    fallback_offsets: torch.Tensor | None = None  # Offsets into fallback storage.
    fallback_buffer: torch.Tensor | None = None  # Shared or private fallback storage.
    fallback_descriptor: torch.Tensor | None = None  # Device allocator descriptor.
    fallback_base: int = 0  # Byte offset of this tensor's region in fallback_buffer.
    fallback_count: torch.Tensor | None = None  # Device scalar with the number of fallback streams.
    fallback_used: torch.Tensor | None = None  # Device scalar with actual fallback bytes used.
    distribution: Distribution = Distribution(DistType.EMPIRICAL)  # Huffman table selector.
    center: torch.Tensor | int = 0  # CUDA center scalar used by encode/decode.
    shape: tuple[int, ...] = ()  # Original tensor shape.

    def memory_size(self):
        """Return GPU allocation bytes owned by one compressed tensor.
        """
        allocations: dict[tuple[torch.device, int], int] = {}
        for f in fields(self):
            tensor = getattr(self, f.name)
            if isinstance(tensor, torch.Tensor):
                if f.name == "fallback_buffer" and self.fallback_descriptor is not None:
                    continue
                storage = tensor.untyped_storage()
                key = (tensor.device, storage.data_ptr())
                allocations[key] = storage.nbytes()
        total = sum(allocations.values())
        if self.fallback_descriptor is not None:
            total += int(self.fallback_descriptor[1].item())
        return total
