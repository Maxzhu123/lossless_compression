from dataclasses import dataclass, field, fields
from enum import Enum, auto
import math
from typing import TYPE_CHECKING
import torch

if TYPE_CHECKING:
    from .tensor_buffer import TensorBuffer


class NoiseLevel(Enum):
    CLEAN = auto()   # Default fixed-stream Huffman layout.
    MEDIUM = auto()  # Larger fixed payload for medium-noise data.
    HIGH = auto()    # Smaller blocks with extra payload for high-noise data.


class DistType(Enum):
    EMPIRICAL = "empirical"
    GAUSSIAN = "gaussian"
    LAPLACE = "laplace"
    GAMMA = "gamma"
    POLYNOMIAL = "polynomial"


class StorageLayout(Enum):
    """ If the tensor is compressed or raw format."""
    RAW = auto()
    COMPRESSED = auto()


@dataclass
class Distribution:
    """Codec distribution selector.

    ``family`` is a :class:`DistType` enum member. ``param`` is the source
    value scale: empirical-distribution scale, Gaussian standard deviation, or
    Laplace scale, and defaults to 1. ``mean`` is the source distribution's
    location and defaults to 0. ``zero_prob`` is the probability of an exact
    zero and defaults to 0. ``None`` means no extra zero point mass; the
    distribution's natural zero probability is used. The first two parameters
    are rounded to the nearest 0.25 and ``zero_prob`` to the nearest 0.05 so
    the table cache stays small.
    """

    family: DistType = DistType.EMPIRICAL
    param: float = 1.0
    mean: float = 0.0
    noise_level: NoiseLevel = NoiseLevel.CLEAN
    zero_prob: float | None = 0.0

    def __post_init__(self):
        if not isinstance(self.family, DistType):
            raise TypeError("family must be a DistType member")
        if not isinstance(self.noise_level, NoiseLevel):
            raise TypeError("noise_level must be a NoiseLevel member")
        param = float(self.param)
        mean = float(self.mean)
        zero_prob = self.zero_prob
        if zero_prob is None:
            # No explicit zero inflation requested: use the distribution's
            # natural zero probability, i.e. no added point mass.
            zero_prob = 0.0
        zero_prob = float(zero_prob)
        if not math.isfinite(param) or param <= 0.0:
            raise ValueError("distribution parameter must be finite and positive")
        if not math.isfinite(mean):
            raise ValueError("distribution mean must be finite")
        if not math.isfinite(zero_prob) or not (0.0 <= zero_prob <= 1.0):
            raise ValueError("zero_prob must be finite and between 0 and 1")
        self.param = float(max(0.25, round(param / 0.25) * 0.25))
        self.mean = float(round(mean / 0.25) * 0.25)
        self.zero_prob = round(max(0.0, min(1.0, zero_prob)), 2)

    def __str__(self) -> str:
        label = "std" if self.family == DistType.GAUSSIAN else "scale"
        return (
            f"{self.family.value}/{label}={self.param}/mean={self.mean}/"
            f"{self.noise_level.name.lower()}"
        )


@dataclass(frozen=True)
class CompressedTensor:
    data: torch.Tensor  # Fixed exponent payload or source fallback.
    size: int  # Number of elements represented by codec storage, including padding.
    sign_mantissa: torch.Tensor | None = None  # Raw sign + 7-bit mantissa.
    offsets: torch.Tensor | None = None  # Overflowing stream IDs.
    fallback_starts: torch.Tensor | None = None  # First fallback step per stream.
    fallback_offsets: torch.Tensor | None = None  # Offsets into fallback storage.
    fallback_buffer: torch.Tensor | None = None  # Shared or private fallback storage.
    fallback_descriptor: torch.Tensor | None = None  # Device allocator descriptor.
    buffer: TensorBuffer | None = None  # Allocator that owns fallback_descriptor.
    fallback_base: int = 0  # Byte offset of this tensor's region in fallback_buffer.
    fallback_count: torch.Tensor | None = None  # Device scalar with the number of fallback streams.
    fallback_used: torch.Tensor | None = None  # Device scalar with actual fallback bytes used.
    distribution: Distribution = field(
        default_factory=Distribution
    )  # Huffman table selector.
    center: torch.Tensor | int = 0  # CUDA center scalar used by encode/decode.
    shape: tuple[int, ...] = ()  # Original tensor shape.
    layout: StorageLayout = StorageLayout.RAW

    @property
    def logical_numel(self) -> int:
        """Return the number of elements exposed by the logical tensor shape."""
        return math.prod(self.shape)

    @property
    def storage_numel(self) -> int:
        """Return the element count represented by compressed storage."""
        return self.size

    def memory_size(self) -> int:
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

    def memory_buffer_size(self):
        total = 0
        if self.fallback_descriptor is not None:
            total += int(self.fallback_descriptor[1].item())
        return total


    def free(self) -> None:
        """Release this tensor's fallback allocation, if buffer-backed."""
        if self.fallback_descriptor is None or self.buffer is None:
            return
        from .tensor_buffer import Allocation

        self.buffer.free(Allocation(self.fallback_descriptor, self.buffer))
