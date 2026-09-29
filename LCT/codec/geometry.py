"""GPU-independent geometry of the compressed storage format."""

from ..comp_format import Distribution, DistType, NoiseLevel

BLOCK_SYMBOLS = 65536
LANES = 256
LANE_BITS = 800


def geometry(distribution: Distribution) -> tuple[int, int, int, int]:
    """Return ``(block_symbols, lanes, steps, fixed_words)`` for the codec.

    Each lane encodes ``steps`` symbols within a fixed bit budget, returned
    as a count of 32-bit words. Symbols exceeding that budget use overflow
    storage. Geometry depends on the distribution, not the GPU backend.
    """
    steps = BLOCK_SYMBOLS // LANES
    if distribution.noise_level == NoiseLevel.CLEAN:
        block_symbols = BLOCK_SYMBOLS
        lane_bits = (
            LANE_BITS
            if distribution.family != DistType.GAUSSIAN
            else LANE_BITS - 32
        )
    elif distribution.noise_level == NoiseLevel.SPARSE:
        block_symbols = BLOCK_SYMBOLS
        lane_bits = 5 * steps // 2
    elif distribution.noise_level == NoiseLevel.MEDIUM:
        block_symbols = BLOCK_SYMBOLS // 2
        steps //= 2
        lane_bits = LANE_BITS - 6 * 32
    elif distribution.noise_level == NoiseLevel.HIGH:
        block_symbols = BLOCK_SYMBOLS // 2
        steps //= 2
        lane_bits = LANE_BITS - 3 * 32
    else:
        raise ValueError(f"Unknown noise level: {distribution.noise_level}")
    return block_symbols, LANES, steps, lane_bits // 32
