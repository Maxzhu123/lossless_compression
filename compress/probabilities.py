import math
import numpy as np


def _normal_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _probabilities_from_frequency(frequency):
    """Expand a frequency dict into a 256-entry probability table."""
    frequency = dict(frequency)
    if "tail" not in frequency:
        frequency["tail"] = 1_500_000
    total = sum(frequency.values())
    common = [v for v in frequency if v != "tail"]
    tail = frequency["tail"]
    rare = [v for v in range(-128, 128) if v not in common]
    probs = np.zeros(256, dtype=np.float64)
    for v in common:
        probs[v & 255] = frequency[v] / total
    if rare:
        probs[[v & 255 for v in rare]] = tail / total / len(rare)
    return probs


def _make_standard_frequency():
    """Default static Huffman codebook for the benchmark body."""
    return {
        -9: 97_300,
        -8: 193_000,
        -7: 386_000,
        -6: 762_000,
        -5: 1_488_000,
        -4: 2_850_000,
        -3: 5_186_000,
        -2: 8_623_000,
        -1: 11_866_000,
        0: 11_812_000,
        1: 5_271_000,
        2: 976_000,
        3: 390_000,
    }


def standard_probabilities():
    """Probability table for the standard benchmark distribution."""
    return _probabilities_from_frequency(_make_standard_frequency())


def gaussian_probabilities(std=2.0, uniform_mix=0.01):
    """Rounded/clamped Gaussian int8 probability table.

    A small uniform mixture is included so the designed codebook remains
    robust to the localized uniform-noise cases in the benchmark.
    """
    p = np.zeros(256, dtype=np.float64)
    for v in range(-128, 128):
        lo = _normal_cdf((v - 0.5) / std)
        hi = _normal_cdf((v + 0.5) / std)
        if v == -128:
            lo = 0.0
        if v == 127:
            hi = 1.0
        p[v & 255] = hi - lo
    p /= p.sum()
    uniform = np.full(256, 1.0 / 256.0, dtype=np.float64)
    p = (1.0 - uniform_mix) * p + uniform_mix * uniform
    return p


def laplace_probabilities(scale=1.5):
    """Rounded/clamped Laplace int8 probability table."""
    p = np.zeros(256, dtype=np.float64)

    def cdf(x):
        if x < 0:
            return 0.5 * math.exp(x / scale)
        return 1.0 - 0.5 * math.exp(-x / scale)

    for v in range(-128, 128):
        lo = cdf(v - 0.5)
        hi = cdf(v + 0.5)
        if v == -128:
            lo = 0.0
        if v == 127:
            hi = 1.0
        p[v & 255] = hi - lo
    p /= p.sum()
    return p
