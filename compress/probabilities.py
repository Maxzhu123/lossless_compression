"""Exponent probabilities induced by the benchmark value distributions."""

import math

import numpy as np


def _exponent_probabilities(magnitude_cdf):
    """Map a value-magnitude CDF onto the codec's centered exponent symbols."""
    raw = np.zeros(256, dtype=np.float64)
    for exponent in range(-127, 129):
        low = 0.0 if exponent == -127 else math.ldexp(1.0, exponent)
        high = math.inf if exponent == 128 else math.ldexp(1.0, exponent + 1)
        raw[exponent & 255] = magnitude_cdf(high) - magnitude_cdf(low)
    raw /= raw.sum()

    exponents = np.arange(-127, 129)
    mean = float(np.dot(exponents, raw[exponents & 255]))
    center = math.floor(mean + 0.5) if mean >= 0.0 else -math.floor(-mean + 0.5)
    if abs(center) <= 1:
        center = 0
    probabilities = np.zeros(256, dtype=np.float64)
    for exponent in exponents:
        probabilities[(exponent - center) & 255] = raw[exponent & 255]
    return probabilities


def gaussian_probabilities(std: float = 2.0):
    """BF16 exponent probabilities for values sampled from ``N(0, std²)``."""
    scale = std * math.sqrt(2.0)

    def magnitude_cdf(value):
        return 1.0 if math.isinf(value) else math.erf(value / scale)

    return _exponent_probabilities(magnitude_cdf)


def laplace_probabilities(scale: float = 1.5):
    """BF16 exponent probabilities for values from symmetric Laplace(scale)."""

    def magnitude_cdf(value):
        return 1.0 if math.isinf(value) else -math.expm1(-value / scale)

    return _exponent_probabilities(magnitude_cdf)


def standard_probabilities(scale: float = 0.5):
    """BF16 exponent probabilities for the standard body-and-tail values."""
    tail_probability = 0.05
    tail_alpha = 2.8
    tail_start = -scale * math.log(tail_probability)

    def magnitude_cdf(value):
        if math.isinf(value):
            return 1.0
        if value <= tail_start:
            return -math.expm1(-value / scale)
        survival = tail_probability * (
            tail_start / value
        ) ** (tail_alpha - 1.0)
        return 1.0 - survival

    return _exponent_probabilities(magnitude_cdf)
