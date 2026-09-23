"""Qwen-specific codebooks; independent of nanoGPT and MLP settings."""
from LCT.comp_format import Distribution, DistType, NoiseLevel

weight_dist = Distribution(DistType.LAPLACE, noise_level=NoiseLevel.CLEAN)
activation_dist = Distribution(DistType.EMPIRICAL, noise_level=NoiseLevel.CLEAN)
momentum_dist = Distribution(DistType.EMPIRICAL, noise_level=NoiseLevel.CLEAN, zero_prob=0.02)
