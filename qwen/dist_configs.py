"""Qwen-specific codebooks; independent of nanoGPT and MLP settings."""
from LCT.comp_format import Distribution, DistType, NoiseLevel

weight_dist = Distribution(DistType.LAPLACE, noise_level=NoiseLevel.CLEAN)
activation_dist = Distribution(DistType.EMPIRICAL, noise_level=NoiseLevel.CLEAN)
# Shared FlashAttention/output-projection cache: tuned against Qwen outputs.
attention_output_dist = Distribution(DistType.EMPIRICAL, param=0.75, noise_level=NoiseLevel.CLEAN)
# Tuned on Qwen Muon buffers; no zero point mass was observed.
momentum_dist = Distribution(DistType.EMPIRICAL, param=0.75, noise_level=NoiseLevel.CLEAN, zero_prob=0.0)
