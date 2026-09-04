from LCT.compression.format import NoiseLevel, DistType, Distribution

# Weight Distribution
weight_dist = Distribution(DistType.LAPLACE, noise_level=NoiseLevel.CLEAN)

# Momentum distribution
momentum_dist = Distribution(DistType.EMPIRICAL, noise_level=NoiseLevel.CLEAN,
                            zero_prob=0.02)

# Activation distribution
act_dist = Distribution(DistType.LAPLACE, noise_level=NoiseLevel.CLEAN,
                        zero_prob=0.5)
