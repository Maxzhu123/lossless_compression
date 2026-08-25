import torch
from matplotlib import pyplot as plt


def bf16_exponent_entropy_gaussian(std, device=None):
    """
    Analytical entropy of the BF16 exponent field for

        X ~ N(0, std^2)

    using the Gaussian CDF.

    Returns:
        entropy: scalar tensor, bits
        probs:   tensor of shape [256], probability of each exponent field
    """
    std = torch.as_tensor(std, dtype=torch.float64, device=device)

    if torch.any(std <= 0):
        raise ValueError("std must be > 0")

    # BF16 exponent bias = 127.
    #
    # exponent e = 1,...,254 corresponds to
    #
    #   2^(e-127) <= |X| < 2^(e-126)
    #
    e = torch.arange(1, 255, dtype=torch.float64, device=std.device)

    lo = torch.pow(2.0, e - 127)
    hi = torch.pow(2.0, e - 126)

    # Standard normal CDF
    normal = torch.distributions.Normal(
        torch.tensor(0.0, dtype=torch.float64, device=std.device),
        torch.tensor(1.0, dtype=torch.float64, device=std.device),
    )

    # Symmetry around zero:
    # P(lo <= |X| < hi)
    # = 2 * [Phi(hi/std) - Phi(lo/std)]
    p_normal = 2.0 * (
        normal.cdf(hi / std) -
        normal.cdf(lo / std)
    )

    probs = torch.zeros(256, dtype=torch.float64, device=std.device)

    # Exponent 0: |X| < 2^-126
    tiny = torch.tensor(2.0, dtype=torch.float64, device=std.device) ** -126
    probs[0] = 2.0 * normal.cdf(tiny / std) - 1.0

    # Normal exponent fields 1..254
    probs[1:255] = p_normal

    # Exponent 255 represents values beyond the finite BF16 normal range.
    # P(|X| >= 2^128)
    overflow = torch.tensor(2.0, dtype=torch.float64, device=std.device) ** 128
    probs[255] = 2.0 * (1.0 - normal.cdf(overflow / std))

    # Guard against tiny negative values from floating-point CDF subtraction.
    probs = probs.clamp_min(0.0)

    # Normalize in case of small floating-point error.
    probs = probs / probs.sum()

    # Shannon entropy in bits
    nz = probs > 0
    entropy = -(probs[nz] * torch.log2(probs[nz])).sum()

    return entropy, probs

entropies = []
stds = [1., 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
for std in stds:
    entropy, _ = bf16_exponent_entropy_gaussian(std)
    entropies.append(entropy)

# Plot
plt.plot(stds, entropies)
plt.show()



