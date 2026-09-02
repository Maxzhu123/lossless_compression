import math
import torch
from matplotlib import pyplot as plt

from sparse_utils import MyCompressed
from compress.code_storage import Distribution, DistType, NoiseLevel
from mlp_model.dist_visualise import sample_distribution


h = torch.load("test.pt").cuda()

zero_frac = (h == 0).float().mean()
print(f'{zero_frac = :.2%}')
print(f'{h.nbytes // 1024 = } KB')

dist = Distribution(DistType.GAMMA, noise_level=NoiseLevel.CLEAN, zero_prob=0.5)
h_comp = MyCompressed(h, dist=dist, buffer=None)
print(f'{h_comp.nbytes // 1024 = } KB')

# Gamma visualisation distribution: shape=param, scale=mean.
gamma_dist = Distribution(DistType.GAMMA, param=1., mean=2., zero_prob=0.5)

def frexp_center(t: torch.Tensor) -> int:
    bits = t.view(torch.int16)
    bf16_exp = ((bits >> 7) & 0xFF).flatten().to(torch.int32) - 127
    nz = bf16_exp[bf16_exp != -127]
    mean = nz.float().mean().item()
    center = math.floor(mean + 0.5) if mean >= 0 else -math.floor(-mean + 0.5)
    # frexp exponent = BF16 exponent + 1
    return center + 1

# Observed frexp exponents for non-zero values, centered by the codec center.
_, exp = torch.frexp(h)
exp = exp[h != 0] - frexp_center(h)

# Sample the Gamma codebook distribution and use frexp directly on the sampled values.
expected = sample_distribution(gamma_dist, 10000)
_, expected_exp = torch.frexp(expected)
print(expected_exp)
expected_exp = expected_exp[expected != 0] - frexp_center(expected)

plt.hist(exp[:10000].cpu().float(), bins=50, density=True, label='observed',
         color='tab:blue', alpha=0.6)
plt.hist(expected_exp.cpu().float(), bins=50, density=True,
         label='gamma expected', color='tab:orange', alpha=0.6)
plt.xlim(-10, 5)
plt.legend()
plt.show()
