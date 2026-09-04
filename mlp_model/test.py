import torch
from matplotlib import pyplot as plt

from sparse_utils import MyCompressed
from compress.code_storage import Distribution, DistType, NoiseLevel
from compress.compress import TensorBuffer


buffer = TensorBuffer(100_000_000)
h = torch.load("test2.pt").cuda()

zero_frac = (h == 0).float().mean()
print(f'{zero_frac = :.2%}')
print(f'{h.nbytes // 1024 = } KB')

dist = Distribution(DistType.EMPIRICAL, noise_level=NoiseLevel.CLEAN, zero_prob=0.02)
h_comp = MyCompressed(h, dist=dist, buffer=buffer)
print(f'{h_comp.nbytes // 1024 = } KB')

print(f'{h_comp.x.memory_buffer_size()}')
_, exp = torch.frexp(h.flatten())

plt.hist(exp[:10000].cpu().float(), bins=50, density=True)
# plt.xlim(-10, 5)
plt.show()

