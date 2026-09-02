import torch
from matplotlib import pyplot as plt

from sparse_utils import MyCompressed


h = torch.load("test.pt").cuda()

zero_frac = (h == 0).float().mean()
print(f'{zero_frac = :.2%}')
print(f'{h.nbytes // 1024 = } KB')

h_comp = MyCompressed(h, buffer=None, zero_prob=0.5)
print(f'{h_comp.nbytes // 1024 = } KB')

plt.hist(h.flatten()[:10000].cpu().float(), bins=50, density=True)
plt.show()
