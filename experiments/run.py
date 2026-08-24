import torch

from compress.code_storage import Distribution, DistType
from compress.compress import compress, decompress

x = torch.randn(100_000_000, device="cuda", dtype=torch.bfloat16)
print(f'x: {x.nbytes // 1024} KB')

for std in [0.5, 1., 1.5, 2., 2.5, 3., 3.5]:
    dist = Distribution(DistType.GAUSSIAN, param=std)
    compressed_x = compress(x, distribution=dist)
    print(f"{std} compression size: {compressed_x.memory_size() // 1024} KB")
