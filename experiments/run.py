import torch

from compress.code_storage import Distribution, DistType
from compress.compress import compress, decompress

x = torch.randn(100_000_000, device="cuda", dtype=torch.bfloat16)

print(f'x: {x.nbytes//1024} KB')
dist = Distribution(DistType.GAUSSIAN, param=2.5)
compressed_x = compress(x, distribution=dist)

print(f"Compressed size: {compressed_x.memory_size() // 1024} KB")

dist = Distribution(DistType.GAUSSIAN, param=2.)
compressed_x = compress(x, distribution=dist)

print(f"Compressed size: {compressed_x.memory_size() // 1024} KB")
