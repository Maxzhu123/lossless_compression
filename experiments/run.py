import torch

from compress.code_storage import Distribution
from compress.compress import compress, decompress

x = torch.randn(10_000_000, device="cuda", dtype=torch.bfloat16)

