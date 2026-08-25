import torch

from compress.tensor_buffer import TensorBuffer, visualize_buffer
from compress.compress import compress

G = torch.Generator(device="cuda").manual_seed(0)
x = torch.randn([4096, 21504], dtype=torch.bfloat16, device="cuda", generator=G)

buffer = TensorBuffer(10000000)

compress(x, buffer=buffer)


print(visualize_buffer(buffer))
