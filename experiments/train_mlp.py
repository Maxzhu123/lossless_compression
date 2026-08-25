import torch
import torch.nn as nn
from torch import Tensor
from torch.autograd import Function
import torch.nn.functional as F
import time
from cprint import c_print

from compress.tensor_buffer import TensorBuffer, visualize_buffer
from sparse_utils import MyCompressed, SparseSGDM
from mlps import FFN

COMPRESSED = True
BUFFER = True
c_print(f"Compressed: {COMPRESSED}", color="bright_blue")
c_print(f"Buffer: {BUFFER}", color="bright_blue")


class FFNLayer(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, G, buffer=None):
        super().__init__()
        W1 = torch.randn(hidden_features, in_features, dtype=torch.bfloat16, device="cuda", requires_grad=True)
        W2 = torch.randn(out_features, hidden_features, dtype=torch.bfloat16, device="cuda", requires_grad=True)
        nn.init.xavier_uniform_(W1, generator=G)
        nn.init.xavier_uniform_(W2, generator=G)

        print(f'{W1.std() = }, {W2.std() = }')
        if COMPRESSED:
            self.W1 = MyCompressed(W1, buffer=buffer)
            self.W2 = MyCompressed(W2, buffer=buffer)
        else:
            self.W1 = W1
            self.W2 = W2

    def forward(self, x):
        out = FFN.apply(x, self.W1, self.W2)
        return out

    def sparse_parameters(self):
        return [self.W1, self.W2]


class Model(nn.Module):
    def __init__(self, num_layers, in_features, hidden_features, out_features, G, buffer=None):
        super().__init__()
        self.layers = nn.ModuleList(
            [FFNLayer(in_features, hidden_features, out_features, G, buffer=buffer) for _ in range(num_layers)]
        )

    def forward(self, x):
        for layer in self.layers:
            x = x + layer(F.rms_norm(x, normalized_shape=(x.shape[-1],)))
        return x

    def sparse_parameters(self):
        return [p for layer in self.layers for p in layer.sparse_parameters()]


def main():
    G = torch.Generator(device="cuda")
    G.manual_seed(0)

    if BUFFER:
        buffer = TensorBuffer(2048)
    else:
        buffer = None

    model = Model(8, 2048, 24000, 2048, G, buffer=buffer)
    optimiser = SparseSGDM(model.sparse_parameters(), lr=0.001, momentum=0.9)

    x = torch.randn(1000, 2048, dtype=torch.bfloat16, device="cuda", generator=G)
    y_hat = x.norm(dim=0)

    # Warmup
    for i in range(5):
        y = model(x)
        loss = (y - y_hat).pow(2).mean()
        loss.backward()
        optimiser.step()
        optimiser.zero_grad()

    # Main run
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    st = time.perf_counter()
    for i in range(50):
        y = model(x)
        loss = (y - y_hat).pow(2).mean()

        loss.backward()
        optimiser.step()
        optimiser.zero_grad()

        if i % 10 == 0:
            print(f'{i} loss = {loss.item():.4f}')
            c_print(visualize_buffer(buffer), color="yellow")

    torch.cuda.synchronize()
    end = time.perf_counter()

    print(y)
    print(f'Time: {end - st:.4f}s')

    print(f'Max memory: {torch.cuda.max_memory_allocated() // 1024**2} MB')


if __name__ == "__main__":
    main()

