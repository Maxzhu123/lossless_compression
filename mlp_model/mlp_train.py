import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from pathlib import Path
from cprint import c_print
from torchvision.datasets import MNIST

from LCT.tensor_buffer import TensorBuffer, visualize_buffer
from LCT.LCTensor import MyCompressed
from LCT.sparse_utils import SparseSGDM, SparseMuon
from LCT.mlps import RMSFFN
from LCT.dist_configs import weight_dist

# Independent compression options; optimiser compression applies to momentum state.
COMPRESS_WEIGHTS = False
COMPRESS_ACTIVATIONS = False
COMPRESS_OPTIMISER = False
BUFFER = False
USE_MUON = False
BATCH_SIZE = 12000
EPOCHS = 6
IN_FEATURES = 4096
c_print(f"Compress weights: {COMPRESS_WEIGHTS}", color="bright_blue")
c_print(f"Compress activations: {COMPRESS_ACTIVATIONS}", color="bright_blue")
c_print(f"Compress optimiser: {COMPRESS_OPTIMISER}", color="bright_blue")
c_print(f"Buffer: {BUFFER}", color="bright_blue")


class FFNLayer(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, G, buffer=None):
        super().__init__()
        W1 = torch.randn(hidden_features, in_features, dtype=torch.bfloat16, device="cuda", requires_grad=True)
        W2 = torch.randn(out_features, hidden_features, dtype=torch.bfloat16, device="cuda", requires_grad=True)
        nn.init.xavier_uniform_(W1, generator=G)
        nn.init.xavier_uniform_(W2, generator=G)

        if COMPRESS_WEIGHTS:
            self.W1 = MyCompressed(W1, buffer=buffer, dist=weight_dist)
            self.W2 = MyCompressed(W2, buffer=buffer, dist=weight_dist)
        else:
            self.W1 = W1
            self.W2 = W2

    def forward(self, x, buffer:TensorBuffer):
        return RMSFFN.apply(x, self.W1, self.W2, buffer, COMPRESS_ACTIVATIONS)

    def sparse_parameters(self):
        return [self.W1, self.W2]


class Model(nn.Module):
    def __init__(self, num_layers, in_features, hidden_features, out_features, G, buffer=None):
        super().__init__()
        self.layers = nn.ModuleList(
            [FFNLayer(in_features, hidden_features, out_features, G, buffer=buffer) for _ in range(num_layers)]
        )

    def forward(self, x, buffer: TensorBuffer):
        for layer in self.layers:
            x = x + layer(x, buffer=buffer)
        return x

    def sparse_parameters(self):
        return [p for layer in self.layers for p in layer.sparse_parameters()]


def make_mnist_loader(batch_size, in_features, generator):
    """Load MNIST onto the GPU once and return a shuffled batch iterator factory."""
    train_dataset = MNIST(
        root=Path(__file__).resolve().parents[1] / "artefacts" / "data",
        train=True,
        download=True,
    )
    # MNIST fits on the GPU: normalize once and fetch whole batches without
    # per-image PIL conversion, collation, or repeated host-to-device copies.
    train_images = train_dataset.data.flatten(1).to(device="cuda", dtype=torch.float32)
    train_images = train_images.div_(255).to(dtype=torch.bfloat16)
    train_labels = train_dataset.targets.to(device="cuda")
    del train_dataset

    def train_batches():
        indices = torch.randperm(train_labels.size(0), device="cuda", generator=generator)
        for batch_indices in indices.split(batch_size):
            images = train_images[batch_indices]
            # Keep the benchmark's model width while accepting 28x28 MNIST images.
            images = F.pad(images, (0, in_features - images.shape[1]))
            yield images, train_labels[batch_indices]

    return train_batches


def main():
    G = torch.Generator(device="cuda")
    G.manual_seed(0)

    if BUFFER and (COMPRESS_WEIGHTS or COMPRESS_ACTIVATIONS or COMPRESS_OPTIMISER):
        buffer = TensorBuffer(500_000_000)
    else:
        buffer = None

    train_batches = make_mnist_loader(BATCH_SIZE, IN_FEATURES, G)

    model = Model(8, IN_FEATURES, 21504, IN_FEATURES, G, buffer=buffer)
    if USE_MUON:
        optimiser = SparseMuon(model.sparse_parameters(), lr=0.02, mu=0.95,
                               buffer=buffer, compressed=COMPRESS_OPTIMISER)
    else:
        optimiser = SparseSGDM(model.sparse_parameters(), lr=0.0005, momentum=0.9,
                               buffer=buffer, compressed=COMPRESS_OPTIMISER)
    c_print(f"Optimiser: {type(optimiser).__name__}", color="bright_blue")

    # Warmup
    x, labels = next(train_batches())
    model.train()
    for i in range(5):
        # The first ten output features serve as logits for digits 0 through 9.
        logits = model(x, buffer=buffer)[:, :10].float().contiguous()
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimiser.step()
        optimiser.zero_grad()
    del x, labels, logits, loss

    # Main run
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    st = time.perf_counter()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        total_correct = 0
        total_examples = 0
        for x, labels in train_batches():
            logits = model(x, buffer=buffer)[:, :10].float().contiguous()
            loss = F.cross_entropy(logits, labels)

            loss.backward()
            optimiser.step()
            optimiser.zero_grad()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_examples += batch_size

        print(
            f'Epoch {epoch + 1}/{EPOCHS} '
            f'train loss = {total_loss / total_examples:.4f}, '
            f'train accuracy = {total_correct / total_examples:.2%}'
        )
        if buffer is not None and epoch % 10 == 0:
            c_print(visualize_buffer(buffer), color="yellow")

    torch.cuda.synchronize()
    end = time.perf_counter()

    print(f'Time: {end - st:.4f}s')

    print(f'Max memory: {torch.cuda.max_memory_allocated() // 1024**2} MB')


if __name__ == "__main__":
    main()
