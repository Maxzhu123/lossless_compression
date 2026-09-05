from typing import Iterable, TYPE_CHECKING
import torch
from torch import Tensor

from LCT.LCTensor import MyCompressed
from dist_configs import momentum_dist
if TYPE_CHECKING:
    from LCT.tensor_buffer import TensorBuffer


class SparseSGDM:
    def __init__(self, params: Iterable[MyCompressed | Tensor], lr, momentum=0.9,
                 buffer: TensorBuffer|None=None, compressed=True):
        for p in params:
            assert isinstance(p, (MyCompressed, Tensor)), "params must be a list of MySparse or Tensor"
        self.params = list(params)
        self.lr = lr
        self.momentum = torch.tensor(momentum, dtype=torch.float32, device="cuda")
        self.compressed = compressed
        self.buffer = buffer

        # One momentum tensor per parameter.
        self.momentums: list[Tensor | MyCompressed | None] = [None for _ in self.params]

        self.neg_lr = torch.tensor([-self.lr], dtype=torch.float32, device="cuda")

    @torch.no_grad()
    def step(self):

        for i, p in enumerate(self.params):
            g = p.grad

            # 1) Update momentum
            mom = self.momentums[i]

            if mom is None:
                # Init momentum on first step with gradient
                mom = g
                if self.compressed:
                    mom = MyCompressed(mom, buffer=self.buffer, dist=momentum_dist)
                self.momentums[i] = mom
            else:
                # Update momentum with gradient, mom = mom * self.momentum + g
                if self.compressed:
                    mom.mul_add_(self.momentum, g)
                else:
                    torch.add(g, mom, alpha=0.9, out=mom)

            # 2) Update parameter with momentum, p = p - self.lr * mom
            if self.compressed:
                if isinstance(p, MyCompressed):
                    p.add_comp_(mom, self.neg_lr)
                else:
                    update = mom.decompress() * self.neg_lr
                    p.add_(update)
            else:
                update = mom * self.neg_lr
                p.add_(update)

    def zero_grad(self):
        for p in self.params:
            p.grad = None

