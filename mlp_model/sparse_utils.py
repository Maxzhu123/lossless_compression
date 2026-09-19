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


def zeropower_via_newtonschulz5(G: Tensor) -> Tensor:
    assert G.ndim >= 2
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    a, b, c = 2, -1.5, 0.5
    for _ in range(12):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


@torch.compile(fullgraph=True)
def muon_update(grad, momentum, mu=0.95, nesterov=True):
    momentum.lerp_(grad.to(momentum.dtype), 1 - mu)
    if nesterov:
        update = grad.lerp_(momentum.to(grad.dtype), mu)
    else:
        update = momentum
    update = zeropower_via_newtonschulz5(update)
    update *= max(1, grad.size(-2) / grad.size(-1))**0.5
    return update


class SparseMuon:
    """Muon with bfloat16 momentum, optionally compressed between steps."""

    def __init__(self, params: Iterable[MyCompressed | Tensor], lr=0.02, weight_decay=0, mu=0.95,
                 buffer: TensorBuffer|None=None, compressed=True):
        self.params = list(params)
        for p in self.params:
            assert isinstance(p, (MyCompressed, Tensor)) and p.ndim >= 2, "Muon requires matrix parameters"
        self.lr = lr
        self.weight_decay = weight_decay
        self.mu = mu
        self.compressed = compressed
        self.buffer = buffer

        # One momentum tensor per parameter.
        self.momentums: list[Tensor | MyCompressed | None] = [None for _ in self.params]

    @torch.no_grad()
    def step(self):
        for i, p in enumerate(self.params):
            g = p.grad
            if g is None:
                continue

            # 1) Update momentum and compute the dense Muon update.
            mom = self.momentums[i]
            if mom is None:
                mom = torch.zeros_like(g, dtype=torch.bfloat16)
            elif self.compressed:
                mom = mom.decompress_free()
            self.momentums[i] = None

            update = muon_update(g, mom, mu=self.mu)
            if self.compressed:
                mom = MyCompressed(mom, buffer=self.buffer, dist=momentum_dist)
            self.momentums[i] = mom
            del mom

            # 2) Apply weight decay and the Muon update.
            if isinstance(p, MyCompressed):
                decay = torch.tensor(
                    [1 - self.lr * self.weight_decay], dtype=torch.float32, device=p.device,
                )
                update.mul_(-self.lr)
                p.mul_add_(decay, update)
            else:
                p.mul_(1 - self.lr * self.weight_decay)
                p.add_(update, alpha=-self.lr)
            del update

    def zero_grad(self):
        for p in self.params:
            p.grad = None


Muon = SparseMuon
