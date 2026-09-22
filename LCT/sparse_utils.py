from typing import Iterable, TYPE_CHECKING
import math
import torch
from torch import Tensor

from LCT.LCTensor import MyCompressed
from LCT.compress import a_compA_add_B
from LCT.dist_configs import momentum_dist
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
                    # Decode and apply momentum in the fused kernel, keeping
                    # only a BF16 result temporary and preserving p's identity.
                    p.copy_(a_compA_add_B(mom.x, self.neg_lr, p))
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

        self.neg_lr = torch.tensor([-self.lr], dtype=torch.float32, device="cuda")

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
                p.mul_add_(decay, update, beta=self.neg_lr, alpha_is_one=self.weight_decay == 0)
            else:
                p.mul_(1 - self.lr * self.weight_decay)
                p.add_(update, alpha=-self.lr)
            del update

    def zero_grad(self):
        for p in self.params:
            p.grad = None


class SparseAdamW:
    """AdamW with optional lossless BF16 moment storage and LCT matrix weights.

    FP32 scalar/vector parameters and their moments remain dense. Dense and
    compressed modes use the same update arithmetic and state precision.
    """

    def __init__(self, param_groups, *, betas=(0.8, 0.95), eps=1e-10,
                 weight_decay=0, compressed=True, buffer=None):
        self.param_groups = [dict(group, params=list(group['params'])) for group in param_groups]
        self.betas, self.eps, self.weight_decay = betas, eps, weight_decay
        self.compressed, self.buffer = compressed, buffer
        self.state = {}

    @torch.no_grad()
    def step(self):
        beta1, beta2 = self.betas
        for group in self.param_groups:
            lr = group['lr']
            for p in group['params']:
                grad = p.grad
                if grad is None:
                    continue
                state = self.state.setdefault(p, {'step': 0})
                state['step'] += 1
                moments = []
                for key in ('exp_avg', 'exp_avg_sq'):
                    value = state.pop(key, None)
                    if value is None:
                        value = torch.zeros_like(grad)
                    elif isinstance(value, MyCompressed):
                        value = value.decompress_free()
                    moments.append(value)
                avg, square_avg = moments
                avg.lerp_(grad, 1 - beta1)
                square_avg.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                denom = square_avg.sqrt().div_(math.sqrt(1 - beta2 ** state['step'])).add_(self.eps)
                weight = p.decompress() if isinstance(p, MyCompressed) else p
                weight.mul_(1 - lr * self.weight_decay)
                weight.addcdiv_(avg, denom, value=-lr / (1 - beta1 ** state['step']))
                if isinstance(p, MyCompressed):
                    old = p.x
                    p.x = MyCompressed(weight, buffer=old.buffer, dist=old.distribution).x
                    old.free()
                for key, value in zip(('exp_avg', 'exp_avg_sq'), moments):
                    state[key] = (MyCompressed(value, buffer=self.buffer, dist=momentum_dist)
                                  if self.compressed and value.dtype == torch.bfloat16 else value)

    def zero_grad(self):
        for group in self.param_groups:
            for p in group['params']:
                p.grad = None
