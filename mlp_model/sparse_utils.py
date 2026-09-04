import torch
from typing import Iterable, TYPE_CHECKING
from torch import Tensor
import math

from compress.comp_tensor import CompressedTensor
from compress.compression.format import NoiseLevel, DistType, Distribution
from compress.compress import (
    compress,
    compA_add_B,
    a_compA_add_compB,
    a_compA_add_B,
    decompress,
)
from dist_configs import momentum_dist
if TYPE_CHECKING:
    from compress.tensor_buffer import TensorBuffer


class MyCompressed(Tensor):
    __torch_function__ = torch._C._disabled_torch_function_impl
    x: CompressedTensor

    @staticmethod
    def __new__(cls, x, buffer: TensorBuffer|None, dist: Distribution):
        assert x.dtype == torch.bfloat16
        assert x.device.type == "cuda", f"x.device={x.device}"
        return torch.Tensor._make_wrapper_subclass(
            cls,
            x.shape, dtype=x.dtype, device=x.device,
            # autograd belongs to the inner representation.
            requires_grad=x.requires_grad,
        )

    def __init__(self, x: Tensor, buffer: TensorBuffer|None, dist: Distribution):
        self.x: CompressedTensor = compress(x, buffer=buffer, distribution=dist)

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func is torch.ops.aten.t.default:
            x = args[0]
            return decompress(x.x).T
        if func is torch.ops.aten.permute.default:
            x = args[0]
            return decompress(x.x).T

        if func is torch.ops.aten.mm.default:
            a, b = args
            if isinstance(a, MyCompressed):
                a = decompress(a.x)
            if isinstance(b, MyCompressed):
                b = decompress(b.x)
            return a @ b
        raise NotImplementedError(f"{func} not implemented for MySparse")

    @property
    def nbytes(self):
        return self.x.memory_size()

    @property
    def dense_nbytes(self):
        return math.prod(self.x.shape) * 2 # bfloat16 is 2 bytes

    def __repr__(self):
        return f"MySparse({self.x})"

    def add_(self, update: Tensor):
        """ Inplace add,
            x <- x + update
        """
        prev = self.x
        self.x = compA_add_B(self.x, update,
                             dense_output=False, distribution=prev.distribution, buffer=prev.buffer)
        prev.free()

    def mul_add_(self, alpha: Tensor, update: Tensor):
        """ Inplace multiply-add,
            x <- alpha * x + update
        """
        prev = self.x
        self.x = a_compA_add_B(prev, alpha, update,
                               dense_output=False, distribution=prev.distribution, buffer=prev.buffer)
        prev.free()

    def add_comp_(self, update_comp: MyCompressed, alpha: Tensor):
        """ Inplace add with another compressed tensor,
            x <- x + alpha * update_comp
        """
        prev = self.x
        self.x = a_compA_add_compB(update_comp.x, alpha, prev,
            dense_output=False, buffer=prev.buffer, distribution=prev.distribution,
        )
        prev.free()

    def decompress(self) -> Tensor:
        return decompress(self.x)

    def decompress_free(self) -> Tensor:
        """ Decompresses the compressed tensor and frees the compressed representation."""
        x_dense = self.decompress()
        self.x.free()
        return x_dense

    def free(self):
        self.x.free()


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
        self.momentums: list[Tensor|MyCompressed|None] = [None for _ in self.params]

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


def main():
    torch.manual_seed(0)
    x = torch.randn(1000, 1000, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    x_sparse = MyCompressed(x, None)
    x_sparse.requires_grad_(True)

    optimiser = SparseSGDM([x_sparse], lr=100000)

    print(f'{x.nbytes = }')
    print(f'{x_sparse.nbytes = }')

    y = x @ x_sparse #@ x
    y = y.mean()
    y.backward()#

    print(f'{decompress(x_sparse.x) = }')
    optimiser.step()
    print(f'{decompress(x_sparse.x) = }')


if __name__ == "__main__":
    main()
