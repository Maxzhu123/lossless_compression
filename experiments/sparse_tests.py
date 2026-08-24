import torch
from torch import optim
from typing import Iterable

from compress.code_storage import CompressedTensor
from compress.compress import compress, decompress


class MySparse(torch.Tensor):
    __torch_function__ = torch._C._disabled_torch_function_impl

    @staticmethod
    def __new__(cls, x):
        assert x.dtype == torch.bfloat16
        assert x.device.type == "cuda", f"x.device={x.device}"
        return torch.Tensor._make_wrapper_subclass(
            cls,
            x.shape, dtype=x.dtype, device=x.device,
            # autograd belongs to the inner representation.
            requires_grad=False,
        )

    def __init__(self, values):
        self.x: CompressedTensor = compress(values)

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func is torch.ops.aten.t.default:
            x = args[0]
            return decompress(x.x).T

        if func is torch.ops.aten.mm.default:
            a, b = args
            if isinstance(a, MySparse):
                a = decompress(a.x)
            if isinstance(b, MySparse):
                b = decompress(b.x)
            return a @ b
        raise NotImplementedError(f"{func} not implemented for MySparse")

    @property
    def nbytes(self):
        return self.x.memory_size()

    def __repr__(self):
        return f"MySparse({self.x})"

    def apply_update(self, update: torch.Tensor):
        print("apply_update")


class SparseSGDM:
    def __init__(self, params: Iterable[MySparse], lr, momentum=0.9):
        self.params = list(params)
        self.lr = lr
        self.momentum = momentum

        # One momentum tensor per parameter.
        self.buffers = [None] * len(self.params)

    @torch.no_grad()
    def step(self):

        for i, p in enumerate(self.params):
            g = p.grad

            # First step matches standard PyTorch SGD momentum behavior:
            # buffer = grad
            if self.buffers[i] is None:
                buf = g.clone()
            else:
                buf = self.buffers[i]
                buf = buf * self.momentum + g

            self.buffers[i] = buf

            # This is the tensor that would normally be added to p.
            update = buf * (-self.lr)

            p.apply_update(update)

    def zero_grad(self):
        for p in self.params:
            p.grad = None


def main():
    torch.manual_seed(0)
    x = torch.randn(1000, 1000, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    x_sparse = MySparse(x)
    x_sparse.requires_grad_(True)

    optimiser = SparseSGDM([x_sparse], lr=0.01)

    print(f'{x.nbytes = }')
    print(f'{x_sparse.nbytes = }')

    y = x @ x_sparse #@ x
    y = y.mean()
    y.backward()
    print(f'{x.grad.shape = }')
    print(f'{x_sparse.grad.shape = }')

    optimiser.step()

if __name__ == "__main__":
    main()
