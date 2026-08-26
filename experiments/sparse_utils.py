import torch
from typing import Iterable, TYPE_CHECKING
from torch import Tensor

from compress.code_storage import CompressedTensor, Distribution, DistType
from compress.compress import compress, decompress, compressed_add

if TYPE_CHECKING:
    from compress.tensor_buffer import TensorBuffer

class MyCompressed(Tensor):
    __torch_function__ = torch._C._disabled_torch_function_impl

    @staticmethod
    def __new__(cls, x, buffer):
        assert x.dtype == torch.bfloat16
        assert x.device.type == "cuda", f"x.device={x.device}"
        return torch.Tensor._make_wrapper_subclass(
            cls,
            x.shape, dtype=x.dtype, device=x.device,
            # autograd belongs to the inner representation.
            requires_grad=x.requires_grad,
        )

    def __init__(self, values, buffer: TensorBuffer):
        dist = Distribution(DistType.GAUSSIAN, 0.5)
        self.x: CompressedTensor = compress(values, buffer=buffer, distribution=dist)

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

    def __repr__(self):
        return f"MySparse({self.x})"

    def apply_update(self, update: Tensor):
        prev = self.x

        # x_dense = decompress(self.x)
        # self.x.free()
        # self.x = None
        # x_dense.add_(update)
        # self.x = compress(x_dense)

        # x_dense = compressed_add(self.x, update, dense_output=True)
        # self.x.free()
        # self.x = compress(x_dense)

        self.x = compressed_add(self.x, update,
                                dense_output=False, distribution=prev.distribution, buffer=prev.buffer)
        prev.free()
        # print(self.x.memory_buffer_size())

class SparseSGDM:
    def __init__(self, params: Iterable[MyCompressed | Tensor], lr, momentum=0.9):
        for p in params:
            assert isinstance(p, (MyCompressed, Tensor)), "params must be a list of MySparse or Tensor"
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

            if isinstance(p, MyCompressed):
                p.apply_update(update)
            else:
                p.add_(update)

    def zero_grad(self):
        for p in self.params:
            p.grad = None


def main():
    torch.manual_seed(0)
    x = torch.randn(1000, 1000, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    x_sparse = MyCompressed(x)
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
