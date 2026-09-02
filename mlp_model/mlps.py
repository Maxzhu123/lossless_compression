from typing import TYPE_CHECKING, cast

import torch
from torch.autograd import Function

from sparse_utils import MyCompressed
from compress.code_storage import CompressedTensor, Distribution, DistType, NoiseLevel

if TYPE_CHECKING:
    from compress.tensor_buffer import TensorBuffer
    from torch import Tensor

class FFN(Function):
    """Dense baseline autograd FFN for comparison.

    For ``x[B, D]``, ``W1[H, D]``, and ``W2[D, H]`` computes
    ``z = relu(x @ W1.T)`` and ``output = z @ W2.T``.
    """

    @staticmethod
    def forward(ctx, x, W1, W2, buffer: TensorBuffer=None):
        """Run the dense FFN forward pass and save tensors for backward."""
        z = x @ W1.T
        h = z.relu_()
        output = h @ W2.T

        dist = Distribution(DistType.LAPLACE, noise_level=NoiseLevel.CLEAN, param=0.75, zero_prob=0.5)

        h = MyCompressed(h, buffer=buffer, dist=dist)
        ctx.save_for_backward(x, W1, W2, h)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        """Compute dense FFN gradients from ``grad_output[B, D]``."""
        needs_x = ctx.needs_input_grad[0]

        x, W1, W2, h = ctx.saved_tensors

        h = cast(MyCompressed, h)
        comp_bytes = h.nbytes
        h = h.decompress_free()
        decomp_bytes = h.nbytes
        ratio = comp_bytes / decomp_bytes
        print(f'Ratio: {ratio:.2f}')
        if ratio > 0.76:
            torch.save(h, "test.pt")
            exit(5)

        grad_z = grad_output @ W2
        grad_W2 = grad_output.T @ h

        grad_preact = torch.ops.aten.threshold_backward.grad_input(
            grad_z, h, 0, grad_input=grad_z)

        del h, grad_z
        if not torch.compiler.is_compiling():
            ctx.maybe_clear_saved_tensors()

        if needs_x:
            grad_x = grad_preact @ W1
        else:
            grad_x = None
        grad_W1 = grad_preact.T @ x
        return grad_x, grad_W1, grad_W2, None, None
