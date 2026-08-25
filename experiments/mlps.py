import torch
from torch import Tensor
from torch.autograd import Function


class FFN(Function):
    """Dense baseline autograd FFN for comparison.

    For ``x[B, D]``, ``W1[H, D]``, and ``W2[D, H]`` computes
    ``z = relu(x @ W1.T)`` and ``output = z @ W2.T``.
    """

    @staticmethod
    def forward(ctx, x, W1, W2, e1=None):
        """Run the dense FFN forward pass and save tensors for backward."""
        z = x @ W1.T
        z.relu_()
        output = z @ W2.T

        ctx.save_for_backward(x, W1, W2, z)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """Compute dense FFN gradients from ``grad_output[B, D]``."""
        x, W1, W2, z = ctx.saved_tensors
        needs_x = ctx.needs_input_grad[0]

        grad_z = grad_output @ W2
        grad_W2 = grad_output.T @ z

        grad_preact = torch.ops.aten.threshold_backward.grad_input(
            grad_z, z, 0, grad_input=grad_z)

        del z, grad_z
        if not torch.compiler.is_compiling():
            ctx.maybe_clear_saved_tensors()

        if needs_x:
            grad_x = grad_preact @ W1
        else:
            grad_x = None
        grad_W1 = grad_preact.T @ x
        return grad_x, grad_W1, grad_W2, None, None