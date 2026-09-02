"""Registry of pointwise operation policies."""

from dataclasses import dataclass

import torch
import triton


@triton.jit
def add_op(left, right):
    """Add operands inside a specialized pointwise kernel."""
    return left + right


@triton.jit
def multiply_op(left, right):
    """Multiply operands inside a specialized pointwise kernel."""
    return left * right


@dataclass(frozen=True)
class PointwiseOp:
    """Pair an in-kernel operation with its PyTorch correctness reference."""
    name: str
    arity: int
    triton_fn: object
    torch_fn: object


ADD = PointwiseOp("add", 2, add_op, torch.add)
MULTIPLY = PointwiseOp("multiply", 2, multiply_op, torch.mul)
SCALAR_MUL_ADD = PointwiseOp(
    "scalar_mul_add",
    2,
    None,
    lambda left, right, alpha=1.0: (
        (left.float() * alpha + right.float()).to(torch.bfloat16)
    ),
)
# Keep scalar_mul_add out of the generic benchmark registry; it is exposed
# through the public API and composed from the existing multiply/add paths.
POINTWISE_OPS = {operation.name: operation for operation in (ADD, MULTIPLY)}
