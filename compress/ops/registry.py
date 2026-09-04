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
    triton_fn: object
    torch_fn: object


ADD = PointwiseOp("add", add_op, torch.add)
MULTIPLY = PointwiseOp("multiply", multiply_op, torch.mul)
SCALAR_MUL_ADD = PointwiseOp(
    "scalar_mul_add",
    None,
    lambda left, right, alpha=1.0: (
        (left.float() * alpha + right.float()).to(torch.bfloat16)
    ),
)
POINTWISE_OPS = {
    operation.name: operation
    for operation in (ADD, MULTIPLY, SCALAR_MUL_ADD)
}
