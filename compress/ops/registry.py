"""Registry of pointwise operation policies."""

from dataclasses import dataclass

import torch
import triton


@triton.jit
def add_op(left, right):
    return left + right


@triton.jit
def multiply_op(left, right):
    return left * right


@dataclass(frozen=True)
class PointwiseOp:
    name: str
    arity: int
    triton_fn: object
    torch_fn: object


ADD = PointwiseOp("add", 2, add_op, torch.add)
MULTIPLY = PointwiseOp("multiply", 2, multiply_op, torch.mul)
POINTWISE_OPS = {operation.name: operation for operation in (ADD, MULTIPLY)}
