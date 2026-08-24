import torch
import torch.nn as nn
from torch import Tensor
from torch.autograd import Function

from compress.code_storage import CompressedTensor
from sparse_tests import MySparse


class LinearFunction(Function):
    @staticmethod
    def forward(ctx, x, weight: MySparse | Tensor):
        # x:      (batch, in_features)
        # weight: (out_features, in_features)

        ctx.save_for_backward(x, weight)

        return x @ weight.T

    @staticmethod
    def backward(ctx, grad_output):
        x, weight = ctx.saved_tensors

        grad_x = grad_output @ weight
        grad_weight = grad_output.T @ x

        return grad_x, grad_weight


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super(MLP, self).__init__()
        W1 = torch.randn(hidden_features, in_features)
        W2 = torch.randn(out_features, hidden_features)
        self.W1 = torch.nn.Parameter(W1)
        self.W2 = torch.nn.Parameter(W2)

        nn.init.xavier_uniform_(self.W1)
        nn.init.xavier_uniform_(self.W2)


    def forward(self, x):
        x = LinearFunction.apply(x, self.W1)
        x = LinearFunction.apply(x, self.W2)
        return x


def main():
    model = MLP(100, 200, 100)
    x = torch.randn(100, 100)
    y = model(x)

    print(y)


if __name__ == "__main__":
    main()

