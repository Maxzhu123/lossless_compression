"""Explicit BF16 causal FlashAttention and output projection for nanoGPT."""
import torch
import torch.nn.functional as F

from LCT.compress import compress, decompress
from LCT.layers import dense_weight


class _AttentionState:
    def __init__(self, tensors, compressed, buffer, distribution):
        self.compressed = compressed
        self.values = []
        for tensor in tensors:
            if compressed:
                # The trainer supplies BHTD views of contiguous BTHD storage.
                self.values.append(compress(tensor.transpose(1, 2),
                                            distribution=distribution, buffer=buffer))
            else:
                # Detach the output to avoid a tensor -> grad_fn -> ctx cycle.
                self.values.append(tensor)

    def unpack(self):
        if self.compressed:
            return [decompress(value).transpose(1, 2) for value in self.values]
        return self.values

    def close(self):
        if self.compressed:
            for value in self.values:
                value.free()


class FlashAttention(torch.autograd.Function):
    """Native flash kernels with an explicit Q/K/V/output cache, without hooks.

    Inputs are BHTD; the projected output is BTD. Causal, scale 0.12, no dropout,
    matching the nanoGPT trainer. FP32 statistics and RNG state stay unchanged.
    """

    @staticmethod
    def forward(ctx, q, k, v, weight, bias, buffer, compressed, distribution, min_elements):
        out, lse, cq, ck, max_q, max_k, rng, unused, _ = (
            torch.ops.aten._scaled_dot_product_flash_attention.default(
                q, k, v, 0.0, True, False, scale=0.12))
        ctx.weight = weight
        ctx.bias_dtype = bias.dtype
        ctx.state = _AttentionState((q, k, v, out),
                                    compressed and q.numel() >= min_elements,
                                    buffer, distribution)
        ctx.auxiliary = (lse, cq, ck, max_q, max_k, rng, unused)
        hidden = out.transpose(1, 2).reshape(q.shape[0], q.shape[2], -1)
        return F.linear(hidden, dense_weight(weight), bias.to(out.dtype))

    @staticmethod
    def backward(ctx, grad_output):
        q, k, v, out = ctx.state.unpack()
        batch, heads, length, head_dim = q.shape
        g = grad_output.reshape(-1, grad_output.shape[-1])
        hidden = out.transpose(1, 2).reshape(-1, heads * head_dim)
        grad_weight = g.T @ hidden
        grad_bias = g.sum(0).to(ctx.bias_dtype)
        grad_attention = (g @ dense_weight(ctx.weight)).reshape(
            batch, length, heads, head_dim).transpose(1, 2)
        lse, cq, ck, max_q, max_k, rng, unused = ctx.auxiliary
        dq, dk, dv = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(
            grad_attention, q, k, v, out, lse, cq, ck, max_q, max_k,
            0.0, True, rng, unused, scale=0.12)
        ctx.state.close()
        ctx.state = ctx.auxiliary = ctx.weight = None
        return dq, dk, dv, grad_weight, grad_bias, None, None, None, None
