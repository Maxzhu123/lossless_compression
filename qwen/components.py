"""Explicit LCT autograd components for the local Qwen3 implementation."""
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from LCT.LCTensor import LCTTensor
from LCT.comp_tensor import CompressedTensor
from LCT.compress import compress, decompress
from LCT.components.layers import dense_weight
from qwen.dist_configs import activation_dist, weight_dist


@dataclass
class Compression:
    weights: bool = False
    activations: bool = False
    min_elements: int = 65536
    buffer: object = None


def save_activation(x, settings):
    if settings is not None and settings.activations and x.numel() >= settings.min_elements:
        return compress(x, distribution=activation_dist, buffer=settings.buffer)
    return x.detach()


def restore_activation(value):
    if isinstance(value, CompressedTensor):
        x = decompress(value)
        value.free()
        return x
    return value


class LinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, settings, decoded_weight):
        w = dense_weight(weight) if decoded_weight is None else decoded_weight
        ctx.input = save_activation(x, settings)
        ctx.save_for_backward(weight if decoded_weight is None else decoded_weight)
        ctx.bias_dtype = bias.dtype if bias is not None else None
        return F.linear(x, w, bias)

    @staticmethod
    def backward(ctx, grad):
        x = restore_activation(ctx.input)
        ctx.input = None
        weight, = ctx.saved_tensors
        g = grad.reshape(-1, grad.shape[-1])
        dx = (g @ dense_weight(weight)).reshape(x.shape)
        dw = g.T @ x.reshape(-1, x.shape[-1])
        db = g.sum(0).to(ctx.bias_dtype) if ctx.bias_dtype is not None else None
        return dx, dw, db, None, None


class Linear(nn.Linear):
    lct = None

    def forward(self, x):
        if self.lct is None:
            return F.linear(x, self.weight, self.bias)
        return LinearFunction.apply(x, self.weight, self.bias,
                                    self.lct if torch.is_grad_enabled() else None, None)


class EmbeddingFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, indices, weight, padding_idx):
        ctx.save_for_backward(indices)
        ctx.num_embeddings = weight.shape[0]
        ctx.padding_idx = -1 if padding_idx is None else padding_idx
        return F.embedding(indices, dense_weight(weight), padding_idx)

    @staticmethod
    def backward(ctx, grad):
        indices, = ctx.saved_tensors
        return None, torch.ops.aten.embedding_dense_backward.default(
            grad, indices, ctx.num_embeddings, ctx.padding_idx, False), None


class Embedding(nn.Embedding):
    lct = None

    def forward(self, indices):
        if self.lct is None:
            return super().forward(indices)
        return EmbeddingFunction.apply(indices, self.weight, self.padding_idx)


def _rms_forward(x, gain, eps):
    # Qwen rounds to the input dtype before multiplying by the norm gain.
    rstd = (x.float().square().mean(-1, keepdim=True) + eps).rsqrt()
    return gain * (x.float() * rstd).to(x.dtype), rstd


def _rms_backward(grad, x, gain, rstd):
    xf = x.float()
    gn = (grad * gain).to(x.dtype).float()
    dx = (rstd * (gn - xf * rstd.square() * (gn * xf).mean(-1, keepdim=True))).to(x.dtype)
    normalized = (xf * rstd).to(x.dtype)
    dg = (grad * normalized).sum(tuple(range(x.ndim - 1))).to(gain.dtype)
    return dx, dg


class RMSNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gain, eps, settings):
        output, rstd = _rms_forward(x, gain, eps)
        ctx.input = save_activation(x, settings)
        ctx.save_for_backward(gain, rstd)
        return output

    @staticmethod
    def backward(ctx, grad):
        x = restore_activation(ctx.input)
        ctx.input = None
        gain, rstd = ctx.saved_tensors
        dx, dg = _rms_backward(grad, x, gain, rstd)
        return dx, dg, None, None


class RMSNormQKVFunction(torch.autograd.Function):
    """Share one RMSNorm input cache across unequal-width Q/K/V projections."""

    @staticmethod
    def forward(ctx, x, gain, eps, settings, qw, qb, kw, kb, vw, vb):
        normalized, rstd = _rms_forward(x, gain, eps)
        outputs = tuple(F.linear(normalized, dense_weight(w), b)
                        for w, b in ((qw, qb), (kw, kb), (vw, vb)))
        ctx.input = save_activation(x, settings)
        ctx.save_for_backward(gain, rstd, qw, kw, vw)
        ctx.bias_dtypes = tuple(b.dtype if b is not None else None for b in (qb, kb, vb))
        return outputs

    @staticmethod
    def backward(ctx, grad_q, grad_k, grad_v):
        x = restore_activation(ctx.input)
        ctx.input = None
        gain, rstd, qw, kw, vw = ctx.saved_tensors
        gq, gk, gv = (g.reshape(-1, g.shape[-1]) for g in (grad_q, grad_k, grad_v))
        # Match the reverse projection order of the original autograd graph.
        grad_normalized = gv @ dense_weight(vw)
        grad_normalized.add_(gk @ dense_weight(kw))
        grad_normalized.add_(gq @ dense_weight(qw))
        dx, dg = _rms_backward(grad_normalized.reshape(x.shape), x, gain, rstd)
        del grad_normalized
        normalized = (gain * (x.float() * rstd).to(x.dtype)).reshape(-1, x.shape[-1])
        gradients = []
        for g, dtype in zip((gq, gk, gv), ctx.bias_dtypes):
            gradients.extend((g.T @ normalized, g.sum(0).to(dtype) if dtype is not None else None))
        return dx, dg, None, None, *gradients


class RMSNormGateUpFunction(torch.autograd.Function):
    """Share one RMSNorm input cache across the SwiGLU gate/up projections."""

    @staticmethod
    def forward(ctx, x, gain, eps, settings, gate_weight, up_weight):
        normalized, rstd = _rms_forward(x, gain, eps)
        gate = F.linear(normalized, dense_weight(gate_weight))
        up = F.linear(normalized, dense_weight(up_weight))
        ctx.input = save_activation(x, settings)
        ctx.save_for_backward(gain, rstd, gate_weight, up_weight)
        return gate, up

    @staticmethod
    def backward(ctx, grad_gate, grad_up):
        x = restore_activation(ctx.input)
        ctx.input = None
        gain, rstd, gate_weight, up_weight = ctx.saved_tensors
        gg = grad_gate.reshape(-1, grad_gate.shape[-1])
        gu = grad_up.reshape(-1, grad_up.shape[-1])
        grad_normalized = gu @ dense_weight(up_weight)
        grad_normalized.add_(gg @ dense_weight(gate_weight))
        dx, dg = _rms_backward(grad_normalized.reshape(x.shape), x, gain, rstd)
        del grad_normalized
        normalized = (gain * (x.float() * rstd).to(x.dtype)).reshape(-1, x.shape[-1])
        return dx, dg, None, None, gg.T @ normalized, gu.T @ normalized


class SwiGLULinearFunction(torch.autograd.Function):
    """Save gate/up once; reconstruct their product for the weight gradient."""

    @staticmethod
    def forward(ctx, gate, up, weight, settings):
        ctx.gate = save_activation(gate, settings)
        ctx.up = save_activation(up, settings)
        ctx.save_for_backward(weight)
        return F.linear(F.silu(gate) * up, dense_weight(weight))

    @staticmethod
    def backward(ctx, grad):
        gate, up = restore_activation(ctx.gate), restore_activation(ctx.up)
        ctx.gate = ctx.up = None
        weight, = ctx.saved_tensors
        g = grad.reshape(-1, grad.shape[-1])
        grad_hidden = (g @ dense_weight(weight)).reshape(gate.shape)
        activated = F.silu(gate)
        grad_gate = torch.ops.aten.silu_backward.default(grad_hidden * up, gate)
        grad_up = grad_hidden * activated
        del grad_hidden
        hidden = (activated * up).reshape(-1, gate.shape[-1])
        grad_weight = g.T @ hidden
        return grad_gate, grad_up, grad_weight, None


class FlashAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, scale, dropout, settings):
        out, lse, cq, ck, mq, mk, rng, unused, _ = torch.ops.aten._scaled_dot_product_flash_attention.default(
            q, k, v, dropout, True, False, scale=scale)
        ctx.values = [save_activation(x, settings) for x in (q, k, v, out)]
        ctx.auxiliary = (lse, cq, ck, mq, mk, rng, unused)
        ctx.scale, ctx.dropout = scale, dropout
        return out

    @staticmethod
    def backward(ctx, grad):
        q, k, v, out = [restore_activation(x) for x in ctx.values]
        ctx.values = None
        lse, cq, ck, mq, mk, rng, unused = ctx.auxiliary
        dq, dk, dv = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(
            grad, q, k, v, out, lse, cq, ck, mq, mk, ctx.dropout, True, rng, unused, scale=ctx.scale)
        ctx.auxiliary = None
        return dq, dk, dv, None, None, None


def configure_compression(model, settings):
    """Call after loading/moving the HF model and before constructing optimizers."""
    if not settings.weights and not settings.activations:
        model.config.use_cache = False
        return
    packed = {}
    for module in model.modules():
        module.lct = settings
        if settings.weights and isinstance(module, (nn.Linear, nn.Embedding)):
            weight = module.weight
            if id(weight) not in packed:
                packed[id(weight)] = LCTTensor(weight, buffer=settings.buffer, dist=weight_dist)
            del module._parameters['weight']
            module.weight = packed[id(weight)]
    model.config.use_cache = False


def named_trainable_tensors(model):
    seen = set()
    for name, p in model.named_parameters():
        seen.add(id(p))
        yield name, p
    for name, module in model.named_modules():
        p = getattr(module, 'weight', None)
        if isinstance(p, LCTTensor) and id(p) not in seen:
            seen.add(id(p))
            yield f'{name}.weight', p


def dense_state_dict(model):
    state = {name: p.detach().cpu() for name, p in model.state_dict().items()}
    decoded = {}
    for name, module in model.named_modules():
        p = getattr(module, 'weight', None)
        if isinstance(p, LCTTensor):
            if id(p) not in decoded:
                decoded[id(p)] = p.decompress().detach().cpu()
            state[f'{name}.weight'] = decoded[id(p)]
    return state
