"""BF16 trainable matrices with optional persistent LCT weight storage."""
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd.function import once_differentiable

from LCT.LCTensor import MyCompressed
from LCT.dist_configs import weight_dist


def dense_weight(weight):
    return weight.decompress() if isinstance(weight, MyCompressed) else weight


class _Linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias):
        ctx.save_for_backward(x, weight)
        ctx.bias_dtype = bias.dtype if bias is not None else None
        return F.linear(x, dense_weight(weight), bias.to(x.dtype) if bias is not None else None)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        x, weight = ctx.saved_tensors
        g = grad.reshape(-1, grad.shape[-1])
        grad_x = (g @ dense_weight(weight)).reshape(x.shape) if ctx.needs_input_grad[0] else None
        grad_weight = g.T @ x.reshape(-1, x.shape[-1]) if ctx.needs_input_grad[1] else None
        grad_bias = g.sum(0).to(ctx.bias_dtype) if ctx.bias_dtype is not None and ctx.needs_input_grad[2] else None
        return grad_x, grad_weight, grad_bias


class _RMSNormLinear(torch.autograd.Function):
    """One RMSNorm feeding one or more projections, without saving its output."""

    @staticmethod
    def forward(ctx, x, norm_weight, eps, *parameters):
        weights, biases = parameters[::2], parameters[1::2]
        gain = norm_weight.to(x.dtype)
        normalized, rstd = torch.ops.aten._fused_rms_norm.default(
            x, [x.shape[-1]], gain, eps)
        ctx.save_for_backward(x, norm_weight, rstd, *weights)
        ctx.bias_dtypes = tuple(b.dtype if b is not None else None for b in biases)
        ctx.eps = eps
        ctx.set_materialize_grads(False)
        return tuple(F.linear(normalized, dense_weight(w), b.to(x.dtype) if b is not None else None)
                     for w, b in zip(weights, biases))

    @staticmethod
    @once_differentiable
    def backward(ctx, *grad_outputs):
        x, norm_weight, rstd, *weights = ctx.saved_tensors
        gain = norm_weight.to(x.dtype)
        need_norm_grad = ctx.needs_input_grad[0] or ctx.needs_input_grad[1]
        grad_normalized = None
        # Accumulate projection gradients before the shared norm backward.
        for grad, weight in reversed(tuple(zip(grad_outputs, weights))):
            if grad is not None and need_norm_grad:
                contribution = grad.reshape(-1, grad.shape[-1]) @ dense_weight(weight)
                if grad_normalized is None:
                    grad_normalized = contribution
                else:
                    grad_normalized.add_(contribution)
                del contribution
        grad_x = grad_gain = None
        if grad_normalized is not None:
            grad_x, grad_gain = torch.ops.aten._fused_rms_norm_backward.default(
                grad_normalized.reshape(x.shape), x, [x.shape[-1]], rstd, gain,
                [ctx.needs_input_grad[0], ctx.needs_input_grad[1]])
            if grad_gain is not None:
                grad_gain = grad_gain.to(norm_weight.dtype)
        del grad_normalized

        # Recompute with the native norm to preserve its BF16 rounding and
        # learned-gain semantics; this tensor is never retained from forward.
        normalized = None
        if any(g is not None and ctx.needs_input_grad[3 + 2*i]
               for i, g in enumerate(grad_outputs)):
            normalized = F.rms_norm(x, [x.shape[-1]], gain, ctx.eps).reshape(-1, x.shape[-1])
        parameter_grads = []
        for i, grad in enumerate(grad_outputs):
            grad_weight = grad_bias = None
            if grad is not None:
                g = grad.reshape(-1, grad.shape[-1])
                if ctx.needs_input_grad[3 + 2*i]:
                    grad_weight = g.T @ normalized
                if ctx.bias_dtypes[i] is not None and ctx.needs_input_grad[4 + 2*i]:
                    grad_bias = g.sum(0).to(ctx.bias_dtypes[i])
            parameter_grads.extend((grad_weight, grad_bias))
        return grad_x, grad_gain, None, *parameter_grads


def rms_norm_linears(x, norm_weight, layers, eps=None):
    """Share a recomputed RMSNorm input across projections such as Q/K/V."""
    if not layers:
        raise ValueError("at least one linear projection is required")
    parameters = tuple(p for layer in layers for p in (layer.weight, layer.bias))
    return _RMSNormLinear.apply(x, norm_weight, eps, *parameters)


class _Embedding(torch.autograd.Function):
    @staticmethod
    def forward(ctx, indices, weight):
        ctx.save_for_backward(indices)
        ctx.num_embeddings = weight.shape[0]
        return F.embedding(indices, dense_weight(weight))

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        indices, = ctx.saved_tensors
        return None, torch.ops.aten.embedding_dense_backward(
            grad, indices, ctx.num_embeddings, -1, False)


class _CompressedWeight:
    def compress_weight(self, buffer=None):
        """Call after initialization, device placement, and checkpoint loading."""
        if isinstance(self.weight, MyCompressed):
            return
        packed = MyCompressed(self.weight, buffer=buffer, dist=weight_dist)
        del self._parameters['weight']
        self.weight = packed

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super()._save_to_state_dict(destination, prefix, keep_vars)
        if isinstance(self.weight, MyCompressed):
            destination[prefix + 'weight'] = self.weight.decompress()

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        if isinstance(self.weight, MyCompressed):
            state_dict = state_dict.copy()
            key = prefix + 'weight'
            if key not in state_dict:
                if strict:
                    missing_keys.append(key)
            else:
                value = state_dict.pop(key)
                if value.shape != self.weight.shape:
                    error_msgs.append(f'Size mismatch for {key}: {value.shape} vs {self.weight.shape}')
                else:
                    old = self.weight.x
                    value = value.to(device=self.weight.device, dtype=torch.bfloat16)
                    self.weight.x = MyCompressed(value, buffer=old.buffer, dist=old.distribution).x
                    old.free()
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)


class Linear(_CompressedWeight, nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias=bias)
        self.weight = nn.Parameter(self.weight.to(torch.bfloat16))

    def forward(self, x):
        return _Linear.apply(x, self.weight, self.bias)

    def forward_rms_norm(self, x, norm_weight, eps=None):
        return rms_norm_linears(x, norm_weight, (self,), eps)[0]


class Embedding(_CompressedWeight, nn.Embedding):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__(num_embeddings, embedding_dim, dtype=torch.bfloat16)

    def forward(self, indices):
        return _Embedding.apply(indices, self.weight)


def named_trainable_tensors(model):
    """Include compressed leaves, which cannot be wrapped in nn.Parameter."""
    yield from model.named_parameters()
    for name, module in model.named_modules():
        if isinstance(module, (Linear, Embedding)) and isinstance(module.weight, MyCompressed):
            yield (name + '.' if name else '') + 'weight', module.weight
