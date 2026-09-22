"""BF16 trainable matrices with optional persistent LCT weight storage."""
import torch
from torch import nn
import torch.nn.functional as F

from LCT.LCTensor import MyCompressed
from LCT.compress import compress, decompress
from LCT.dist_configs import weight_dist, act_dist, act_relu_dist


def dense_weight(weight):
    return weight.decompress() if isinstance(weight, MyCompressed) else weight


@torch.compile(fullgraph=True)
def _restore_normalized(x, rstd, gain):
    # Match fused RMSNorm: apply the gain in FP32, then round once to BF16.
    return (x.float() * rstd * gain.float()).to(x.dtype).reshape(-1, x.shape[-1])


class _RMSNormLinear(torch.autograd.Function):
    """RMSNorm followed by one linear projection, caching only the norm input."""

    @staticmethod
    def forward(ctx, x, norm_weight, buffer, compressed, distribution, weight, bias):
        normalized, rstd = torch.ops.aten._fused_rms_norm.default(
            x, [x.shape[-1]], norm_weight.to(x.dtype), None)
        ctx.compressed = compressed
        ctx.input = compress(x.detach(), distribution=distribution, buffer=buffer) if compressed else x.detach()
        ctx.save_for_backward(norm_weight, rstd, weight)
        ctx.bias_dtype = bias.dtype
        return F.linear(normalized, dense_weight(weight), bias.to(x.dtype))

    @staticmethod
    def backward(ctx, grad):
        norm_weight, rstd, weight = ctx.saved_tensors
        x = decompress(ctx.input) if ctx.compressed else ctx.input
        gain = norm_weight.to(x.dtype)
        grad = grad.reshape(-1, grad.shape[-1])
        grad_normalized = grad @ dense_weight(weight)
        grad_x, grad_gain = torch.ops.aten._fused_rms_norm_backward.default(
            grad_normalized.reshape(x.shape), x, [x.shape[-1]], rstd, gain, [True, True])
        del grad_normalized
        normalized = _restore_normalized(x, rstd, gain)
        grad_weight = grad.T @ normalized
        grad_bias = grad.sum(0).to(ctx.bias_dtype)
        if ctx.compressed:
            ctx.input.free()
        ctx.input = None
        return grad_x, grad_gain.to(norm_weight.dtype), None, None, None, grad_weight, grad_bias


class _RMSNormQKV(torch.autograd.Function):
    """One shared RMSNorm and input cache for the three Q/K/V projections."""

    @staticmethod
    def forward(ctx, x, norm_weight, buffer, compressed, distribution,
                q_weight, q_bias, k_weight, k_bias, v_weight, v_bias):
        weights = (q_weight, k_weight, v_weight)
        biases = (q_bias, k_bias, v_bias)
        gain = norm_weight.to(x.dtype)
        normalized, rstd = torch.ops.aten._fused_rms_norm.default(
            x, [x.shape[-1]], gain, None)
        ctx.compressed = compressed
        if compressed:
            ctx.input = compress(x.detach(), distribution=distribution, buffer=buffer)
        else:
            ctx.input = x.detach()
        ctx.save_for_backward(norm_weight, rstd, *weights)
        ctx.bias_dtypes = tuple(b.dtype for b in biases)
        return tuple(F.linear(normalized, dense_weight(w), b.to(x.dtype))
                     for w, b in zip(weights, biases))

    @staticmethod
    def backward(ctx, grad_q, grad_k, grad_v):
        norm_weight, rstd, *weights = ctx.saved_tensors
        x = decompress(ctx.input) if ctx.compressed else ctx.input
        gain = norm_weight.to(x.dtype)
        grads = [grad.reshape(-1, grad.shape[-1]) for grad in (grad_q, grad_k, grad_v)]
        # Accumulate projection gradients before the shared norm backward.
        grad_normalized = grads[-1] @ dense_weight(weights[-1])
        for grad, weight in zip(grads[-2::-1], weights[-2::-1]):
            grad_normalized.add_(grad @ dense_weight(weight))
        grad_x, grad_gain = torch.ops.aten._fused_rms_norm_backward.default(
            grad_normalized.reshape(x.shape), x, [x.shape[-1]], rstd, gain, [True, True])
        grad_gain = grad_gain.to(norm_weight.dtype)
        del grad_normalized

        # Reuse the saved reciprocal RMS instead of repeating the reduction.
        normalized = _restore_normalized(x, rstd, gain)
        parameter_grads = []
        for grad, bias_dtype in zip(grads, ctx.bias_dtypes):
            parameter_grads.extend((grad.T @ normalized, grad.sum(0).to(bias_dtype)))
        if ctx.compressed:
            ctx.input.free()
        ctx.input = None
        return grad_x, grad_gain, None, None, None, *parameter_grads


def rms_norm_qkv(x, norm_weight, q, k, v, *, buffer=None, compressed=False,
                 distribution=None, min_elements=65536):
    """Normalize once and project to Q/K/V, sharing the cached input."""
    return _RMSNormQKV.apply(x, norm_weight, buffer, compressed and x.numel() >= min_elements,
                             distribution or act_dist, q.weight, q.bias, k.weight, k.bias,
                             v.weight, v.bias)


class RMSLinear:
    """Functional RMSNorm/linear with recomputation in backward."""

    @staticmethod
    def apply(x, weight, norm_weight, bias, *, buffer=None, compressed=False,
              distribution=None, min_elements=65536):
        return _RMSNormLinear.apply(x, norm_weight, buffer, compressed and x.numel() >= min_elements,
                                    distribution or act_dist, weight, bias)


class RMSNormFunction(torch.autograd.Function):
    """RMSNorm with an explicitly owned, optionally compressed input cache."""

    @staticmethod
    def forward(ctx, x, gains, buffer, compressed, distribution, min_elements):
        gain = gains.to(x.dtype) if gains is not None else None
        output, rstd = torch.ops.aten._fused_rms_norm.default(
            x, [x.shape[-1]], gain, None)
        ctx.compressed = compressed and x.numel() >= min_elements
        if ctx.compressed:
            ctx.input = compress(x.detach(), distribution=distribution or act_dist, buffer=buffer)
        else:
            ctx.input = x.detach()
        ctx.save_for_backward(gains, rstd)
        return output

    @staticmethod
    def backward(ctx, grad):
        gains, rstd = ctx.saved_tensors
        x = decompress(ctx.input) if ctx.compressed else ctx.input
        gain = gains.to(x.dtype) if gains is not None else None
        grad_x, grad_gains = torch.ops.aten._fused_rms_norm_backward.default(
            grad, x, [x.shape[-1]], rstd, gain, [True, gains is not None])
        if grad_gains is not None:
            grad_gains = grad_gains.to(gains.dtype)
        if ctx.compressed:
            ctx.input.free()
        ctx.input = None
        return grad_x, grad_gains, None, None, None, None


class _Embedding(torch.autograd.Function):
    @staticmethod
    def forward(ctx, indices, weight):
        ctx.save_for_backward(indices)
        ctx.num_embeddings = weight.shape[0]
        return F.embedding(indices, dense_weight(weight))

    @staticmethod
    def backward(ctx, grad):
        indices, = ctx.saved_tensors
        return None, torch.ops.aten.embedding_dense_backward(
            grad, indices, ctx.num_embeddings, -1, False)


class Relu2Linear(torch.autograd.Function):
    """Save ReLU(z) once; reconstruct its square for the projection gradient."""

    @staticmethod
    def forward(ctx, z, weight, bias, buffer=None, compressed=False,
                distribution=None, min_elements=65536):
        relu = z.relu()
        ctx.compressed = compressed and relu.numel() >= min_elements
        if ctx.compressed:
            ctx.relu = compress(relu, distribution=distribution or act_relu_dist, buffer=buffer)
        else:
            ctx.relu = relu
        ctx.save_for_backward(weight)
        ctx.bias_dtype = bias.dtype
        return F.linear(relu.square(), dense_weight(weight), bias.to(z.dtype))

    @staticmethod
    def backward(ctx, grad):
        weight, = ctx.saved_tensors
        relu = decompress(ctx.relu) if ctx.compressed else ctx.relu
        g = grad.reshape(-1, grad.shape[-1])
        grad_z = (g @ dense_weight(weight)).reshape(relu.shape)
        grad_z.mul_(2 * relu)
        grad_z.masked_fill_(relu == 0, 0)
        grad_weight = g.T @ relu.square().reshape(-1, relu.shape[-1])
        grad_bias = g.sum(0).to(ctx.bias_dtype)
        if ctx.compressed:
            ctx.relu.free()
        ctx.relu = None
        return grad_z, grad_weight, grad_bias, None, None, None, None


class _CompressedWeight:
    def compress_weight(self, buffer=None, distribution=None):
        """Call after initialization, device placement, and checkpoint loading."""
        if isinstance(self.weight, MyCompressed):
            return
        packed = MyCompressed(self.weight, buffer=buffer, dist=distribution or weight_dist)
        del self._parameters['weight']
        self.weight = packed


class Linear(_CompressedWeight, nn.Linear):
    """Parameter holder for the functional projection operators."""

    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features, bias=True)
        self.weight = nn.Parameter(self.weight.to(torch.bfloat16))


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
