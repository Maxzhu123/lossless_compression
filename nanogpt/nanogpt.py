"""NanoGPT model definition."""
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gains = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.rms_norm(x, (x.size(-1),), weight=self.gains.type_as(x))


class Linear(nn.Linear):
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features, bias=True)

    def forward(self, x):
        return F.linear(x, self.weight.type_as(x), self.bias.type_as(x))


class _RMSLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, norm_weight):
        normalized, rstd = torch.ops.aten._fused_rms_norm.default(
            x, [x.shape[-1]], norm_weight, None,
        )
        output = F.linear(normalized, weight, bias)
        ctx.save_for_backward(x, weight, norm_weight, rstd)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, norm_weight, rstd = ctx.saved_tensors
        if not torch.compiler.is_compiling():
            ctx.maybe_clear_saved_tensors()

        grad_output_flat = grad_output.reshape(-1, weight.shape[0])
        grad_x = grad_weight = grad_bias = grad_norm_weight = None
        if ctx.needs_input_grad[2]:
            grad_bias = grad_output_flat.sum(dim=0)
        if ctx.needs_input_grad[1]:
            # Recompute only for the weight gradient. The fused kernel avoids
            # full-size float32 temporaries and preserves RMSNorm's rounding.
            normalized, _ = torch.ops.aten._fused_rms_norm.default(
                x, [x.shape[-1]], norm_weight, None,
            )
            grad_weight = grad_output_flat.T @ normalized.reshape(-1, x.shape[-1])
            del normalized
        if ctx.needs_input_grad[0] or ctx.needs_input_grad[3]:
            grad_normalized = (grad_output_flat @ weight).reshape(x.shape)
            grad_x, grad_norm_weight = torch.ops.aten._fused_rms_norm_backward.default(
                grad_normalized, x, [x.shape[-1]], rstd, norm_weight,
                [ctx.needs_input_grad[0], ctx.needs_input_grad[3]],
            )
        return grad_x, grad_weight, grad_bias, grad_norm_weight


class RMSLinear(Linear):
    """RMSNorm + linear, saving the input and reciprocal RMS, not normalized x."""

    def forward(self, x: Tensor, norm_weight: Tensor):
        return _RMSLinearFunction.apply(
            x, self.weight.type_as(x), self.bias.type_as(x), norm_weight.type_as(x),
        )


class Rotary(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        angular_freq = (1 / 1024) ** torch.linspace(0, 1, steps=dim // 4, dtype=torch.float32)
        self.register_buffer("angular_freq", torch.cat([angular_freq, angular_freq.new_zeros(dim // 4)]))

    def forward(self, x_BTHD: Tensor):
        pos = torch.arange(x_BTHD.size(1), dtype=torch.float32, device=x_BTHD.device)
        theta = torch.outer(pos, self.angular_freq)[None, :, None, :]
        cos, sin = theta.cos(), theta.sin()
        x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), 3).type_as(x_BTHD)


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, head_dim=128):
        super().__init__()
        self.num_heads = dim // head_dim
        self.head_dim = head_dim
        hdim = self.num_heads * self.head_dim
        self.q = Linear(dim, hdim)
        self.k = Linear(dim, hdim)
        self.v = Linear(dim, hdim)
        self.proj = Linear(hdim, dim)
        self.rotary = Rotary(head_dim)

    @torch.compile()
    def forward(self, x: Tensor):
        B, T = x.size(0), x.size(1)
        q = self.q(x).view(B, T, self.num_heads, self.head_dim)
        k = self.k(x).view(B, T, self.num_heads, self.head_dim)
        v = self.v(x).view(B, T, self.num_heads, self.head_dim)
        q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),))
        q, k = self.rotary(q), self.rotary(k)
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2),
                                           v.transpose(1, 2), scale=0.12, is_causal=True).transpose(1, 2)
        y = y.contiguous().view(B, T, self.num_heads * self.head_dim)
        y = self.proj(y)
        return y


class MLP(nn.Module):
    def __init__(
        self,
        dim: int,
        layer_num: int,
        cfg: dict,
    ):
        super().__init__()
        self.layer_num = layer_num
        self.cfg = cfg
        hdim = 4 * dim
        self.fc = RMSLinear(dim, hdim)
        self.proj = Linear(hdim, dim)

    def _forward_fc(self, x: Tensor, norm_weight: Tensor):
        return self.fc(x, norm_weight)

    def _forward_basic(self, x: Tensor, norm_weight: Tensor):
        x = self._forward_fc(x, norm_weight)
        # Linear can return a view; custom-autograd outputs cannot be modified
        # in place. Release that output before allocating the squared activation.
        x = x.relu()
        x = x.square()
        x = self.proj(x)
        return x

    def forward(self, x: Tensor, norm_weight: Tensor):
        if self.cfg['checkpoint']:
            return checkpoint(self._forward_basic, x, norm_weight, use_reentrant=True)
        return self._forward_basic(x, norm_weight)


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        layer_num: int,
        cfg: dict,
    ):
        super().__init__()
        self.attn = CausalSelfAttention(dim)
        self.mlp = MLP(dim, layer_num, cfg)
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)

    def forward(self, x: Tensor):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(x, self.norm2.gains)
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int, model_dim: int, cfg: dict):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, model_dim).bfloat16()
        self.blocks = nn.ModuleList([
            Block(model_dim, layer_num, cfg)
            for layer_num in range(num_layers)
        ])
        self.proj = Linear(model_dim, vocab_size)
        self.norm1 = RMSNorm(model_dim)
        self.norm2 = RMSNorm(model_dim)

    def forward(self, inputs: Tensor, targets: Tensor):
        x = self.norm1(self.embed(inputs))
        for block in self.blocks:
            x = block(x)
        return self.output_projection(x, targets)

    @torch.compile()
    def output_projection(self, x, targets):
        logits = self.proj(self.norm2(x)).float()
        logits = 15 * logits * (logits.square() + 15**2).rsqrt()
        return F.cross_entropy(logits.view(targets.numel(), -1), targets.view(-1), reduction="sum")
