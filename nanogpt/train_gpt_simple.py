"""
train_gpt_simple.py

This file descends from the [NanoGPT speedrun](https://github.com/KellerJordan/modded-nanogpt).
It was prepared as a simplified version of the speedrun for use in neural net optimization research.
"""
import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
from pathlib import Path
import time
from datetime import datetime

import torch
from torch import Tensor, nn
from torch.optim import AdamW
import torch.nn.functional as F

from lib_sparse.layers import FusedRMSNormMLP, Relu2Linear
from lib_sparse.bitsparse import TensorBuffer

from dataloader import load_data_shard

DATA_ROOT = Path(__file__).resolve().parent
USE_BITSPARSE = True  # Compress saved MLP activations with 15-bit BF16 packing.
USE_TENSOR_BUFFER = True  # Requires USE_BITSPARSE.
BUFFER_SIZE_MIB = 2408  # Covers 12 layers × 64 sequences × 1024 tokens * 768 * 4, packed BF16.
SEQ_LEN = 1024
TRAIN_BATCH_TOKENS = 8 * 64 * 1024  # Tokens per optimizer step.
VAL_TOKENS = 20 * 524288
TRAIN_MICROBATCH_SEQUENCES = 64  # Tune without changing tokens per optimizer step.
VAL_MICROBATCH_SEQUENCES = 32


def data_generator(pattern, batch_size, seq_len=SEQ_LEN, device=None):
    """Yield token batches on one GPU, cycling through training shards."""
    if batch_size % seq_len:
        raise ValueError("Token batch must divide evenly into sequences")
    files = sorted(DATA_ROOT.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No data shards matched {DATA_ROOT / pattern}")
    if device is None:
        device = torch.device("cuda", torch.cuda.current_device())
    shard_index, position = 0, 0
    tokens = load_data_shard(files[shard_index])
    while True:
        if tokens.numel() < batch_size + 1:
            raise ValueError(f"Shard is too small for the global batch: {files[shard_index]}")
        if position + batch_size + 1 > tokens.numel():
            shard_index = (shard_index + 1) % len(files)
            del tokens
            tokens = load_data_shard(files[shard_index])
            position = 0
            continue
        window = tokens[position:position + batch_size + 1]
        inputs = window[:-1].to(device=device, dtype=torch.int32, non_blocking=True)
        targets = window[1:].to(device=device, dtype=torch.int64, non_blocking=True)
        position += batch_size
        yield inputs.view(-1, seq_len), targets.view(-1, seq_len)


########################################
#             Architecture             #
########################################

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

class Rotary(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        # half-truncate RoPE (w/ base freq tuning)
        angular_freq = (1 / 1024) ** torch.linspace(0, 1, steps=dim//4, dtype=torch.float32)
        self.register_buffer("angular_freq", torch.cat([angular_freq, angular_freq.new_zeros(dim//4)]))

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
    def __init__(self, dim: int, use_bitsparse: bool = False, pack_sbit: bool = True):
        super().__init__()
        self.use_bitsparse = use_bitsparse
        self.pack_sbit = pack_sbit
        self.sparse_data = None
        hdim = 4 * dim
        self.fc = Linear(dim, hdim)
        self.proj = Linear(hdim, dim)

    def forward(self, x: Tensor, norm_weight: Tensor):
        # Both modes share the fused norm and F.rms_norm's default epsilon.
        z = FusedRMSNormMLP.apply(
            x, self.fc.weight.type_as(x), norm_weight.type_as(x), None,
        )
        z = z + self.fc.bias.type_as(x)
        if self.use_bitsparse:
            y = Relu2Linear.apply(
                z.reshape(-1, z.shape[-1]), self.proj.weight.type_as(x), self.sparse_data, self.pack_sbit,
            )
            return y.reshape(x.shape) + self.proj.bias.type_as(x)
        return self.proj(z.relu_().square())

class Block(nn.Module):
    def __init__(self, dim: int, use_bitsparse: bool = False, pack_sbit: bool = True):
        super().__init__()
        self.attn = CausalSelfAttention(dim)
        self.mlp = MLP(dim, use_bitsparse, pack_sbit)
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)

    def forward(self, x: Tensor):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(x, self.norm2.gains)
        return x

class GPT(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int, model_dim: int, use_bitsparse: bool = False,
                 pack_sbit: bool = True):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, model_dim).bfloat16()
        self.blocks = nn.ModuleList([Block(model_dim, use_bitsparse, pack_sbit) for _ in range(num_layers)])
        self.pack_sbit = pack_sbit
        self.proj = Linear(model_dim, vocab_size)
        self.norm1 = RMSNorm(model_dim)
        self.norm2 = RMSNorm(model_dim)
        self.sparse_data = None

    def set_tensor_buffer(self, buffer):
        self.sparse_data = buffer
        for block in self.blocks:
            block.mlp.sparse_data = buffer

    def required_buffer_bytes(self, tokens):
        if not self.pack_sbit:
            return sum(tokens * block.mlp.fc.out_features * 2 for block in self.blocks)
        # Each layer's starting offset is aligned to eight logical values.
        values = sum((tokens * block.mlp.fc.out_features + 7) // 8 * 8 for block in self.blocks)
        return values * 15 // 8

    def forward(self, inputs: Tensor, targets: Tensor):
        if self.sparse_data is not None:
            # Only one outstanding microbatch: finish backward before the next forward.
            self.sparse_data.reset_buffer()
        x = self.norm1(self.embed(inputs))
        for block in self.blocks:
            x = block(x)
        return self._cross_entropy(x, targets)

    @torch.compile(dynamic=False)
    def _cross_entropy(self, x: Tensor, targets: Tensor):
        logits = self.proj(self.norm2(x)).float()
        logits = 15 * logits * (logits.square() + 15**2).rsqrt()
        return F.cross_entropy(logits.view(targets.numel(), -1), targets.view(-1), reduction="sum")


########################################
#              Optimizer               #
########################################

def zeropower_via_newtonschulz5(G: Tensor) -> Tensor:
    assert G.ndim >= 2
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations, not optimizing for wallclock speed
    a, b, c = 2, -1.5, 0.5
    for _ in range(12):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X

@torch.compile(fullgraph=True)
def muon_update(grad, momentum, mu=0.95, nesterov=True):
    momentum.lerp_(grad, 1 - mu)
    update = grad.lerp_(momentum, mu) if nesterov else momentum
    update = zeropower_via_newtonschulz5(update)
    update *= max(1, grad.size(-2) / grad.size(-1))**0.5
    return update

class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, weight_decay=0, mu=0.95):
        assert isinstance(params, list) and len(params) >= 1 and isinstance(params[0], torch.nn.Parameter)
        params = sorted(params, key=lambda x: x.size(), reverse=True)
        defaults = dict(lr=lr, weight_decay=weight_decay, mu=mu)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if len(state) == 0:
                    state["momentum"] = torch.zeros_like(p)
                update = muon_update(p.grad, state["momentum"], mu=group["mu"])
                p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(update, alpha=-group["lr"])


########################################
#                Setup                 #
########################################


def train(device):
    save_every = 300
    if USE_TENSOR_BUFFER and not USE_BITSPARSE:
        raise ValueError("USE_TENSOR_BUFFER requires USE_BITSPARSE=True")

    def print_log(s, console=True, log=True):
        if console:
            print(s, flush=True)
        if log:
            with open(logfile, "a") as f:
                print(s, file=f)

    def memory_stats():
        return (f"allocated:{torch.cuda.memory_allocated(device) / 2**20:.1f}MiB"
                f" peak_allocated:{torch.cuda.max_memory_allocated(device) / 2**20:.1f}MiB"
                f" reserved:{torch.cuda.memory_reserved(device) / 2**20:.1f}MiB"
                f" peak_reserved:{torch.cuda.max_memory_reserved(device) / 2**20:.1f}MiB")

    val_tokens = VAL_TOKENS
    batch_size = TRAIN_BATCH_TOKENS
    if TRAIN_MICROBATCH_SEQUENCES < 1 or VAL_MICROBATCH_SEQUENCES < 1:
        raise ValueError("Microbatch sizes must be positive")
    if batch_size % SEQ_LEN:
        raise ValueError("Token batch must divide evenly into sequences")
    sequences = batch_size // SEQ_LEN
    accumulation_steps = (sequences + TRAIN_MICROBATCH_SEQUENCES - 1) // TRAIN_MICROBATCH_SEQUENCES

    val_inputs, val_targets = next(data_generator("data/fineweb10B/fineweb_val_*.bin", val_tokens, SEQ_LEN, device="cpu"))

    model = GPT(vocab_size=50304, num_layers=12, model_dim=768, use_bitsparse=USE_BITSPARSE).to(device)
    model.compile(dynamic=False)



    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S_%f')
    model_dir = DATA_ROOT / "logs" / timestamp
    model_dir.mkdir(parents=True)
    logfile = model_dir / f"{timestamp}.txt"
    print(logfile)
    # we begin by logging this file itself
    print_log(Path(__file__).read_text(), console=False)
    print_log(f"USE_BITSPARSE={USE_BITSPARSE}", console=True)
    print_log(f"sequence length={SEQ_LEN}, batch={batch_size} tokens, "
           f"train microbatch={TRAIN_MICROBATCH_SEQUENCES} sequences, "
           f"accumulation={accumulation_steps}, validation microbatch={VAL_MICROBATCH_SEQUENCES} sequences")
    print_log("="*100)
    print_log(f"Running PyTorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}"
           + f" on {torch.cuda.get_device_name(device)}")
    print_log("="*100)


    ########################################
    #       Init & Optim Hyperparams       #
    ########################################

    # we want to minimize this while still reaching 3.28 val loss
    train_steps = 3350

    # initialize model parameters
    for name, p in model.named_parameters():
        w = p.data
        if name.endswith("weight"):
            if "proj" in name:
                w.zero_()
            elif "embed" in name:
                w.normal_()  # default torch init
            else:
                w.normal_(std=0.33**0.5 / w.size(-1)**0.5)  # default torch init
        elif name.endswith("bias"):
            w.zero_()
        elif name.endswith("gains"):
            w.normal_(mean=1, std=0)
        else:
            raise Exception(f"Uninitialized parameter: {name}")
    print("Weights initialised")

    # create the optimizer(s)
    optimizer1 = AdamW([dict(params=[model.embed.weight], lr=0.3),
                        dict(params=[model.proj.weight], lr=1/320),
                        dict(params=[p for p in model.parameters() if p.ndim < 2], lr=0.01)],
                       betas=(0.8, 0.95), eps=1e-10, weight_decay=0, fused=True)
    optimizer2 = Muon([p for p in model.blocks.parameters() if p.ndim >= 2],
                      lr=0.035, weight_decay=0.025)
    optimizers = [optimizer1, optimizer2]
    assert set(p for opt in optimizers for group in opt.param_groups
               for p in group["params"]) == set(model.parameters())
    for opt in optimizers:
        for group in opt.param_groups:
            group["initial_lr"] = group["lr"]

    # learning rate schedule: stable then decay
    def set_hparams(step, cooldown_frac=0.7):
        progress = step / train_steps
        assert 0 <= progress < 1
        if progress < 1 - cooldown_frac:
            eta = 1.0
        else:
            eta = (1 - progress) / cooldown_frac
        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["initial_lr"] * eta

    ########################################
    #        Training and Validation       #
    ########################################

    train_loader = data_generator("data/fineweb10B/fineweb_train_*.bin", batch_size, SEQ_LEN)
    if USE_TENSOR_BUFFER:
        model.set_tensor_buffer(TensorBuffer(
            BUFFER_SIZE_MIB * 2**20, device=device, dtype=torch.bfloat16, pack_sbit=True,
        ))
        print_log(f"Tensor buffer: {BUFFER_SIZE_MIB} MiB")

    # save model at step 0 before any training
    with torch.no_grad():
        torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()},
                   f"{model_dir}/0.pt")
    torch.cuda.synchronize(device)

    # start the clock
    training_time = 0
    last_val_step = 0
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    step_start = t0
    print("Starting training")
    for step in range(train_steps + 1):

        # --------------- VALIDATION SECTION -----------------
        val_step_freq = 125 if step / train_steps < 0.9 else 25
        if step == train_steps or step % val_step_freq == 0:
            # stop the clock
            torch.cuda.synchronize(device)
            time_since_last_val = time.perf_counter() - t0
            step_avg = time_since_last_val / (step - last_val_step) if step > 0 else float("nan")
            last_val_step = step
            training_time += time_since_last_val
            model.eval()
            torch.cuda.reset_peak_memory_stats(device)
            val_loss = 0
            with torch.no_grad():
                val_mbs = VAL_MICROBATCH_SEQUENCES
                for i in range(0, len(val_inputs), val_mbs):
                    v_in = val_inputs[i:i+val_mbs].to(device=device, non_blocking=True)
                    v_tgt = val_targets[i:i+val_mbs].to(device=device, non_blocking=True)
                    val_loss += model(v_in, v_tgt)
            val_loss /= val_tokens
            print_log(f"step:{step}/{train_steps} val_loss:{val_loss:.5f} train_time:{training_time:.3f}s"
                   + f" time_since_last_val:{time_since_last_val:.3f}s {memory_stats()}", console=True)
            model.train()
            # start the clock again
            torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            step_start = t0

            del v_in, v_tgt
            torch.cuda.empty_cache()

        if step == train_steps:
            break

        # --------------- TRAINING SECTION -----------------
        torch.cuda.reset_peak_memory_stats(device)
        inputs, targets = next(train_loader)
        train_mbs = TRAIN_MICROBATCH_SEQUENCES
        for i in range(0, len(inputs), train_mbs):
            model(inputs[i:i+train_mbs], targets[i:i+train_mbs]).backward()
        for name, p in model.named_parameters():
            assert p.grad is not None, name
        # set optimization hyperparameters and take a step
        set_hparams(step)
        for opt in optimizers:
            opt.step()
        model.zero_grad(set_to_none=True)
        if (step + 1) % save_every == 0 or step + 1 == train_steps:
            with torch.no_grad():
                torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()},
                           f"{model_dir}/{step + 1}.pt")
            torch.cuda.synchronize(device)
        torch.cuda.synchronize(device)
        approx_training_time = training_time + (time.perf_counter() - t0)
        step_time = time.perf_counter() - step_start
        step_start = time.perf_counter()
        print_log(f"step:{step+1}/{train_steps} train_time:{approx_training_time:.3f}s"
               + f" step_time:{step_time:.3f}s {memory_stats()}", console=True)



def main():
    if not torch.cuda.is_available():
        raise RuntimeError("Training requires CUDA")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    train(device)


if __name__ == "__main__":
    main()
