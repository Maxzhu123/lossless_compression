"""Quick CUDA memory check. Select USE_BITSPARSE below and run directly.

Uses the training model, initialization, data, optimizers and batch settings.
No validation, checkpoints or log files. First-time compilation can take longer.
"""
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train_gpt_simple import SEQ_LEN, TRAIN_BATCH_TOKENS

USE_BITSPARSE = True
USE_TENSOR_BUFFER = True
PACK_SBIT = True
BUFFER_SIZE_MIB = 2408
SEQUENCES_PER_MICROBATCH = 64
ACCUMULATION_STEPS = (TRAIN_BATCH_TOKENS + SEQ_LEN * SEQUENCES_PER_MICROBATCH - 1) // (SEQ_LEN * SEQUENCES_PER_MICROBATCH)
WARMUP_STEPS = 15
MEASURE_STEPS = 5
COMPILE = True  # Model compilation; Muon retains its own compile decorator.


def main():
    import torch
    from train_gpt_simple import GPT, Muon, data_generator
    from lib_sparse.bitsparse import TensorBuffer

    mode = "bitsparse" if USE_BITSPARSE else "dense"
    print(f"Running {mode}: buffer={USE_TENSOR_BUFFER}, sign packing={PACK_SBIT}, "
          f"compile={COMPILE}, {ACCUMULATION_STEPS} microbatches/step", flush=True)
    if USE_TENSOR_BUFFER and not USE_BITSPARSE:
        raise ValueError("USE_TENSOR_BUFFER requires USE_BITSPARSE=True")
    if not torch.cuda.is_available():
        raise RuntimeError("This test requires CUDA")
    torch.cuda.set_device(0)
    torch.manual_seed(0)
    model = GPT(50304, 12, 768, use_bitsparse=USE_BITSPARSE, pack_sbit=PACK_SBIT).cuda().train()

    if USE_TENSOR_BUFFER:
        model.set_tensor_buffer(TensorBuffer(
            BUFFER_SIZE_MIB * 2**20, device="cuda", dtype=torch.bfloat16, pack_sbit=PACK_SBIT,
        ))

    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("weight"):
                if "proj" in name:
                    p.zero_()
                elif "embed" in name:
                    p.normal_()
                else:
                    p.normal_(std=0.33**0.5 / p.size(-1)**0.5)
            elif name.endswith("bias"):
                p.zero_()
            elif name.endswith("gains"):
                p.fill_(1)
            else:
                raise ValueError(f"Uninitialized parameter: {name}")

    if COMPILE:
        model.compile(dynamic=False)
    adam = torch.optim.AdamW([
        dict(params=[model.embed.weight], lr=0.3),
        dict(params=[model.proj.weight], lr=1/320),
        dict(params=[p for p in model.parameters() if p.ndim < 2], lr=0.01),
    ], betas=(0.8, 0.95), eps=1e-10, weight_decay=0, fused=True)
    muon = Muon([p for p in model.blocks.parameters() if p.ndim >= 2],
                lr=0.035, weight_decay=0.025)
    tokens = TRAIN_BATCH_TOKENS
    loader = data_generator("data/fineweb10B/fineweb_train_*.bin", tokens, SEQ_LEN)
    peaks = []
    for step in range(WARMUP_STEPS + MEASURE_STEPS):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        inputs, targets = next(loader)
        total_loss = torch.zeros((), device="cuda")
        for i in range(0, len(inputs), SEQUENCES_PER_MICROBATCH):
            loss = model(inputs[i:i+SEQUENCES_PER_MICROBATCH], targets[i:i+SEQUENCES_PER_MICROBATCH])
            total_loss += loss.detach()
            loss.backward()
            del loss
        adam.step()
        muon.step()
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() / 2**20
        warmup = step < WARMUP_STEPS
        if not warmup:
            peaks.append(peak)
        print(f"{mode} step {step+1} ({'warmup' if warmup else 'measured'}): "
              f"peak={peak:.1f} MiB, reserved={torch.cuda.max_memory_reserved()/2**20:.1f} MiB, "
              f"after step={torch.cuda.memory_allocated()/2**20:.1f} MiB, "
              f"loss={total_loss.item()/tokens:.4f}, time={time.perf_counter()-start:.2f}s", flush=True)
    print(f"Peak allocated after warmup: {max(peaks):.1f} MiB", flush=True)



if __name__ == "__main__":
    main()
