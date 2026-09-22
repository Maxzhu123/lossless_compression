"""Quick CUDA time/memory check for train_gpt_lct; edit the options below.

Uses the training model, initialization, data, optimizers and batch settings.
No validation, checkpoints or log files. First-time compilation can take longer.
"""
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import train_gpt_lct as training

COMPRESS_WEIGHTS = False
COMPRESS_ACTIVATIONS = False
COMPRESS_OPTIMISER = False
BUFFER = False
BUFFER_SIZE_MIB = training.BUFFER_SIZE_MIB
SEQ_LEN = training.SEQ_LEN
TRAIN_BATCH_TOKENS = training.TRAIN_BATCH_TOKENS
SEQUENCES_PER_MICROBATCH = training.TRAIN_MICROBATCH_SEQUENCES
WARMUP_STEPS = 5
MEASURE_STEPS = 5
# Use the trainer's compilation boundaries; do not compile the whole LCT model.


def main():
    import torch
    from LCT.tensor_buffer import TensorBuffer

    if WARMUP_STEPS < 0 or MEASURE_STEPS < 1:
        raise ValueError("Warmup must be nonnegative and measured steps must be positive")
    if SEQ_LEN < 1 or TRAIN_BATCH_TOKENS < 1 or SEQUENCES_PER_MICROBATCH < 1:
        raise ValueError("Sequence length, batch tokens, and microbatch size must be positive")
    if TRAIN_BATCH_TOKENS % SEQ_LEN:
        raise ValueError("Token batch must divide evenly into sequences")
    sequences = TRAIN_BATCH_TOKENS // SEQ_LEN
    accumulation_steps = (sequences + SEQUENCES_PER_MICROBATCH - 1) // SEQUENCES_PER_MICROBATCH
    compressed = COMPRESS_WEIGHTS or COMPRESS_ACTIVATIONS or COMPRESS_OPTIMISER
    mode = "lct" if compressed else "dense"
    use_buffer = BUFFER and compressed
    print(f"Running {mode}: weights={COMPRESS_WEIGHTS}, activations={COMPRESS_ACTIVATIONS}, "
          f"Muon momentum={COMPRESS_OPTIMISER}, buffer={use_buffer}, "
          f"{SEQUENCES_PER_MICROBATCH} sequences/microbatch, "
          f"{accumulation_steps} microbatches/step", flush=True)
    if not torch.cuda.is_available():
        raise RuntimeError("This test requires CUDA")
    torch.cuda.set_device(0)
    torch.manual_seed(training.SEED)
    model = training.GPT(training.VOCAB_SIZE, training.NUM_LAYERS, training.MODEL_DIM,
                         compress_activations=COMPRESS_ACTIVATIONS,
                         min_compress_elements=training.MIN_COMPRESS_ELEMENTS).cuda().train()
    training.initialize_model(model)

    if use_buffer:
        model.set_tensor_buffer(TensorBuffer(BUFFER_SIZE_MIB * 2**20, device="cuda"))
    if COMPRESS_WEIGHTS:
        model.compress_weights()

    adam, muon = training.make_optimizers(model, compressed=COMPRESS_OPTIMISER)
    tokens = TRAIN_BATCH_TOKENS
    loader = training.data_generator("data/fineweb10B/fineweb_train_*.bin", tokens, SEQ_LEN)
    peaks = []
    step_times = []
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
        elapsed = time.perf_counter() - start
        peak = torch.cuda.max_memory_allocated() / 2**20
        warmup = step < WARMUP_STEPS
        if not warmup:
            peaks.append(peak)
            step_times.append(elapsed)
        print(f"{mode} step {step+1} ({'warmup' if warmup else 'measured'}): "
              f"peak={peak:.1f} MiB, reserved={torch.cuda.max_memory_reserved()/2**20:.1f} MiB, "
              f"after step={torch.cuda.memory_allocated()/2**20:.1f} MiB, "
              f"loss={total_loss.item()/tokens:.4f}, time={elapsed:.2f}s", flush=True)
    print(f"Peak allocated after warmup: {max(peaks):.1f} MiB", flush=True)
    print(f"Mean step time after warmup: {sum(step_times)/len(step_times):.3f}s", flush=True)



if __name__ == "__main__":
    main()
