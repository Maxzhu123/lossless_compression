"""Qwen CUDA time/memory benchmark; edit the options below and run directly.

Uses the fine-tuning model, FineWeb data, loss, and optimizers. No validation,
checkpoint saving, or log files. Each configuration should run in a fresh process.
"""
import gc
import math
import time

import qwen.train_qwen_lct as training

MODEL_ID = training.MODEL_ID
DATA_DIR = training.DATA_DIR
PRETRAINED = True  # False uses random weights with the same model architecture.
COMPRESS_WEIGHTS = True
COMPRESS_ACTIVATIONS = True
COMPRESS_OPTIMISER = True  # Muon momentum only; AdamW moments remain dense.
BUFFER = False
BUFFER_SIZE_MIB = 256
MIN_COMPRESS_ELEMENTS = 65536
COMPILE = False  # Model compilation; cross-entropy compiles independently.
LOG_GRAPH_BREAKS = False
CHECKPOINT_LAYERS = False
CHECKPOINT_HEAD = False
OPTIMIZER_IN_BACKWARD = True
OPTIMIZER = 'muon'  # 'muon' or 'adamw'
SEQ_LEN = None  # Tokens per sequence; edit this to benchmark other sequence lengths.
HEAD_CHUNK_TOKENS = training.HEAD_CHUNK_TOKENS
WARMUP_STEPS = 3
MEASURE_STEPS = 5


def main():
    import torch
    from transformers import AutoConfig
    from LCT.tensor_buffer import TensorBuffer, _free_regions_snapshot
    from qwen.components import Compression, configure_compression, named_trainable_tensors
    from qwen.data import TokenStream
    from qwen.optimizer_in_backward import OptimizerInBackward

    if SEQ_LEN < 1:
        raise ValueError('SEQ_LEN must be positive')
    if WARMUP_STEPS < 0 or MEASURE_STEPS < 1:
        raise ValueError('Use nonnegative warmup steps and at least one measured step')

    stream = TokenStream(DATA_DIR, 'train', MODEL_ID)
    torch.cuda.set_device(0)
    torch.manual_seed(training.SEED)
    torch._logging.set_logs(graph_breaks=LOG_GRAPH_BREAKS)
    # These helpers read the trainer's options; no training source files are changed.
    training.COMPRESS_OPTIMISER = COMPRESS_OPTIMISER
    training.OPTIMIZER = OPTIMIZER
    training.HEAD_CHUNK_TOKENS = HEAD_CHUNK_TOKENS
    training.CHECKPOINT_HEAD = CHECKPOINT_HEAD
    print(f'Model={MODEL_ID}, pretrained={PRETRAINED}, weights={COMPRESS_WEIGHTS}, '
          f'activations={COMPRESS_ACTIVATIONS}, Muon momentum={COMPRESS_OPTIMISER}, '
          f'optimizer={OPTIMIZER}, optimizer-in-backward={OPTIMIZER_IN_BACKWARD}, '
          f'checkpoint layers={CHECKPOINT_LAYERS}, checkpoint head={CHECKPOINT_HEAD}, compile={COMPILE}, '
          f'batch size=1, sequence length={SEQ_LEN}, head chunk={HEAD_CHUNK_TOKENS}', flush=True)

    if PRETRAINED:
        model = training.Qwen3ForCausalLM.from_pretrained(
            MODEL_ID, dtype=torch.bfloat16, attn_implementation='sdpa').cuda().train()
    else:
        config = AutoConfig.from_pretrained(MODEL_ID)
        config._attn_implementation = 'sdpa'
        with torch.device('cuda'):
            model = training.Qwen3ForCausalLM(config).to(dtype=torch.bfloat16).train()
    use_buffer = BUFFER and (COMPRESS_WEIGHTS or COMPRESS_ACTIVATIONS or COMPRESS_OPTIMISER)
    buffer = TensorBuffer(BUFFER_SIZE_MIB * 2**20, device='cuda') if use_buffer else None
    configure_compression(model, Compression(COMPRESS_WEIGHTS, COMPRESS_ACTIVATIONS,
                                            MIN_COMPRESS_ELEMENTS, buffer))
    if CHECKPOINT_LAYERS:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': True})
    else:
        model.gradient_checkpointing_disable()
    params = [p for _, p in named_trainable_tensors(model)]
    optimizers = training.make_optimizers(model, buffer, parameterwise=OPTIMIZER_IN_BACKWARD)
    fused = OptimizerInBackward(model, optimizers) if OPTIMIZER_IN_BACKWARD else None
    if COMPILE:
        model.model.compile()
    print(f'{sum(p.numel() for p in params):,} trainable parameters; '
          f'buffer={BUFFER_SIZE_MIB if use_buffer else 0} MiB; fixed learning rates', flush=True)

    peaks, times = [], []
    try:
        for step in range(WARMUP_STEPS + MEASURE_STEPS):
            if step == WARMUP_STEPS:
                gc.collect()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            if fused is not None:
                fused.begin(math.ceil(SEQ_LEN / HEAD_CHUNK_TOKENS), checkpoint_head=CHECKPOINT_HEAD)
            inputs, targets = stream.batch(1, SEQ_LEN)
            loss = training.loss_for_batch(model, inputs, targets) / SEQ_LEN
            step_loss = loss.detach()
            loss.backward()
            del loss
            if fused is not None:
                fused.finish()
            else:
                for optimizer in optimizers:
                    optimizer.step()
                for parameter in params:
                    parameter.grad = None
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            peak = torch.cuda.max_memory_allocated() / 2**20
            final_loss = step_loss.item()
            warmup = step < WARMUP_STEPS
            if not warmup:
                peaks.append(peak)
                times.append(elapsed)
            buffer_text = ''
            if buffer is not None:
                used = buffer.capacity_bytes - sum(size for _, size in _free_regions_snapshot(buffer))
                buffer_text = f'buffer_allocated:{used/2**20:.1f}/{buffer.capacity_bytes/2**20:.1f}MiB, '
            print(f'step {step+1} ({"warmup" if warmup else "measured"}): '
                  f'peak={peak:.1f} MiB, {buffer_text}'
                  f'loss={final_loss:.5f}, time={elapsed:.3f}s', flush=True)
    finally:
        if fused is not None:
            fused.remove()
    mean_time = sum(times) / len(times)
    print(f'Peak allocated after warmup: {max(peaks):.1f} MiB', flush=True)
    print(f'Mean step time after warmup: {mean_time:.3f}s '
          f'({SEQ_LEN/mean_time:.0f} tokens/s)', flush=True)
    return dict(peak_mib=max(peaks), mean_step_seconds=mean_time,
                final_train_loss=final_loss, tokens_per_step=SEQ_LEN)


if __name__ == '__main__':
    # global SEQ_LEN

    for seq_len in [3500]:
        SEQ_LEN = seq_len
        print(f'\n=== Benchmarking sequence length {SEQ_LEN} ===', flush=True)
        main()
