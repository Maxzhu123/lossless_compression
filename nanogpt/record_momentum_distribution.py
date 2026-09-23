"""Record exact Muon momentum exponent counts during early nanoGPT training."""
import csv
from datetime import datetime
import gc
import json
from pathlib import Path

import torch

import test_time_mem as settings
from LCT.LCTensor import LCTTensor
from LCT.tensor_buffer import TensorBuffer

STEPS = 8
ARTEFACTS = Path(__file__).resolve().parents[1] / "artefacts"


@torch.no_grad()
def record_histograms(writer, step, tensors):
    zero_fractions = []
    for name, tensor in tensors:
        x = tensor.decompress() if isinstance(tensor, LCTTensor) else tensor
        bits = x.contiguous().view(torch.int16).to(torch.int32)
        counts = torch.bincount(((bits >> 7) & 255).flatten(), minlength=256).cpu().tolist()
        zeros = int(((bits & 0x7fff) == 0).sum().item())
        assert sum(counts) == x.numel() and counts[255] == 0
        for exponent_byte, count in enumerate(counts):
            writer.writerow((step, name, exponent_byte - 127,
                             count, x.numel(), zeros if exponent_byte == 0 else 0))
        zero_fractions.append(zeros / x.numel())
        del x, bits
    return sum(zero_fractions) / len(zero_fractions)


def main(kind="momentum"):
    training = settings.training
    torch.manual_seed(training.SEED)
    model = training.GPT(training.VOCAB_SIZE, training.NUM_LAYERS, training.MODEL_DIM,
                         compress_activations=settings.COMPRESS_ACTIVATIONS,
                         min_compress_elements=training.MIN_COMPRESS_ELEMENTS).cuda().train()
    training.initialize_model(model)
    compressed = settings.COMPRESS_WEIGHTS or settings.COMPRESS_ACTIVATIONS or settings.COMPRESS_OPTIMISER
    if settings.BUFFER and compressed:
        model.set_tensor_buffer(TensorBuffer(settings.BUFFER_SIZE_MIB * 2**20, device="cuda"))
    if settings.COMPRESS_WEIGHTS:
        model.compress_weights()
    if settings.COMPILE:
        model.compile()
    adam, muon = training.make_optimizers(model, compressed=settings.COMPRESS_OPTIMISER)
    names = {id(p): name for name, p in model.named_trainable_tensors()}
    loader = training.data_generator("data/fineweb10B/fineweb_train_*.bin",
                                     settings.TRAIN_BATCH_TOKENS, settings.SEQ_LEN)

    def tensors():
        if kind == "momentum":
            return [(names[id(p)], m) for p, m in zip(muon.params, muon.momentums)]
        return [(name, p) for name, p in model.named_trainable_tensors()
                if p.ndim == 2 and p.dtype == torch.bfloat16]

    folder = {"momentum": "muon_momentum", "weights": "nanogpt_weights"}[kind]
    run_dir = ARTEFACTS / folder / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir.mkdir(parents=True)
    path = run_dir / "exponents.csv"
    metadata = dict(steps=STEPS, seed=training.SEED, layers=training.NUM_LAYERS,
                    width=training.MODEL_DIM, sequence_length=settings.SEQ_LEN,
                    tokens_per_step=settings.TRAIN_BATCH_TOKENS,
                    microbatch_sequences=settings.SEQUENCES_PER_MICROBATCH,
                    kind=kind, tensor_count=len(tensors()), torch_version=str(torch.__version__),
                    selection="Muon momentum buffers" if kind == "momentum" else
                              "All BF16 weight matrices, including embedding and LM head; excludes biases and gains",
                    source="FineWeb training shards; fresh model initialization",
                    compress_weights=settings.COMPRESS_WEIGHTS,
                    compress_activations=settings.COMPRESS_ACTIVATIONS,
                    compress_momentum=settings.COMPRESS_OPTIMISER,
                    compile=settings.COMPILE,
                    buffer_mib=settings.BUFFER_SIZE_MIB if model.tensor_buffer is not None else 0,
                    sampling="All elements, after each optimizer update; no histogram sampling",
                    exponent="BF16 exponent byte minus 127; -127 includes zero/subnormal; 128 is nonfinite")
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Recording {len(tensors())} {kind} tensors to {path}", flush=True)
    with path.open("x", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("step", "parameter", "exponent", "count", "elements", "zero_count"))
        for step in range(1, STEPS + 1):
            if step == settings.WARMUP_STEPS + 1:
                gc.collect()
            inputs, targets = next(loader)
            total_loss = torch.zeros((), device="cuda")
            for start in range(0, len(inputs), settings.SEQUENCES_PER_MICROBATCH):
                stop = start + settings.SEQUENCES_PER_MICROBATCH
                loss = model(inputs[start:stop], targets[start:stop])
                total_loss += loss.detach()
                loss.backward()
                del loss
            adam.step()
            muon.step()
            model.zero_grad(set_to_none=True)
            zero_fraction = record_histograms(writer, step, tensors())
            file.flush()
            print(f"Step {step}: loss/token={total_loss.item()/settings.TRAIN_BATCH_TOKENS:.4f}, "
                  f"mean zero fraction={zero_fraction:.2%}", flush=True)

    script = "plot_nanogpt_momentum_exponent_history.py" if kind == "momentum" else "plot_nanogpt_weight_exponent_history.py"
    print(f"Plot with: python plots/{script}", flush=True)


if __name__ == "__main__":
    main()
