"""Minimal reproducer for the buffer-vs-no-buffer sparse update slowdown.

The saved tensors are deterministic stand-ins for the BF16 momentum and
parameter tensors involved in ``alpha * mom + param``.  With both operands
buffer-backed, ``a_compA_add_compB`` uses the slow one-pass fused kernel.  With
only the parameter buffer-backed and the momentum private, it takes the fast
dense-decode fallback.
"""

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from compress.code_storage import Distribution, DistType, NoiseLevel
from compress.compress import compress, decompress, a_compA_add_compB
from compress.tensor_buffer import TensorBuffer, Allocation

HERE = Path(__file__).resolve().parent
SHAPE = (4096, 8192)
DIST = Distribution(DistType.EMPIRICAL, zero_prob=0.5)
ALPHA = torch.load(HERE / "alpha.pt", map_location="cuda")
WARMUP = 2
ITERATIONS = 5


def free(compressed, buffer):
    if compressed is not None and compressed.fallback_descriptor is not None and buffer is not None:
        buffer.free(Allocation(compressed.fallback_descriptor, buffer))


def run(mom, param, buffer, mom_buffer):
    """One sparse-update operation; the only difference is mom's storage."""
    mom_c = compress(mom, DIST, mom_buffer)
    param_c = compress(param, DIST, buffer)

    result = a_compA_add_compB(
        mom_c,
        ALPHA,
        param_c,
        dense_output=False,
        buffer=buffer,
        distribution=DIST,
    )

    free(mom_c, mom_buffer)
    free(param_c, buffer)
    return result


def measure(mom, param, buffer, mom_buffer, label):
    for _ in range(WARMUP):
        free(run(mom, param, buffer, mom_buffer), buffer)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(ITERATIONS):
        free(run(mom, param, buffer, mom_buffer), buffer)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - start) / ITERATIONS * 1000.0

    result = run(mom, param, buffer, mom_buffer)
    expected = (param.float() + mom.float() * float(ALPHA.item())).to(torch.bfloat16)
    correct = torch.equal(decompress(result).view(torch.int16), expected.view(torch.int16))
    free(result, buffer)

    assert correct, "Sparse update result is incorrect"
    print(f"{label:12s}: {ms:8.3f} ms/call")
    return ms


def main():
    mom = torch.load(HERE / "mom.pt", map_location="cuda")
    param = torch.load(HERE / "param.pt", map_location="cuda")
    buffer = TensorBuffer(1_000_000_000, device="cuda")

    print(f"shape={SHAPE}  distribution={DIST}")
    print("Timing shared-buffer fused path vs private-momentum dense fallback...\n")

    buffered_ms = measure(mom, param, buffer, buffer, "buffer")
    private_ms = measure(mom, param, buffer, None, "no-buffer")

    print(f"\nratio: buffer/no-buffer = {buffered_ms / private_ms:.2f}x slower")


if __name__ == "__main__":
    main()
