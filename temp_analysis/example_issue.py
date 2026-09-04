"""Minimal reproducer for the buffer-vs-no-buffer sparse update slowdown.

The saved tensors are a momentum tensor and a parameter tensor for the fused
operation ``alpha * mom + param``.  With the shared-buffer path both operands
are buffer-backed, so ``pointwise_scale_add_compressed`` would enter the
one-pass fused kernel.  That kernel scans every fallback stream for every
codec block, which is very slow when the momentum tensor has many overflow
streams.

The no-buffer path in the original MLP setup keeps the momentum private while
the parameter is still buffer-backed.  That gives mixed buffering, so the
function takes the dense-decode fallback and avoids the pathological scan.

This script times those two paths directly.
"""

import torch
import time
import sys
from pathlib import Path
from dataclasses import replace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from compress.code_storage import Distribution, DistType
from compress.compress import (
    compress,
    decompress,
    a_compA_add_compB,
)
from compress.ops.pointwise import (
    COMPRESSED_OUTPUT,
    _launch_scalar_mul_add_compressed_compressed,
)
from compress.codec.runtime import compress_components
from compress.tensor_buffer import TensorBuffer, Allocation

HERE = Path(__file__).resolve().parent
SHAPE = (4096, 8192)
DIST = Distribution(DistType.EMPIRICAL, zero_prob=0.5)
ALPHA = torch.load(HERE / "alpha.pt", map_location="cuda")
WARMUP = 2
ITERATIONS = 5


def make_buffer():
    # Big enough to hold the compressed tensors / result fallback storage.
    return TensorBuffer(1_000_000_000, device="cuda")


def free_compressed(c, buffer):
    if c is not None and c.fallback_descriptor is not None and buffer is not None:
        buffer.free(Allocation(c.fallback_descriptor, buffer))


def run_fused_buffered(mom, param, buffer):
    """Exactly the one-pass fused path that pointwise_scale_add_compressed
    uses when both operands are buffer-backed and the fused path is enabled.
    """
    mom_c = compress(mom, DIST, buffer)
    param_c = compress(param, DIST, buffer)

    alpha = ALPHA
    values, auxiliary = _launch_scalar_mul_add_compressed_compressed(
        mom_c,
        param_c,
        alpha,
        COMPRESSED_OUTPUT,
    )
    result = compress_components(
        auxiliary,
        values,
        mom_c.size,
        DIST,
        buffer,
        mom_c.shape,
        precomputed=True,
        logical_numel=mom_c.logical_numel,
    )
    result = replace(result, layout=mom_c.layout)

    # Keep an output so the result is not optimised away; free in caller.
    free_compressed(mom_c, buffer)
    free_compressed(param_c, buffer)
    return result


def run_dense_fallback(mom, param, buffer):
    """Workaround actually used when the momentum is private (buffer=None):
    mixed buffering makes pointwise_scale_add_compressed decode the parameter
    to dense and avoid the fused compressed-compressed kernel.
    """
    mom_c = compress(mom, DIST, None)          # private momentum
    param_c = compress(param, DIST, buffer)    # buffered parameter

    result = a_compA_add_compB(
        mom_c,
        ALPHA,
        param_c,
        dense_output=False,
        buffer=buffer,
        distribution=DIST,
    )

    free_compressed(mom_c, None)
    free_compressed(param_c, buffer)
    return result


def time_path(fn, mom, param, buffer, label):
    result = None
    for _ in range(WARMUP):
        result = fn(mom, param, buffer)
        free_compressed(result, buffer)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(ITERATIONS):
        result = fn(mom, param, buffer)
        free_compressed(result, buffer)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - start) / ITERATIONS * 1000.0

    # One correctness check with an exact binary-representation-friendly alpha.
    if label == "buffer":
        # Rebuild once with a dense expected result using the same random data.
        a = decompress(compress(mom, DIST, buffer))
        b = decompress(compress(param, DIST, buffer))
        # Use the same alpha as the timed run.
        expected = (b.float() + a.float() * float(ALPHA.item())).to(torch.bfloat16)
        check = decompress(result)
        ok = torch.equal(check.view(torch.int16), expected.view(torch.int16))
    else:
        expected = (param.float() + mom.float() * float(ALPHA.item())).to(torch.bfloat16)
        check = decompress(result)
        ok = torch.equal(check.view(torch.int16), expected.view(torch.int16))
    print(f"{label:12s}: {ms:8.3f} ms/call  correct={ok}")
    free_compressed(result, buffer)
    return ms


def main():
    torch.manual_seed(0)
    mom = torch.load(HERE / "mom.pt", map_location="cuda")
    param = torch.load(HERE / "param.pt", map_location="cuda")
    buffer = make_buffer()

    print(f"shape={SHAPE}  distribution={DIST}")
    print("Timing fused buffered path vs dense-decode no-buffer path...\n")
    fused_ms = time_path(run_fused_buffered, mom, param, buffer, "buffer")
    dense_ms = time_path(run_dense_fallback, mom, param, buffer, "no-buffer")

    print(f"\nratio: buffer/no-buffer = {fused_ms / dense_ms:.2f}x slower")


if __name__ == "__main__":
    main()
