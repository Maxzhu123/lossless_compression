"""Benchmark correctness and performance for the sparse SGD update.

Current path (dense materialisation):
    update = mom.decompress() * scale
    result = p + update          # encoded as a compressed tensor

Planned fused path:
    result = scale * mom + p     # both operands start compressed
                                   # dummy implementation only for now
"""

import statistics

import torch

from compress.code_storage import Distribution, DistType
from compress.compress import (
    compress,
    compressed_add,
    compressed_scale_add,
    decompress,
)
from compress.tensor_buffer import Allocation, TensorBuffer


SIZES = [512, 1024, 2048, 4096]
SHAPE = lambda n: (n, 4 * n)
WARMUP = 3
ITERATIONS = 50
TRIALS = 3
SCALE_VALUE = -0.5


def _buffer(numel: int) -> TensorBuffer:
    capacity = (numel * 2 + 64 * 1024 * 1024 + 15) // 16 * 16
    return TensorBuffer(capacity, device="cuda")


def _free(result, buffer: TensorBuffer) -> None:
    if result is not None and result.fallback_descriptor is not None:
        buffer.free(Allocation(result.fallback_descriptor, buffer))


def _assert_bits_equal(left: torch.Tensor, right: torch.Tensor) -> None:
    assert left.shape == right.shape
    assert torch.equal(
        left.contiguous().view(torch.int16),
        right.contiguous().view(torch.int16),
    )


def _make_compressed_pair(n: int):
    shape = SHAPE(n)
    generator = torch.Generator(device="cuda").manual_seed(n)
    p = torch.randn(shape, device="cuda", generator=generator).to(torch.bfloat16)
    mom = torch.randn(shape, device="cuda", generator=generator).to(torch.bfloat16)
    dist = Distribution(DistType.GAUSSIAN)
    p_enc = compress(p, dist, None)
    mom_enc = compress(mom, dist, None)
    return p, mom, p_enc, mom_enc, dist, shape


def _current_update(p_enc, mom_enc, scale, out_buf):
    """Current implementation used by SparseSGDM, expressed as a pure op."""
    update = decompress(mom_enc) * scale
    return compressed_add(
        p_enc,
        update,
        dense_output=False,
        buffer=out_buf,
        distribution=p_enc.distribution,
    )


def _fused_update(p_enc, mom_enc, scale, out_buf):
    """Fused sparse-update entry point.

    ``compressed_scale_add`` currently decodes the second operand to dense and
    then uses the scalar multiply-add fused path. This avoids materialising
    ``scale * mom`` as the current method does, while keeping both operands
    compressed in the API.
    """
    return compressed_scale_add(
        mom_enc,
        scale,
        p_enc,
        dense_output=False,
        buffer=out_buf,
        distribution=p_enc.distribution,
    )


def _time(function, iterations: int = ITERATIONS) -> float:
    for _ in range(WARMUP):
        function()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        function()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iterations


def _compare(first, second, release_first, release_second):
    first_times = []
    second_times = []
    for trial in range(TRIALS):
        ordered = (
            ((first, release_first), (second, release_second))
            if trial % 2 == 0
            else ((second, release_second), (first, release_first))
        )
        for fn, release in ordered:
            # timed functions return a compressed result that must be freed
            # so the shared output buffer is available for the next call.
            for _ in range(WARMUP):
                result = fn()
                release(result)
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(ITERATIONS):
                result = fn()
                release(result)
            end.record()
            end.synchronize()
            elapsed = start.elapsed_time(end) / ITERATIONS
            if fn is first:
                first_times.append(elapsed)
            else:
                second_times.append(elapsed)
    return statistics.median(first_times), statistics.median(second_times)


def _benchmark_shape(n: int) -> tuple[float, float]:
    out_buf = _buffer(SHAPE(n)[0] * SHAPE(n)[1])
    p, mom, p_enc, mom_enc, dist, shape = _make_compressed_pair(n)
    scale = torch.tensor([SCALE_VALUE], device="cuda", dtype=torch.float32)

    # Correctness.
    expected = (p.float() + mom.float() * scale).to(torch.bfloat16)
    current_result = _current_update(p_enc, mom_enc, scale, out_buf)
    _assert_bits_equal(decompress(current_result), expected)
    _free(current_result, out_buf)
    fused_result = _fused_update(p_enc, mom_enc, scale, out_buf)
    _assert_bits_equal(decompress(fused_result), expected)
    _free(fused_result, out_buf)

    current = lambda: _current_update(p_enc, mom_enc, scale, out_buf)
    fused = lambda: _fused_update(p_enc, mom_enc, scale, out_buf)
    current_ms, fused_ms = _compare(
        current,
        fused,
        lambda result: _free(result, out_buf),
        lambda result: _free(result, out_buf),
    )

    print(
        f"shape={shape!s:>14s}  current={current_ms:7.4f} ms  "
        f"fused={fused_ms:7.4f} ms  "
        f"reduction={(current_ms - fused_ms) / current_ms:6.2%}"
    )

    out_buf.reset()
    del p, mom, p_enc, mom_enc, current_result, fused_result
    torch.cuda.empty_cache()
    return current_ms, fused_ms


def main() -> None:
    total_current_ms = 0.0
    total_fused_ms = 0.0
    for n in SIZES:
        current_ms, fused_ms = _benchmark_shape(n)
        total_current_ms += current_ms
        total_fused_ms += fused_ms

    print(f"Total current: {total_current_ms:.5g}ms")
    print(f"Total fused: {total_fused_ms:.5g}ms")


if __name__ == "__main__":
    main()
