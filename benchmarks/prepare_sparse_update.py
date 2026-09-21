"""Time alpha * A_compressed + beta * B_dense, with compressed output.

Compare separate scaling plus the existing fused add against the new two-scale
operation. Requires a_compA_add_b_B; there is no substitute implementation.
"""

import statistics
import torch

from LCT.comp_format import DistType, Distribution
from LCT.compress import compress, a_compA_add_B, a_compA_add_b_B
from LCT.tensor_buffer import TensorBuffer


SIZES = [512, 1024, 2048, 4096]
CASES = [("without_decay", 1.0), ("with_decay", 0.998)]
BETA = -0.02
WARMUP = 3
ITERATIONS = 50
TRIALS = 3


def _time(operation) -> float:
    for _ in range(WARMUP):
        result = operation()
        result.free()
    torch.cuda.synchronize()

    events = [
        (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        for _ in range(ITERATIONS)
    ]
    for start, end in events:
        start.record()
        result = operation()
        end.record()
        # Output release is outside the measured interval.
        result.free()
    torch.cuda.synchronize()
    return statistics.mean(start.elapsed_time(end) for start, end in events)


def _benchmark(n: int, label: str, alpha_value: float) -> None:
    shape = (n, 4 * n)
    numel = shape[0] * shape[1]
    buffer = TensorBuffer(numel * 2 + 64 * 1024 * 1024, device="cuda")
    generator = torch.Generator(device="cuda").manual_seed(n)
    A = torch.randn(shape, device="cuda", generator=generator).bfloat16()
    B = torch.randn(shape, device="cuda", generator=generator).bfloat16()
    A_comp = compress(A, Distribution(DistType.GAUSSIAN), buffer)
    del A
    alpha = torch.tensor([alpha_value], device="cuda", dtype=torch.float32)
    beta = torch.tensor([BETA], device="cuda", dtype=torch.float32)
    scaled_B = torch.empty_like(B)

    def separate():
        torch.mul(B, beta, out=scaled_B)
        return a_compA_add_B(
            A_comp, alpha, scaled_B, alpha_is_one=alpha_value == 1.0,
            dense_output=False, buffer=buffer,
        )

    def fused():
        return a_compA_add_b_B(
            A_comp, alpha, B, beta, alpha_is_one=alpha_value == 1.0,
            dense_output=False, buffer=buffer,
        )

    timings = {separate: [], fused: []}
    for trial in range(TRIALS):
        if trial % 2 == 0:
            operations = (separate, fused)
        else:
            operations = (fused, separate)
        for operation in operations:
            timings[operation].append(_time(operation))
    separate_ms = statistics.median(timings[separate])
    fused_ms = statistics.median(timings[fused])
    print(
        f"shape={shape!s:>14s}  {label:>13s}  "
        f"separate={separate_ms:.4f} ms  fused={fused_ms:.4f} ms  "
        f"reduction={(separate_ms - fused_ms) / separate_ms:.2%}"
    )
    A_comp.free()


def main() -> None:
    for label, alpha in CASES:
        for n in SIZES:
            _benchmark(n, label, alpha)


if __name__ == "__main__":
    main()
