"""Correctness and regression checks for matrix-tiled compressed weights."""

import torch

from compress.code_storage import Distribution, DistType, NoiseLevel
from compress.compress import (
    compress_matrix, compressed_linear, compressed_matmul, decompress,
)
from compress.tensor_buffer import TensorBuffer


def _assert_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.02)


def _check(n: int, k: int, m: int, *, buffer=None, arbitrary=False) -> None:
    generator = torch.Generator(device="cuda").manual_seed(n + k + m)
    if arbitrary:
        bits = torch.randint(
            -32768, 32767, (n, k), dtype=torch.int16,
            device="cuda", generator=generator,
        )
        weight = bits.view(torch.bfloat16)
        distribution = Distribution(
            DistType.GAUSSIAN, noise_level=NoiseLevel.HIGH,
        )
    else:
        weight = (
            torch.randn((n, k), device="cuda", generator=generator) * 0.25
        ).to(torch.bfloat16)
        distribution = Distribution(DistType.GAUSSIAN, 0.25)
    activations = torch.randn(
        (m, k), dtype=torch.bfloat16, device="cuda", generator=generator,
    )
    left = torch.randn(
        (m, n), dtype=torch.bfloat16, device="cuda", generator=generator,
    )
    encoded = compress_matrix(weight, distribution, buffer)
    assert torch.equal(weight.view(torch.int16), decompress(encoded).view(torch.int16))
    if not arbitrary:
        _assert_close(compressed_linear(activations, encoded), activations @ weight.T)
        _assert_close(compressed_matmul(left, encoded), left @ weight)
    encoded.free()


def main() -> None:
    _check(128, 128, 7)
    _check(300, 270, 17)
    _check(257, 513, 17, buffer=TensorBuffer(32 * 1024 * 1024, device="cuda"))
    _check(77, 91, 5, arbitrary=True)
    print("passed")


if __name__ == "__main__":
    main()
