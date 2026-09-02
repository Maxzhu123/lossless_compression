"""Helpers for sampling from codec distribution objects for visualisation."""

import math

import torch

from compress.code_storage import Distribution, DistType


def _sample_continuous(
    dist: Distribution, n: int, generator: torch.Generator, device: str = "cuda",
) -> torch.Tensor:
    if dist.family == DistType.GAUSSIAN:
        values = torch.randn(
            n, device=device, dtype=torch.float32, generator=generator
        )
        return values * dist.param + dist.mean

    if dist.family == DistType.LAPLACE:
        u = torch.rand(
            n, device=device, dtype=torch.float32, generator=generator
        ) - 0.5
        values = -dist.param * torch.sign(u) * torch.log1p(-2.0 * u.abs())
        return values + dist.mean

    if dist.family == DistType.GAMMA:
        if dist.mean <= 0.0:
            raise ValueError("Gamma scale must be positive; use mean=scale")
        shape = torch.tensor(dist.param, device=device, dtype=torch.float32)
        rate = torch.tensor(1.0 / dist.mean, device=device, dtype=torch.float32)
        values = torch.distributions.Gamma(shape, rate).sample((n,))
        return values

    if dist.family == DistType.POLYNOMIAL:
        if dist.mean <= 1.0:
            raise ValueError("Polynomial tail exponent must be >1; use mean=exponent")
        exponent = dist.mean
        scale = dist.param
        tail_probability = 0.01
        tail_start = -scale * math.log(tail_probability)
        u = torch.rand(n, device=device, dtype=torch.float32, generator=generator)
        body = u < (1.0 - tail_probability)
        values = torch.empty_like(u)
        values[body] = -scale * torch.log1p(-u[body])
        values[~body] = tail_start * (
            tail_probability / (1.0 - u[~body])
        ) ** (1.0 / (exponent - 1.0))
        signs = torch.randint(
            0, 2, (n,), device=device, dtype=torch.int8, generator=generator
        ).to(torch.float32)
        return values * (signs * 2.0 - 1.0)

    # DistType.EMPIRICAL: exponential body plus power-law tail.
    tail_probability = 0.05
    tail_alpha = 2.8
    tail_start = -dist.param * math.log(tail_probability)
    u = torch.rand(n, device=device, dtype=torch.float32, generator=generator)
    body = u < (1.0 - tail_probability)
    values = torch.empty_like(u)
    values[body] = -dist.param * torch.log1p(-u[body])
    values[~body] = tail_start * (
        tail_probability / (1.0 - u[~body])
    ) ** (1.0 / (tail_alpha - 1.0))
    signs = torch.randint(
        0, 2, (n,), device=device, dtype=torch.int8, generator=generator
    ).to(torch.float32)
    return values * (signs * 2.0 - 1.0)


def sample_distribution(
    dist: Distribution,
    n: int,
    device: str = "cuda",
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample ``n`` bfloat16 values from a codec ``Distribution``."""
    if generator is None:
        generator = torch.Generator(device=device).manual_seed(0)

    values = _sample_continuous(dist, n, generator, device)
    if dist.zero_prob > 0.0:
        zeros = torch.rand(
            n, device=device, dtype=torch.float32, generator=generator
        ) < dist.zero_prob
        values = values.clone()
        values[zeros] = 0.0

    return values.to(torch.bfloat16)
