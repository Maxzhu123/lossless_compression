"""Compose typed operations and select model overrides before family defaults."""

import torch

from ..kernels.generic.decode import decode as decode_triton
from ..kernels.generic.encode import encode, encode_components
from . import config
from .device import device_profile
from .interfaces import KernelBackend


def decode(data):
    """Select the decoder from the runtime configuration without probing support."""
    if config.use_tilelang:
        from ..kernels.tilelang.decode import decode as decode_tilelang
        return decode_tilelang(data)
    else:
        return decode_triton(data)




GENERIC = KernelBackend(
    encode=encode, encode_components=encode_components, decode=decode,
)

# The generic backend composes shared encoding with the optimized decoder.
# Specializations can override individual fields with replace().
FAMILY_BACKENDS: dict[str, KernelBackend] = {"generic": GENERIC}
MODEL_BACKENDS: dict[str, KernelBackend] = {}


def backend_for(device: torch.device) -> KernelBackend:
    """Resolve exact model -> family -> generic without caching mutable registries."""
    profile = device_profile(device)
    if profile.model in MODEL_BACKENDS:
        return MODEL_BACKENDS[profile.model]
    return FAMILY_BACKENDS.get(profile.family, FAMILY_BACKENDS["generic"])
