"""Compose typed operations and select model overrides before family defaults."""

import torch

from ..kernels.generic.decode import decode
from ..kernels.generic.encode import encode, encode_components
from .device import device_profile
from .interfaces import KernelBackend


GENERIC = KernelBackend(
    encode=encode, encode_components=encode_components, decode=decode,
)

# Register only working implementations. Every GPU currently uses the existing
# generic kernels. Specializations can override individual fields with replace().
FAMILY_BACKENDS: dict[str, KernelBackend] = {"generic": GENERIC}
MODEL_BACKENDS: dict[str, KernelBackend] = {}


def backend_for(device: torch.device) -> KernelBackend:
    """Resolve exact model -> family -> generic without caching mutable registries."""
    profile = device_profile(device)
    if profile.model in MODEL_BACKENDS:
        return MODEL_BACKENDS[profile.model]
    return FAMILY_BACKENDS.get(profile.family, FAMILY_BACKENDS["generic"])
