"""Select individual codec kernels, with model, profile and family overrides.

Resolution order is most-specific first:

1. ``MODEL_BACKENDS`` -- an exact GPU model registration;
2. ``KERNEL_PROFILES`` (via :func:`device_profile`) -- a dedicated kernel
   package such as ``rtx_4090``;
3. ``FAMILY_BACKENDS`` -- an architecture family such as ``hopper``;
4. the generic backend, or TileLang when ``config.use_tilelang`` is set.
"""

from dataclasses import dataclass
from functools import cache
from typing import Any, Callable

import torch

from .generic.decode import _decode_kernel
from .generic.encode import _encode_kernel
from ..codec import config
from .device import device_profile


@dataclass(frozen=True)
class KernelBackend:
    # Triton and TileLang expose different compiled-kernel launch interfaces.
    encode: Any
    decode: Any
    encode_tilelang: bool = False
    decode_tilelang: bool = False


GENERIC = KernelBackend(encode=_encode_kernel, decode=_decode_kernel)

# Overrides replace individual kernels while retaining the shared codec pipeline.
FAMILY_BACKENDS: dict[str, KernelBackend] = {"generic": GENERIC}
MODEL_BACKENDS: dict[str, KernelBackend] = {}


@cache
def _tilelang_backend() -> KernelBackend:
    """Import TileLang only when its kernels are selected."""
    from .tilelang.encode import encode_kernel
    from .tilelang.decode import decode_kernel

    return KernelBackend(encode=encode_kernel, decode=decode_kernel,
                         encode_tilelang=True, decode_tilelang=True)


@cache
def _rtx_4090_backend() -> KernelBackend:
    """Import the RTX 4090 kernel package only when that GPU is present."""
    from .rtx_4090 import BACKEND

    return BACKEND


# Kernel-implementation profiles declared in ``device.KERNEL_PROFILES``.
PROFILE_FACTORIES: dict[str, Callable[[], KernelBackend]] = {
    "rtx_4090": _rtx_4090_backend,
}


def backend_for(device: torch.device) -> KernelBackend:
    """Resolve exact model -> GPU profile -> family -> configured default."""
    profile = device_profile(device)
    if profile.model in MODEL_BACKENDS:
        return MODEL_BACKENDS[profile.model]
    factory = PROFILE_FACTORIES.get(profile.profile)
    if factory is not None:
        return factory()
    if profile.family != "generic" and profile.family in FAMILY_BACKENDS:
        return FAMILY_BACKENDS[profile.family]
    default = FAMILY_BACKENDS["generic"]
    return _tilelang_backend() if config.use_tilelang and default is GENERIC else default
