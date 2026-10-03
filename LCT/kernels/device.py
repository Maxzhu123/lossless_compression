"""Model and architecture selection for a single-GPU process."""

from dataclasses import dataclass
from functools import lru_cache

import torch


@dataclass(frozen=True)
class DeviceProfile:
    model: str
    family: str
    profile: str


def model_key(name: str) -> str:
    """Normalize an exact name, preserving Ti, laptop, memory and other variants."""
    name = " ".join(name.casefold().split()).removeprefix("nvidia ")
    return name.replace(" ", "_")


# GPUs with a dedicated kernel-implementation package under ``LCT/kernels/``.
# The value names that package; ``generic`` means "no specialized package".
KERNEL_PROFILES: dict[str, str] = {
    "geforce_rtx_4090": "rtx_4090",
}


def profile_for(model: str) -> str:
    """Resolve the kernel-implementation profile for a normalized model name."""
    return KERNEL_PROFILES.get(model, "generic")


@lru_cache(maxsize=1)
def device_profile(device: torch.device) -> DeviceProfile:
    """Cache the model, family, and kernel-implementation profile of the GPU."""
    if device.type != "cuda":
        raise ValueError(f"LCT kernel dispatch requires a CUDA device, got {device}")
    properties = torch.cuda.get_device_properties(device)
    capability = (properties.major, properties.minor)
    # ROCm shares torch's CUDA namespace. Unknown architectures and vendors
    # must not accidentally select NVIDIA Hopper-specific kernels.
    families = {
        (8, 0): "ampere", (8, 6): "ampere", (8, 7): "ampere",
        (8, 9): "ada", (9, 0): "hopper",
    }
    family = families.get(capability, "generic") if torch.version.hip is None else "generic"
    model = model_key(properties.name)
    return DeviceProfile(model=model, family=family, profile=profile_for(model))
