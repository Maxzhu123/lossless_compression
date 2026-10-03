"""Assemble the RTX 4090 kernel backend."""

from ..backends import KernelBackend
from .decode import _decode_kernel
from .encode import _encode_kernel

# Both kernels are Triton; the TileLang flags stay False.  Each ships a single
# measured launch configuration, which is Triton's fast path: with one config
# ``Autotuner.run`` skips cache-key construction entirely.
BACKEND = KernelBackend(encode=_encode_kernel, decode=_decode_kernel)
