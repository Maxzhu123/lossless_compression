"""RTX 4090 (Ada, sm_89) kernel profile.

Holds the Triton kernels selected for the NVIDIA GeForce RTX 4090.  The kernels
are body-identical to their :mod:`LCT.kernels.generic` counterparts -- the Ada
delta is launch tuning, not a different algorithm, so the storage format and
compression ratios are unchanged:

* ``num_warps=4`` blocks, which beat 8 warps at every measured geometry
  (encode 215 vs 227 us at 50M, 784 vs 811 us at 200M);
* single-config autotune lists, which are also Triton's fast path -- with one
  config ``Autotuner.run`` skips building a cache key, saving ~11 us of host
  time per launch, which matters because the encode path launches several
  kernels per call.

Overflow-pass tuning for Ada (grid caps, scatter tile) lives in
:mod:`LCT.codec.autotune`, because those kernels are shared and the pipeline
imports them directly.

Kernels are selected by :func:`LCT.kernels.backends.backend_for` through the
``rtx_4090`` entry in :data:`LCT.kernels.device.KERNEL_PROFILES`.
"""

from .backend import BACKEND

__all__ = ["BACKEND"]
