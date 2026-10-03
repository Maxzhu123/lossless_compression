"""Launch configurations tuned for the RTX 4090 (Ada, sm_89).

Each list is pinned to the winning configuration measured on the RTX 4090 with
``benchmarks/prepare.py``.  Pinning matters twice over:

* the winning configuration is kept even when it is not in the generic list;
* a single-element config list is Triton's fast path -- ``Autotuner.run`` skips
  building a cache key entirely, which removes ~11 us of host time per launch.
"""

import triton

# Encode: 4 warps beat 8 at both 50M (215 vs 227 us) and 200M (784 vs 811 us).
ENCODE_AUTOTUNE_CONFIGS = [
    triton.Config({}, num_warps=4, num_stages=2),
]

# Decode: flat across stages/maxnreg; 4 warps is the clear winner.
DECODE_AUTOTUNE_CONFIGS = [
    triton.Config({}, num_warps=4, num_stages=2),
]

# Scope note: this module only configures the two kernels the RTX 4090 package
# actually owns (``encode.py`` / ``decode.py``).  The overflow compaction and
# scatter kernels are shared -- ``LCT/codec/encode.py`` and ``LCT/codec/decode.py``
# import them straight from ``LCT/kernels/generic/compaction.py`` -- so their
# launch settings and grid caps are tuned in ``LCT/codec/autotune.py``, under the
# ``rtx_4090`` profile branch.  Defining copies of them here would have no effect.

