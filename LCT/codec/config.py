"""Runtime kernel selection shared by codec dispatch.

Set ``LCT.codec.config.use_tilelang = False`` to use the Triton decoder.
When enabled, TileLang is assumed to be installed and usable on the device.
"""

import os

use_tilelang: bool = False

# Retune on each program run, retaining compiled kernels.
reset_cache: bool = True

if use_tilelang and reset_cache:
    os.environ["TILELANG_AUTO_TUNING_DISABLE_CACHE"] = "1"
