"""Shared Triton autotuning configurations."""

import triton


ESTIMATE_CENTER_AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK": 4096}, num_warps=4, num_stages=2),
]
ENCODE_AUTOTUNE_CONFIGS = [
    triton.Config({}, num_warps=8, num_stages=2),
    triton.Config({}, num_warps=8, num_stages=2, maxnreg=64),
    triton.Config({}, num_warps=4, num_stages=2, maxnreg=64),
    triton.Config({}, num_warps=4, num_stages=3, maxnreg=64),
    triton.Config({}, num_warps=4, num_stages=4, maxnreg=128),
]
COMPACT_BAD_STREAMS_AUTOTUNE_CONFIGS = [
    triton.Config({}, num_warps=1, num_stages=2),
    triton.Config({}, num_warps=2, num_stages=2),
    triton.Config({}, num_warps=2, num_stages=3),
]
COMPACT_EXTRA_AUTOTUNE_CONFIGS = [
    triton.Config({}, num_warps=1, num_stages=2),
    triton.Config({}, num_warps=2, num_stages=2),
    triton.Config({}, num_warps=4, num_stages=2),
    triton.Config({}, num_warps=2, num_stages=3),
]
SCATTER_FALLBACK_AUTOTUNE_CONFIGS = [
    triton.Config({}, num_warps=1, num_stages=2),
    triton.Config({}, num_warps=2, num_stages=2),
    triton.Config({}, num_warps=4, num_stages=2),
    triton.Config({}, num_warps=2, num_stages=3),
]
DECODE_AUTOTUNE_CONFIGS = [
    triton.Config({}, num_warps=8, num_stages=2),
    triton.Config({}, num_warps=4, num_stages=5, maxnreg=64),
    triton.Config({}, num_warps=2, num_stages=2, maxnreg=64),
    triton.Config({}, num_warps=2, num_stages=3, maxnreg=64),
    triton.Config({}, num_warps=1, num_stages=2, maxnreg=None),
    triton.Config({}, num_warps=1, num_stages=3, maxnreg=None),
]
