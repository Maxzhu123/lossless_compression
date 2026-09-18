"""Build and load the local, format-compatible DFloat11 CPU encoder."""
import ctypes as C
from functools import lru_cache
import hashlib
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile

import numpy as np
import torch


@lru_cache(maxsize=1)
def _library():
    source = Path(__file__).with_name('_dfloat11_encode.c')
    key = hashlib.sha256(source.read_bytes() + platform.platform().encode()).hexdigest()[:16]
    cache = source.parent / '__pycache__'
    cache.mkdir(exist_ok=True)
    target = cache / f'dfloat11_encoder_{key}.so'
    if not target.exists():
        compiler = shutil.which('cc')
        if compiler is None:
            raise RuntimeError("DFloat11 native encoding requires a C compiler; use DFloat11(encoder='reference') instead")
        with tempfile.TemporaryDirectory(dir=cache) as directory:
            output = Path(directory) / 'encoder.so'
            subprocess.run([compiler, '-O3', '-std=c99', '-shared', '-fPIC',
                            str(source), '-o', str(output)], check=True, capture_output=True)
            output.replace(target)
    library = C.CDLL(str(target))
    library.histogram.argtypes = [C.c_void_p, C.c_size_t, C.c_void_p]
    library.histogram.restype = None
    library.encode.argtypes = [C.c_void_p, C.c_size_t, C.c_void_p, C.c_void_p,
                               C.c_uint, C.c_uint32] + [C.c_void_p] * 4
    library.encode.restype = C.c_size_t
    return library


def histogram(flat):
    counts = np.zeros(256, dtype=np.uint64)
    _library().histogram(flat.data_ptr(), flat.numel(), counts.ctypes.data)
    return {symbol: int(count) for symbol, count in enumerate(counts) if count}


def encode(flat, codec, counter):
    lengths = np.zeros(256, dtype=np.uint8)
    codes = np.zeros(256, dtype=np.uint32)
    for symbol in counter:
        lengths[symbol], codes[symbol] = codec._table[symbol]
    eof_length, eof_code = codec._table[codec._eof]
    bits = sum(count * int(lengths[symbol]) for symbol, count in counter.items())
    n_bytes = (bits + 7) // 8
    if n_bytes > 2**31 - 1:
        raise ValueError('DFloat11 CUDA decoder uses signed 32-bit byte counts')
    blocks = (n_bytes + 4095) // 4096
    # Guard bytes match the wrapper's original decoder lookahead padding.
    encoded = torch.zeros(n_bytes + 12, dtype=torch.uint8)
    gaps = torch.zeros(blocks * 320 + 1, dtype=torch.uint8)
    positions = torch.empty(blocks + 1, dtype=torch.uint32)
    sm = torch.empty(flat.numel(), dtype=torch.uint8)
    written = _library().encode(
        flat.data_ptr(), flat.numel(), lengths.ctypes.data, codes.ctypes.data,
        eof_length, eof_code, encoded.data_ptr(), sm.data_ptr(),
        positions.data_ptr(), gaps.data_ptr(),
    )
    if written != n_bytes:
        raise RuntimeError('DFloat11 native encoder produced an unexpected byte count')
    return encoded, sm, positions, gaps, n_bytes
