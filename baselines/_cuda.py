"""Launch the vendored DFloat11 PTX through the CUDA driver (no CuPy)."""
import ctypes as C
import os
from pathlib import Path

import torch

_driver = None
_modules = {}


def _check(result):
    if result:
        raise RuntimeError(f"DFloat11 CUDA driver call failed with error {result}")


def launch_decode(buffers, *, n_bytes, n_elements, shared_bytes):
    global _driver
    if _driver is None:
        _driver = C.CDLL("nvcuda.dll" if os.name == "nt" else "libcuda.so.1")
        _driver.cuModuleLoad.argtypes = [C.POINTER(C.c_void_p), C.c_char_p]
        _driver.cuModuleGetFunction.argtypes = [C.POINTER(C.c_void_p), C.c_void_p, C.c_char_p]
        _driver.cuLaunchKernel.argtypes = [C.c_void_p] + [C.c_uint] * 7 + [
            C.c_void_p, C.POINTER(C.c_void_p), C.c_void_p,
        ]
        _driver.cuFuncSetAttribute.argtypes = [C.c_void_p, C.c_int, C.c_int]
    device = buffers[0].device
    with torch.cuda.device(device):
        # PyTorch owns the primary context and allocations. Its current stream
        # orders this launch with preceding copies and subsequent torch ops.
        stream = torch.cuda.current_stream(device)
        if device not in _modules:
            module, function = C.c_void_p(), C.c_void_p()
            path = Path(__file__).parent / "_vendor/dfloat11/decode.ptx"
            _check(_driver.cuModuleLoad(C.byref(module), os.fsencode(path)))
            _check(_driver.cuModuleGetFunction(C.byref(function), module, b"decode"))
            _modules[device] = (module, function)
        function = _modules[device][1]
        # CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
        _check(_driver.cuFuncSetAttribute(function, 8, shared_bytes))
        values = [C.c_uint64(t.data_ptr()) for t in buffers]
        values += [C.c_int(buffers[0].shape[0]), C.c_int(n_bytes), C.c_int(n_elements)]
        args = (C.c_void_p * len(values))(*[C.cast(C.pointer(v), C.c_void_p) for v in values])
        _check(_driver.cuLaunchKernel(function, (n_bytes + 4095) // 4096, 1, 1,
                                     512, 1, 1, shared_bytes, stream.cuda_stream, args, None))
        for tensor in buffers:
            tensor.record_stream(stream)
