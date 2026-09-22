"""Lossless saved-activation compression, including native FlashAttention state."""
import weakref

import torch

from LCT.LCTensor import MyCompressed
from LCT.compress import compress, decompress
from LCT.comp_tensor import CompressedTensor
from LCT.dist_configs import act_dist


class _SavedActivation:
    def __init__(self, tensor, buffer):
        self.packed = compress(tensor.detach(), distribution=act_dist, buffer=buffer)
        # Unpack may be called more than once (including retain_graph=True).
        # Release the arena allocation only when autograd releases this owner.
        self._cleanup = weakref.finalize(self, self.packed.free)


class ActivationCompression(torch.autograd.graph.saved_tensors_hooks):
    """Compress large BF16 saved tensors; leave parameters and FP32 stats alone.

    Native SDPA/FlashAttention still owns its forward/backward implementation.
    Its BF16 Q/K/V/output saves use this hook, while FP32 log-sum-exp and RNG
    state retain their original representation. No attention matrix is created.
    """

    def __init__(self, parameters=(), *, buffer=None, min_elements=65536):
        self.buffer = buffer
        self.min_elements = min_elements
        self.protected = {
            (p.device, p.untyped_storage().data_ptr())
            for p in parameters if not isinstance(p, MyCompressed)
        }
        self.packed_count = 0
        super().__init__(self.pack, self.unpack)

    def pack(self, tensor):
        if isinstance(tensor, MyCompressed):
            # Reuse the persistent representation. Return dense data on unpack
            # because autograd detaches tensors returned by saved-tensor hooks.
            return tensor.x
        if (not tensor.is_cuda
                or tensor.dtype != torch.bfloat16 or tensor.numel() < self.min_elements
                or tensor.numel() == 0
                or (tensor.device, tensor.untyped_storage().data_ptr()) in self.protected):
            return tensor
        self.packed_count += 1
        return _SavedActivation(tensor, self.buffer)

    @staticmethod
    def unpack(value):
        if isinstance(value, _SavedActivation):
            return decompress(value.packed)
        return decompress(value) if isinstance(value, CompressedTensor) else value
