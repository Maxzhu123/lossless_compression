"""Dense counterpart of the compressed scalar multiply-add weight update."""
import triton
from triton import language as tl

from LCT.kernels.pointwise_scalar import _scaled_sum


@triton.jit
def muon_dense_update_kernel(weight, update, decay, neg_lr, numel, columns,
                             ws0, ws1, us0, us1,
                             ALPHA_IS_ONE: tl.constexpr, BLOCK: tl.constexpr):
    indices = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    rows, cols = indices // columns, indices % columns
    wp = weight + rows * ws0 + cols * ws1
    up = update + rows * us0 + cols * us1
    w = tl.load(wp, indices < numel, 0).to(tl.float32)
    u = tl.load(up, indices < numel, 0).to(tl.float32)
    result = _scaled_sum(w, u, tl.load(decay), tl.load(neg_lr), True, ALPHA_IS_ONE)
    tl.store(wp, result, indices < numel)
