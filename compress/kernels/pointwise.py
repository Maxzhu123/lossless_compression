"""Pointwise epilogues inlined into codec kernels."""

import triton
from triton import language as tl


DECODE_ONLY = tl.constexpr(0)
ADD_DENSE = tl.constexpr(1)
ADD_COMPONENTS = tl.constexpr(2)


@triton.jit
def store_decoded(
    output, auxiliary, other, offset, packed, mask, MODE: tl.constexpr,
):
    """Store decoded BF16 bits through the selected pointwise epilogue."""
    if MODE == DECODE_ONLY:
        tl.store(output + offset, packed.to(tl.int16), mask=mask)
    else:
        value = packed.to(tl.int16).to(tl.bfloat16, bitcast=True)
        rhs = tl.load(other + offset, mask=mask, other=0.0)
        result = (value + rhs).to(tl.bfloat16)
        if MODE == ADD_DENSE:
            tl.store(output + offset, result, mask=mask)
        else:
            bits = result.to(tl.int16, bitcast=True).to(tl.int32)
            sign_mantissa = (bits & 0x7F) | ((bits >> 8) & 0x80)
            exponent = (bits >> 7) & 0xFF
            tl.store(output + offset, sign_mantissa.to(tl.uint8), mask=mask)
            tl.store(auxiliary + offset, exponent.to(tl.uint8), mask=mask)
