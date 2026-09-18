"""DFloat11-compatible native/reference encoding and upstream CUDA decoding."""
from dataclasses import dataclass

import torch

from .common import CompressedTensor, check_method, validate_tensor


@dataclass
class Payload:
    luts: torch.Tensor
    encoded: torch.Tensor
    sign_mantissa: torch.Tensor
    positions: torch.Tensor
    gaps: torch.Tensor
    n_bytes: int
    shared_bytes: int


class DFloat11:
    def __init__(self, *, encoder="native"):
        if encoder not in {"native", "reference"}:
            raise ValueError("encoder must be native or reference")
        self.encoder = encoder

    def compress(self, tensor):
        """Fit and encode on CPU, then stage buffers to the input device.

        The released decoder reserves exponents 240..255 for LUT links, so
        reject these rather than silently corrupting huge values/Inf/NaN.
        """
        flat = validate_tensor(tensor).cpu()
        if not flat.numel():
            return CompressedTensor("dfloat11", None, tuple(tensor.shape), tensor.device, 0, 0)
        if flat.numel() > 2**31 - 1:
            raise ValueError("DFloat11 CUDA decoder uses signed 32-bit element counts")
        from ._vendor.dfloat11.dfloat11_utils import get_codec, get_32bit_codec, get_luts, encode_weights
        if self.encoder == "native":
            from ._dfloat11_encoder import histogram, encode
            counter = histogram(flat)
        else:
            _, counter = get_codec(flat)
        if any(symbol >= 240 for symbol in counter):
            raise ValueError("upstream DFloat11 decoder cannot represent BF16 exponents 240..255 (including Inf/NaN)")
        codec, _, table = get_32bit_codec(counter)
        luts = get_luts(table)
        if luts.shape[0] > 17:
            raise ValueError("DFloat11 decoder supports at most 16 lookup tables plus lengths")
        if self.encoder == "native":
            encoded, sm, positions, gaps, n_bytes = encode(flat, codec, counter)
        else:
            encoded, sm, positions, gaps, _ = encode_weights([flat], codec, 8, 512)
            n_bytes = encoded.numel()
        if n_bytes > 2**31 - 1:
            raise ValueError("DFloat11 CUDA decoder uses signed 32-bit byte counts")
        sizes = positions.to(torch.int64).diff()
        shared_bytes = 512 * 4 + 4 + int(sizes.max()) * 2
        # Upstream reads one gap byte beyond the final packed entry, and uses
        # lookahead bytes at the end of the code stream. Supply guard storage.
        if self.encoder == "reference":
            encoded = torch.cat((encoded, torch.zeros(12, dtype=torch.uint8)))
            gaps = torch.cat((gaps, torch.zeros(1, dtype=torch.uint8)))
        size = sum(t.nbytes for t in (luts, encoded, sm, positions, gaps)) - 13
        buffers = [t.to(tensor.device) for t in (luts, encoded, sm, positions, gaps)]
        return CompressedTensor("dfloat11", Payload(*buffers, n_bytes, shared_bytes),
                                tuple(tensor.shape), tensor.device, size, tensor.nbytes)

    def decompress(self, data):
        check_method(data, "dfloat11")
        if data.payload is None:
            return torch.empty(data.shape, dtype=torch.bfloat16, device=data.device)
        p = data.payload
        if data.device.type != "cuda":
            return _decode_cpu(data)
        from ._cuda import launch_decode
        output = torch.empty(data.shape, dtype=torch.bfloat16, device=data.device)
        launch_decode([p.luts, p.encoded, p.sign_mantissa, p.positions, p.gaps, output],
                      n_bytes=p.n_bytes, n_elements=output.numel(), shared_bytes=p.shared_bytes)
        return output


def _decode_cpu(data):
    """Slow reference decoder for validation; not the paper's GPU runtime."""
    p = data.payload
    tables = p.luts.tolist()
    stream = p.encoded.tolist()
    bit = 0
    exponents = []
    for _ in range(data.original_bytes // 2):
        offset = bit
        lut = 0
        while True:
            byte, shift = divmod(offset, 8)
            code = ((stream[byte] << 8 | stream[byte + 1]) >> (8 - shift)) & 255
            symbol = tables[lut][code]
            if symbol < 240:
                break
            lut = 256 - symbol
            offset += 8
        exponents.append(symbol)
        bit += tables[-1][symbol]
    sm = p.sign_mantissa.to(torch.int16)
    exp = torch.tensor(exponents, dtype=torch.int16)
    raw = ((sm & 128) << 8) | (exp << 7) | (sm & 127)
    return raw.view(torch.bfloat16).reshape(data.shape)


def compress(tensor):
    return DFloat11().compress(tensor)


def decompress(data):
    return DFloat11().decompress(data)
