"""Lossless BF16 wrappers for NVIDIA nvCOMP's CUDA codecs."""
import torch

from .common import CompressedTensor, check_method, validate_tensor


ALGORITHMS = {
    "nvcomp_lz4": "LZ4",
    "nvcomp_cascaded": "Cascaded",
    "nvcomp_bitcomp": "Bitcomp",
}


class NVComp:
    def __init__(self, algorithm="LZ4"):
        if algorithm not in ALGORITHMS.values():
            raise ValueError("algorithm must be LZ4, Cascaded, or Bitcomp")
        self.algorithm = algorithm
        self.method = f"nvcomp_{algorithm.lower()}"

    def _codec(self, device):
        # Keep the optional native dependency out of ordinary baseline imports.
        from nvidia import nvcomp

        stream = torch.cuda.current_stream(device).cuda_stream
        # Match the byte view explicitly; Cascaded's wider default can drop
        # trailing bytes when the input is not a whole number of default words.
        codec = nvcomp.Codec(algorithm=self.algorithm, data_type="|u1", device_id=device.index,
                             cuda_stream=stream)
        return nvcomp, codec, stream

    def compress(self, tensor):
        flat = validate_tensor(tensor, cuda=True)
        with torch.cuda.device(tensor.device):
            if flat.numel():
                nvcomp, codec, stream = self._codec(tensor.device)
                source = nvcomp.as_array(flat.view(torch.uint8), cuda_stream=stream)
                storage = torch.empty(codec.get_max_comp_buffer_size(source),
                                      dtype=torch.uint8, device=tensor.device)
                destination = nvcomp.as_array(storage, cuda_stream=stream)
                encoded = codec.encode(source, out=destination)
                # A slice alone would retain the maximum-sized allocation.
                payload = storage[:encoded.buffer_size].clone()
            else:
                payload = torch.empty(0, dtype=torch.uint8, device=tensor.device)
        return CompressedTensor(self.method, payload, tuple(tensor.shape),
                                tensor.device, payload.nbytes, tensor.nbytes)

    def decompress(self, data):
        check_method(data, self.method)
        with torch.cuda.device(data.device):
            raw = torch.empty(data.original_bytes, dtype=torch.uint8, device=data.device)
            if data.original_bytes:
                nvcomp, codec, stream = self._codec(data.device)
                source = nvcomp.as_array(data.payload, cuda_stream=stream)
                destination = nvcomp.as_array(raw, cuda_stream=stream)
                codec.decode(source, out=destination)
        return raw.view(torch.bfloat16).reshape(data.shape)


def compress(tensor, *, algorithm="LZ4"):
    return NVComp(algorithm).compress(tensor)


def decompress(data):
    if not isinstance(data, CompressedTensor) or data.method not in ALGORITHMS:
        raise TypeError("expected an nvCOMP CompressedTensor")
    return NVComp(ALGORITHMS[data.method]).decompress(data)
