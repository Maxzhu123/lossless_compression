"""Shape-preserving helpers around the official CUDA/Triton SplitZip codec."""
import torch

from .common import CompressedTensor, check_method, validate_tensor


class SplitZip:
    def __init__(self, calibration=None, *, chunk_size=1024):
        self.chunk_size = chunk_size
        self._codec = None
        if calibration is not None:
            self.calibrate(calibration)

    def calibrate(self, sample):
        """Fit a codebook on representative BF16 CUDA data; return coverage."""
        flat = validate_tensor(sample, cuda=True)
        if not flat.numel():
            raise ValueError("calibration sample must not be empty")
        from ._vendor.splitzip.codec_gpu import ChunkLocalSplitZipGPU
        codec = ChunkLocalSplitZipGPU(device=sample.device, chunk_size=self.chunk_size)
        with torch.cuda.device(sample.device):
            coverage = codec.calibrate(flat)
        self._codec = codec
        return coverage

    def compress(self, tensor):
        flat = validate_tensor(tensor, cuda=True)
        codec = self._codec
        if codec is None:
            raise RuntimeError("call calibrate(sample) before compressing")
        if tensor.device != codec.device:
            raise ValueError("input and calibration must be on the same CUDA device")
        with torch.cuda.device(tensor.device):
            encoded = codec.encode(flat) if flat.numel() else None
            # Keep the decode table with the payload so recalibration cannot
            # silently change the interpretation of previously encoded tensors.
            table = codec.dec_lut.clone()
        size = (encoded.compressed_bytes if encoded is not None else 0) + table.nbytes
        return CompressedTensor("splitzip", (encoded, table), tuple(tensor.shape),
                                tensor.device, size, tensor.nbytes)

    def decompress(self, data):
        check_method(data, "splitzip")
        encoded, table = data.payload
        if encoded is None:
            return torch.empty(data.shape, dtype=torch.bfloat16, device=data.device)
        from ._vendor.splitzip.codec_gpu import ChunkLocalSplitZipGPU
        codec = ChunkLocalSplitZipGPU(device=data.device, chunk_size=encoded.chunk_size)
        # Decode only reads dec_lut; the other fields satisfy upstream's check.
        codec.enc_lut = codec.enc_lut_marked = codec.common_lut = table
        codec.dec_lut = table
        with torch.cuda.device(data.device):
            return codec.decode(encoded).reshape(data.shape)


def compress(tensor, *, calibration=None, chunk_size=1024):
    """Convenience path; fitting on input is included in this call's cost."""
    return SplitZip(tensor if calibration is None else calibration,
                    chunk_size=chunk_size).compress(tensor)


def decompress(data):
    return SplitZip().decompress(data)
