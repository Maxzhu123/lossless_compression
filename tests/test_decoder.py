"""Bitwise decoder regressions against source tensors and the Triton decoder.

Run with: python -m unittest discover -s tests -p 'test_decoder.py'
"""

import importlib
import unittest
from unittest.mock import patch

import torch

from LCT.comp_format import DistType, Distribution, NoiseLevel, StorageLayout
from LCT.compress import compress, decompress, compA_add_B, compA_mul_B, a_compA_add_b_B, a_compA_add_compB
from LCT.compression.huffman_tables import get_distribution_tables
from LCT.kernels.generic.decode import decode as decode_triton, _restore_fallback
from LCT.kernels.tilelang.decode import decode as decode_tilelang, _decode_kernel
from LCT.tensor_buffer import TensorBuffer, verify_buffer


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class DecoderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(123)
        cls.arena = TensorBuffer(32 * 1024**2, device="cuda")

    def assert_bits(self, actual, expected):
        if not isinstance(actual, torch.Tensor):
            actual = decompress(actual)
        self.assertEqual(actual.shape, expected.shape)
        self.assertTrue(torch.equal(actual.contiguous().view(torch.int16), expected.contiguous().view(torch.int16)))

    def roundtrip(self, source, distribution):
        for arena in (None, self.arena):
            with self.subTest(buffered=arena is not None):
                encoded = compress(source, distribution, arena)
                try:
                    self.assert_bits(decompress(encoded), source)
                    if encoded.layout == StorageLayout.COMPRESSED:
                        self.assert_bits(decode_triton(encoded), source)
                        self.assert_bits(decode_tilelang(encoded), source)
                finally:
                    encoded.free()
        verify_buffer(self.arena)

    def test_every_bf16_representation_and_codebook(self):
        bits = torch.arange(65536, dtype=torch.int32, device="cuda").to(torch.int16)
        source = bits.repeat(3)[:131329].view(torch.bfloat16)
        for family in DistType:
            for noise in NoiseLevel:
                with self.subTest(family=family, noise=noise):
                    self.roundtrip(source, Distribution(family, noise_level=noise))

    def test_tails_block_boundaries_and_noncontiguous(self):
        dist = Distribution(DistType.GAUSSIAN)
        for n in (1, 2, 3, 15, 255, 256, 257, 511, 65535, 65536, 65537, 131071):
            with self.subTest(n=n):
                self.roundtrip(torch.randn(n, device="cuda").bfloat16(), dist)
        self.roundtrip(torch.randn(513, 257, device="cuda").bfloat16().T, dist)

    def test_centers_zeros_and_long_overflow(self):
        dist = Distribution(DistType.GAUSSIAN)
        for exponent in (-100, 0, 100):
            source = (torch.randn(65537, device="cuda") * 2.0**exponent).bfloat16()
            self.roundtrip(source, dist)
        source = torch.randn(131329, device="cuda").bfloat16()
        source[::2] = 0
        source[1::7] = -0.0
        self.roundtrip(source, Distribution(DistType.GAUSSIAN, zero_prob=0.5))
        self.roundtrip(torch.zeros_like(source), dist)
        self.roundtrip((torch.randn_like(source) + 50).bfloat16(), dist)
        bits = torch.arange(4_194_561, dtype=torch.int32, device="cuda").to(torch.int16)
        self.roundtrip(bits.view(torch.bfloat16), dist)

    def test_stored_geometry_and_supplied_centers(self):
        encoder = importlib.import_module("LCT.kernels.generic.encode")
        dist = Distribution(DistType.GAUSSIAN)
        source = torch.randn(65537, device="cuda").bfloat16()
        with patch.object(encoder, "geometry", return_value=(32768, 256, 128, 12)):
            self.roundtrip(source, dist)
        storage_size = 131072
        for value in (-128, -127, -1, 0, 1, 127):
            side = torch.empty(storage_size, dtype=torch.uint8, device="cuda")
            center = torch.tensor([value], dtype=torch.int32, device="cuda")
            encoded = encoder.encode_components(source.view(torch.int16), side, storage_size, dist, self.arena, source.shape, precomputed=False, center=center, logical_numel=source.numel())
            try:
                self.assert_bits(decode_tilelang(encoded), source)
                self.assert_bits(decode_triton(encoded), source)
            finally:
                encoded.free()

    def test_wide_shape_arithmetic_and_cuda_graph(self):
        source = torch.randn(65537, device="cuda").bfloat16()
        encoded = compress(source, Distribution(DistType.GAUSSIAN), self.arena)
        try:
            _, base, rare = get_distribution_tables(encoded.distribution)
            capability = torch.cuda.get_device_capability()
            target = f"cuda -arch=sm_{capability[0]}{capability[1]}"
            kernel = _decode_kernel(*encoded.codec_geometry, rare, target, "int64")
            output = torch.empty_like(source, dtype=torch.int16)
            kernel(encoded.data, encoded.sign_mantissa, base, encoded.center, output)
            _restore_fallback(encoded, output)
            self.assertTrue(torch.equal(output, source.view(torch.int16)))
            for _ in range(3):
                decode_tilelang(encoded)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                result = decode_tilelang(encoded)
            for _ in range(3):
                graph.replay()
            self.assert_bits(result, source)
        finally:
            encoded.free()

    def test_fused_operation_outputs(self):
        x = torch.randn(131329, device="cuda").bfloat16()
        x[::3] *= 512
        y = torch.randn_like(x)
        alpha = torch.tensor([0.998], device="cuda")
        beta = torch.tensor([-0.02], device="cuda")
        dist = Distribution(DistType.GAUSSIAN)
        for arena in (None, self.arena):
            a, b = compress(x, dist, arena), compress(y, dist, arena)
            try:
                cases = [
                    (lambda dense: compA_add_B(a, y, dense_output=dense, buffer=arena), x + y),
                    (lambda dense: compA_mul_B(a, y, dense_output=dense, buffer=arena), x * y),
                    (lambda dense: a_compA_add_b_B(a, alpha, y, beta, dense_output=dense, buffer=arena), torch.addcmul(y.float() * beta, x.float(), alpha).bfloat16()),
                    (lambda dense: a_compA_add_compB(a, alpha, b, dense_output=dense, buffer=arena), torch.addcmul(y.float(), x.float(), alpha).bfloat16()),
                ]
                for function, expected in cases:
                    for dense in (True, False):
                        result = function(dense)
                        try:
                            self.assert_bits(result, expected)
                        finally:
                            if not dense:
                                result.free()
            finally:
                a.free()
                b.free()
        verify_buffer(self.arena)


if __name__ == "__main__":
    unittest.main()
