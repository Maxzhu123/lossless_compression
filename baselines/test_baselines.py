"""Run with python -m unittest baselines.test_baselines -v."""
import contextlib
import io
import unittest

import torch

from baselines import DFloat11, SplitZip, ZipNN
from baselines.common import storage_bytes


class BaselineTests(unittest.TestCase):
    def assert_roundtrip(self, codec, x):
        before = x.detach().clone()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            packed = codec.compress(x)
            restored = codec.decompress(packed)
        self.assertEqual(x.shape, restored.shape)
        self.assertEqual(x.device, restored.device)
        self.assertEqual(x.dtype, restored.dtype)
        self.assertFalse(restored.requires_grad)
        self.assertTrue(torch.equal(x.contiguous().view(torch.int16),
                                    before.contiguous().view(torch.int16)))
        self.assertTrue(torch.equal(x.contiguous().view(torch.int16),
                                    restored.contiguous().view(torch.int16)))
        self.assertGreaterEqual(packed.memory_size(), packed.compressed_bytes)
        return packed

    def test_cpu_roundtrips(self):
        for codec in (DFloat11(), ZipNN(threads=1)):
            for x in (torch.empty(0, 3, dtype=torch.bfloat16),
                      torch.tensor(-0., dtype=torch.bfloat16),
                      torch.zeros(257, dtype=torch.bfloat16),
                      torch.randn(19, 17, dtype=torch.bfloat16).T,
                      torch.randn(8193, dtype=torch.bfloat16)):
                with self.subTest(codec=type(codec).__name__, shape=x.shape):
                    self.assert_roundtrip(codec, x)

    def test_zipnn_all_bit_patterns(self):
        x = torch.arange(-32768, 32768, dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
        self.assert_roundtrip(ZipNN(threads=1), x)

    def test_validation(self):
        with self.assertRaises(ValueError):
            ZipNN(threads=-1)
        for codec in (DFloat11(), SplitZip(), ZipNN(threads=1)):
            with self.assertRaises(TypeError):
                codec.compress(torch.ones(5))
        with self.assertRaises(ValueError):
            DFloat11().compress(torch.tensor([float('inf')], dtype=torch.bfloat16))
        with self.assertRaises(ValueError):
            SplitZip().calibrate(torch.ones(5, dtype=torch.bfloat16))

    def test_storage_counts_allocations_once(self):
        x = torch.empty(100, dtype=torch.uint8)
        self.assertEqual(storage_bytes((x[:2], x[2:])), 100)

    def test_benchmark(self):
        from baselines.benchmark import benchmark
        result = benchmark(ZipNN(threads=1), torch.randn(513, dtype=torch.bfloat16), iterations=2)
        self.assertEqual(result.original_bytes, 1026)
        self.assertGreater(result.compress_ms, 0)
        with self.assertRaises(ValueError):
            benchmark(DFloat11(), torch.empty(0), iterations=0)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_cuda_roundtrips(self):
        sample = torch.randn(4096, dtype=torch.bfloat16, device='cuda')
        for codec in (SplitZip(sample), DFloat11(), ZipNN(threads=1)):
            for n in (0, 1, 2, 3, 4, 1023, 1024, 1025, 8193):
                with self.subTest(codec=type(codec).__name__, n=n):
                    self.assert_roundtrip(codec, torch.randn(n, dtype=torch.bfloat16, device='cuda'))
            self.assert_roundtrip(codec, torch.zeros(257, dtype=torch.bfloat16, device='cuda'))
            self.assert_roundtrip(codec, torch.randn(17, 19, dtype=torch.bfloat16, device='cuda').T)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_splitzip_escapes_and_recalibration(self):
        codec = SplitZip(torch.ones(128, dtype=torch.bfloat16, device='cuda'))
        x = torch.arange(-32768, 32768, dtype=torch.int32, device='cuda').to(torch.int16).view(torch.bfloat16)
        packed = self.assert_roundtrip(codec, x)
        codec.calibrate(torch.zeros(128, dtype=torch.bfloat16, device='cuda'))
        self.assertTrue(torch.equal(codec.decompress(packed).view(torch.int16), x.view(torch.int16)))
        self.assertTrue(torch.equal(SplitZip().decompress(packed).view(torch.int16), x.view(torch.int16)))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_nondefault_stream(self):
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            x = torch.randn(4099, dtype=torch.bfloat16, device='cuda')
            for codec in (SplitZip(x), DFloat11()):
                self.assert_roundtrip(codec, x)
        stream.synchronize()

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_dfloat11_supported_bits_and_multiple_blocks(self):
        bits = torch.arange(-32768, 32768, dtype=torch.int32, device='cuda')
        bits = bits[((bits >> 7) & 255) < 240]
        self.assert_roundtrip(DFloat11(), bits.short().view(torch.bfloat16))
        self.assert_roundtrip(DFloat11(), torch.zeros(65537, dtype=torch.bfloat16, device='cuda'))


if __name__ == '__main__':
    unittest.main()
