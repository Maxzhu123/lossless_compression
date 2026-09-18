"""Check native payload compatibility with the vendored reference encoder."""
import unittest

import torch

from baselines import DFloat11


class NativeEncoderTests(unittest.TestCase):
    def assert_compatible(self, x):
        native, reference = DFloat11(), DFloat11(encoder='reference')
        a, b = native.compress(x), reference.compress(x)
        self.assertEqual(a.compressed_bytes, b.compressed_bytes)
        self.assertEqual(a.memory_size(), b.memory_size())
        if a.payload is not None:
            self.assertEqual(a.payload.n_bytes, b.payload.n_bytes)
            self.assertEqual(a.payload.shared_bytes, b.payload.shared_bytes)
            for field in ('luts', 'encoded', 'sign_mantissa', 'positions', 'gaps'):
                self.assertTrue(torch.equal(getattr(a.payload, field), getattr(b.payload, field)), field)
        self.assertTrue(torch.equal(native.decompress(a).view(torch.int16), x.view(torch.int16)))
        self.assertTrue(torch.equal(reference.decompress(a).view(torch.int16), x.view(torch.int16)))

    def test_boundaries(self):
        for n in (0, 1, 7, 8, 63, 64, 65, 32767, 32768, 32769, 65537):
            with self.subTest(n=n):
                self.assert_compatible(torch.ones(n, dtype=torch.bfloat16))

    def test_random_and_exponent_range(self):
        generator = torch.Generator().manual_seed(7)
        self.assert_compatible(torch.randn(100003, generator=generator).to(torch.bfloat16))
        raw = torch.arange(-32768, 32768, dtype=torch.int32).to(torch.int16)
        raw = raw[((raw >> 7) & 255) < 240]
        self.assert_compatible(raw.view(torch.bfloat16))
        self.assert_compatible(torch.randn(19, 17, generator=generator).to(torch.bfloat16).T)

    def test_invalid_encoder(self):
        with self.assertRaises(ValueError):
            DFloat11(encoder='unknown')

    def test_long_codes(self):
        from baselines._dfloat11_encoder import encode, histogram
        from baselines._vendor.dfloat11.dfloat11_utils import get_32bit_codec, encode_weights
        codec, _, table = get_32bit_codec({i: 1 << i for i in range(31)})
        self.assertGreater(max(length for length, _ in table.values()), 24)
        raw = (torch.arange(31, dtype=torch.int16).repeat(101) << 7)
        x = raw.view(torch.bfloat16)
        encoded, sm, positions, gaps, n_bytes = encode(x, codec, histogram(x))
        expected, expected_sm, expected_positions, expected_gaps, _ = encode_weights([x], codec, 8, 512)
        self.assertEqual(n_bytes, expected.numel())
        for actual, reference in ((encoded[:-12], expected), (sm, expected_sm),
                                  (positions, expected_positions), (gaps[:-1], expected_gaps)):
            self.assertTrue(torch.equal(actual, reference))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_cuda(self):
        for n in (1, 1025, 32769, 1048576):
            with self.subTest(n=n):
                self.assert_compatible(torch.randn(n, device='cuda').to(torch.bfloat16))


if __name__ == '__main__':
    unittest.main()
