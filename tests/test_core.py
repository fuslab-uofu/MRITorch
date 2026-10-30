import unittest

import numpy as np
import torch

from mritorch.fft import fftc, ifftc
from mritorch.utils.array import resize_center, pad_to_size, crop_to_size, zfi_to_size
from mritorch import filters

_atol = 1e-5


class TestFFT(unittest.TestCase):
    def test_roundtrip(self):
        x = torch.randn(2, 3, 8, 9, dtype=torch.cfloat)
        self.assertTrue(torch.allclose(ifftc(fftc(x)), x, atol=_atol))

    def test_parseval(self):
        x = torch.randn(4, 6, 10, dtype=torch.cfloat)
        self.assertAlmostEqual(float(torch.sum(abs(x) ** 2)), float(torch.sum(abs(fftc(x)) ** 2)), places=3)

    def test_dc_at_center(self):
        for n in (8, 9):
            x = torch.ones(n, dtype=torch.cfloat)
            k = fftc(x, dim=-1)
            self.assertEqual(int(torch.argmax(abs(k))), n // 2)

    def test_adjoint(self):
        x = torch.randn(5, 6, 7, dtype=torch.cfloat)
        y = torch.randn(5, 6, 7, dtype=torch.cfloat)
        lhs = torch.vdot(fftc(x).flatten(), y.flatten())
        rhs = torch.vdot(x.flatten(), ifftc(y).flatten())
        self.assertTrue(torch.allclose(lhs, rhs, atol=1e-4))


class TestArray(unittest.TestCase):
    def test_center_alignment(self):
        for n, m in [(8, 12), (8, 11), (9, 12), (9, 4), (8, 5)]:
            x = torch.zeros(n)
            x[n // 2] = 1
            y = resize_center(x, (m,))
            self.assertEqual(y.shape, (m,))
            self.assertEqual(int(torch.argmax(y)), m // 2, f"n={n}, m={m}")

    def test_pad_crop_inverse(self):
        x = torch.randn(3, 7, 8)
        y = crop_to_size(pad_to_size(x, (11, 13)), (7, 8))
        self.assertTrue(torch.equal(x, y))

    def test_pad_and_crop_never_shrink_or_grow(self):
        x = torch.randn(6, 6)
        self.assertEqual(pad_to_size(x, (4, 8)).shape, (6, 8))
        self.assertEqual(crop_to_size(x, (4, 8)).shape, (4, 6))

    def test_none_keeps_size(self):
        x = torch.randn(4, 5, 6)
        self.assertEqual(resize_center(x, (None, 8)).shape, (4, 5, 8))

    def test_zfi_roundtrip(self):
        x = torch.randn(2, 8, 10, dtype=torch.cfloat)
        up = zfi_to_size(x, (16, 20))
        self.assertEqual(up.shape, (2, 16, 20))
        self.assertTrue(torch.allclose(zfi_to_size(up, (8, 10)), x, atol=_atol))

    def test_zfi_preserves_constant(self):
        x = torch.full((8, 8), 2.0 + 1j, dtype=torch.cfloat)
        self.assertTrue(torch.allclose(zfi_to_size(x, (16, 12)), torch.full((16, 12), 2.0 + 1j), atol=_atol))
        legacy = zfi_to_size(x, (16, 16), preserve_intensity=False)
        self.assertTrue(torch.allclose(legacy, torch.full((16, 16), (2.0 + 1j) / 4), atol=_atol))


class TestFilters(unittest.TestCase):
    def test_tukey_matches_scipy(self):
        from scipy.signal.windows import tukey
        for n in (16, 17, 64):
            for r in (0.0, 0.25, 0.5, 1.0):
                w = filters.tukey(n, r, dtype=torch.float64).numpy()
                self.assertTrue(np.allclose(w, tukey(n, r)), f"n={n}, r={r}")

    def test_hann_matches_scipy(self):
        from scipy.signal.windows import hann
        self.assertTrue(np.allclose(filters.hann(15, dtype=torch.float64).numpy(), hann(15)))

    def test_fermi(self):
        for n in (32, 33):
            w = filters.fermi(n, 0.2)
            self.assertEqual(w.shape, (n,))
            self.assertTrue(torch.allclose(w, w.flip(0)))
            self.assertGreater(float(w[n // 2]), 0.99)
            self.assertLess(float(w[0]), 0.05)

    def test_separable_window(self):
        w = filters.separable_window("tukey", (1, 8, 10), [0.5, 0.5, None])
        self.assertEqual(w.shape, (1, 8, 10))
        self.assertTrue(torch.allclose(w[0, :, 0], filters.tukey(8, 0.5)))
        self.assertTrue(torch.allclose(w[0, 3, :], w[0, 3, 0].expand(10)))

    def test_apply_window(self):
        ks = torch.ones(2, 4, 6, dtype=torch.cfloat)
        out = filters.apply_window(ks, "hann", [0.0, 0.0])
        self.assertEqual(out.dtype, torch.cfloat)
        self.assertTrue(torch.allclose(out[0].real, filters.separable_window("hann", (4, 6))))


if __name__ == "__main__":
    unittest.main()
