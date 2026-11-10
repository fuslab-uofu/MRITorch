import unittest
from pathlib import Path

import numpy as np
import torch

from mritorch.coils import espirit_maps, sense_combine, sos

FIXTURE = Path(__file__).parent / "data" / "espirit_synth_bart.npz"   # BART 0.8.00: ecalib -d0 -r16 -k5 -m1 -c0.8


def _global_phase_aligned(a, b):
    """Rotate `a` by the single phase that best matches `b` (inner product over all elements)."""
    ph = torch.sum(b.conj() * a)
    return a * (ph.conj() / ph.abs())


class TestEspiritVsBart(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d = np.load(FIXTURE)
        cls.ksp = torch.from_numpy(cls.d["ksp"])

    def test_matches_bart_ecalib(self):
        for t in ("0.001", "0.01", "0.05"):
            maps, ev = espirit_maps(self.ksp, calib_size=16, kernel_size=5, threshold=float(t), crop=0.8)
            ref = torch.from_numpy(self.d[f"sens_t{t}"])
            # float32 eigen-decompositions differ from BART's at a few pixels where eigenvalues are nearly degenerate
            dm = (maps[0].abs() - ref.abs()).abs()
            self.assertLess(float(dm.median()), 1e-5, t)
            self.assertLess(float(torch.quantile(dm.flatten(), 0.99)), 1e-3, t)
            dev = (ev[0] - torch.from_numpy(self.d[f"ev_t{t}"])).abs()
            self.assertLess(float(torch.quantile(dev.flatten(), 0.99)), 1e-3, t)
            self.assertTrue(torch.equal(ev[0] >= 0.8, torch.from_numpy(self.d[f"ev_t{t}"]) >= 0.8), t)
            aligned = _global_phase_aligned(maps[0], ref)
            self.assertLess(float((aligned - ref).abs().median()), 5e-3, t)

    def test_unit_norm_inside_support(self):
        maps, ev = espirit_maps(self.ksp, calib_size=16, kernel_size=5)
        norm = (maps[0].abs() ** 2).sum(0)
        inside = ev[0] >= 0.8
        self.assertTrue(torch.allclose(norm[inside], torch.ones_like(norm[inside]), atol=1e-4))
        self.assertTrue(torch.all(norm[~inside] == 0))


class TestSenseCombine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        d = np.load(FIXTURE)
        cls.sens_true = torch.from_numpy(d["sens_true"])
        cls.obj = torch.from_numpy(d["obj"])
        cls.ksp = torch.from_numpy(d["ksp"])

    def test_maps_equal_true_sensitivities_up_to_normalisation_and_phase(self):
        maps, ev = espirit_maps(self.ksp, calib_size=16, kernel_size=5, threshold=1e-3)
        m = maps[0]
        s = self.sens_true / torch.sqrt((self.sens_true.abs() ** 2).sum(0)).clamp_min(1e-9)
        inside = (ev[0] >= 0.8) & (self.obj.abs() > 0.3)
        # per-pixel phase differs by a smooth common phase; compare magnitudes
        self.assertLess(float((m.abs() - s.abs())[:, inside].abs().median()), 0.02)

    def test_noise_free_combine_equals_sos(self):
        maps, ev = espirit_maps(self.ksp, calib_size=16, kernel_size=5, threshold=1e-3)
        img = torch.fft.ifftshift(self.ksp, dim=(-2, -1))
        img = torch.fft.fftshift(torch.fft.ifft2(img, norm="ortho"), dim=(-2, -1))
        comb = sense_combine(img, maps[0], coil_dim=0)
        ref = sos(img, coil_dim=0)
        inside = (ev[0] >= 0.8) & (self.obj.abs() > 0.3)
        ratio = comb.abs()[inside] / ref[inside]
        self.assertLess(float((ratio - 1).abs().median()), 0.02)

    def test_not_biased_by_noise_where_sos_is(self):
        """Pure-noise coil images: the matched filter has mean ~ 0 magnitude scaling of the noise, SOS grows with sqrt(coils)."""
        torch.manual_seed(0)
        n_coil = 8
        maps = torch.randn(n_coil, 64, 64, dtype=torch.cfloat)
        maps = maps / torch.sqrt((maps.abs() ** 2).sum(0, keepdim=True))
        noise = torch.randn(n_coil, 64, 64, dtype=torch.cfloat)
        comb = sense_combine(noise, maps, coil_dim=0)
        self.assertAlmostEqual(float(comb.abs().pow(2).mean()), 1.0, delta=0.08)         # variance of one complex coil
        self.assertAlmostEqual(float(sos(noise, coil_dim=0).pow(2).mean()), float(n_coil), delta=0.3)

    def test_regularisation_scales_magnitude(self):
        maps = torch.zeros(2, 4, 4, dtype=torch.cfloat)
        maps[0] = 1.0
        im = maps.clone()
        out = sense_combine(im, maps, coil_dim=0, reg=5e-4)
        self.assertTrue(torch.allclose(out.abs(), torch.full((4, 4), 1 / 1.0005), atol=1e-6))


if __name__ == "__main__":
    unittest.main()


class TestGauge(unittest.TestCase):
    def test_maps_do_not_depend_on_the_data_phase(self):
        d = np.load(FIXTURE)
        k = torch.from_numpy(d["ksp"])
        m1, _ = espirit_maps(k, calib_size=16, kernel_size=5)
        m2, _ = espirit_maps(k * torch.exp(torch.tensor(0.9j)), calib_size=16, kernel_size=5)
        inside = (m1[0].abs() ** 2).sum(0) > 0
        self.assertLess(float((m1 - m2).abs()[:, :, inside].max()), 1e-3)

    def test_two_maps(self):
        d = np.load(FIXTURE)
        m, ev = espirit_maps(torch.from_numpy(d["ksp"]), calib_size=16, kernel_size=5, n_maps=2, crop=0.0)
        self.assertEqual(tuple(m.shape), (2, 8, 48, 48))
        self.assertTrue(torch.all(ev[0] >= ev[1] - 1e-6))
