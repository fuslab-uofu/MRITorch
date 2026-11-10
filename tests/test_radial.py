import math
import unittest

import torch

from mritorch.fft import fftc
from mritorch.recon.radial import Nufft2d, radial_trajectory, sense_cg, calibration_maps

try:
    import finufft  # noqa: F401
    HAS_FINUFFT = True
except ImportError:
    HAS_FINUFFT = False


def _phantom(n=24, n_coil=6):
    y, x = torch.meshgrid(torch.linspace(-1, 1, n), torch.linspace(-1, 1, n), indexing="ij")
    obj = (((x / 0.8) ** 2 + (y / 0.7) ** 2) < 1).to(torch.cfloat) * (1 + 0.3 * torch.sin(3 * x))
    sens = []
    for c in range(n_coil):
        a = 2 * math.pi * c / n_coil
        mag = torch.exp(-((x - 1.2 * math.cos(a)) ** 2 + (y - 1.2 * math.sin(a)) ** 2) / 1.5)
        sens.append(mag * torch.exp(1j * (0.5 * c + 0.4 * x * math.cos(a))))
    sens = torch.stack(sens).to(torch.cfloat)
    sens = sens / torch.sqrt((sens.abs() ** 2).sum(0, keepdim=True))
    return obj, sens


class TestTrajectory(unittest.TestCase):
    def test_values(self):
        t = radial_trajectory(8, 3, 90.0)
        self.assertEqual(tuple(t.shape), (2, 8, 3))
        self.assertTrue(torch.allclose(t[:, 4, :], torch.zeros(2, 3, dtype=torch.float64)))        # k = 0 at index N // 2
        self.assertAlmostEqual(float(t[0, 7, 0]), 3.0)                                              # first view along x
        self.assertAlmostEqual(float(t[1, 7, 1]), 3.0, places=9)                                    # 90 deg: along y
        self.assertAlmostEqual(float(t[0, 0, 0]), -4.0)


class TestNufft(unittest.TestCase):
    def _cartesian(self, n=8):
        k1, k2 = torch.meshgrid(torch.arange(n) - n // 2, torch.arange(n) - n // 2, indexing="ij")
        return torch.stack([k1.reshape(-1), k2.reshape(-1)]).to(torch.float64)

    def test_direct_equals_fft_on_cartesian_grid(self):
        n = 8
        nu = Nufft2d(self._cartesian(n), (n, n), backend="direct")
        x = torch.randn(3, n, n, dtype=torch.cfloat)
        ref = fftc(x, dim=(-2, -1), norm="ortho").reshape(3, -1)
        self.assertTrue(torch.allclose(nu.forward(x), ref, atol=1e-5))

    def test_adjointness(self):
        traj = radial_trajectory(16, 9, 111.246)
        nu = Nufft2d(traj, (16, 16), backend="direct")
        x = torch.randn(2, 16, 16, dtype=torch.cfloat)
        y = torch.randn(2, nu.n_points, dtype=torch.cfloat)
        lhs = torch.sum(nu.forward(x).conj() * y)
        rhs = torch.sum(x.conj() * nu.adjoint(y))
        self.assertAlmostEqual(float(lhs.real), float(rhs.real), delta=1e-3 * abs(float(lhs.real)) + 1e-4)

    @unittest.skipUnless(HAS_FINUFFT, "requires finufft")
    def test_finufft_matches_direct(self):
        traj = radial_trajectory(32, 20, 111.246)
        a = Nufft2d(traj, (32, 32), backend="direct")
        b = Nufft2d(traj, (32, 32), backend="finufft", eps=1e-9)
        x = torch.randn(2, 32, 32, dtype=torch.cfloat)
        y = torch.randn(2, a.n_points, dtype=torch.cfloat)
        self.assertLess(float((a.forward(x) - b.forward(x)).abs().max() / a.forward(x).abs().max()), 1e-5)
        self.assertLess(float((a.adjoint(y) - b.adjoint(y)).abs().max() / a.adjoint(y).abs().max()), 1e-5)


class TestSenseCg(unittest.TestCase):
    def test_recovers_object_from_radial_data(self):
        obj, sens = _phantom()
        n = obj.shape[0]
        nu = Nufft2d(radial_trajectory(n, 60, 111.246), (n, n), backend="direct")
        y = nu.forward(sens * obj)                                           # (coil, pts)
        rec = sense_cg(y, nu, sens, reg=1e-4, n_iter=60)
        mask = obj.abs() > 0
        err = (rec - obj).abs()[mask].mean() / obj.abs()[mask].mean()
        self.assertLess(float(err), 0.05)

    def test_batches_are_independent(self):
        obj, sens = _phantom()
        n = obj.shape[0]
        nu = Nufft2d(radial_trajectory(n, 30, 111.246), (n, n), backend="direct")
        y1 = nu.forward(sens * obj)
        y2 = nu.forward(sens * obj.flip(0))
        both = sense_cg(torch.stack([y1, y2]), nu, sens, reg=1e-3, n_iter=15)
        one = sense_cg(y2[None], nu, sens, reg=1e-3, n_iter=15)
        self.assertTrue(torch.allclose(both[1], one[0], atol=1e-4))

    def test_regularisation_matches_cartesian_matched_filter(self):
        """Fully sampled Cartesian trajectory: x = (sum conj(c) y) / (1 + reg), the BART pics behaviour."""
        obj, sens = _phantom(n=12)
        n = 12
        k1, k2 = torch.meshgrid(torch.arange(n) - n // 2, torch.arange(n) - n // 2, indexing="ij")
        nu = Nufft2d(torch.stack([k1.reshape(-1), k2.reshape(-1)]).to(torch.float64), (n, n), backend="direct")
        y = nu.forward(sens * obj)
        reg = 5e-4
        rec = sense_cg(y, nu, sens, reg=reg, n_iter=30)
        self.assertTrue(torch.allclose(rec, obj / (1 + reg), atol=1e-4))


class TestCalibration(unittest.TestCase):
    def test_maps_from_radial_calibration_data(self):
        obj, sens = _phantom(n=40, n_coil=6)
        n = 40
        nu = Nufft2d(radial_trajectory(n, 80, 111.246), (n, n), backend="direct")
        y = nu.forward(sens * obj).reshape(6, n, 80).permute(1, 2, 0)        # (samp, view, coil)
        maps = calibration_maps(y, 111.246, (n, n), calib_reduction=2, espirit=dict(calib_size=12, kernel_size=4),
                                backend="direct")
        self.assertEqual(tuple(maps.shape), (6, n, n))
        inside = (maps.abs() ** 2).sum(0) > 0.5
        self.assertGreater(int(inside.sum()), 100)
        # the maps reproduce the coil images up to a smooth phase: |sum_c conj(map_c) * coil_c| ~ |obj|
        comb = (maps.conj() * (sens * obj)).sum(0).abs()
        m = inside & (obj.abs() > 0.3)
        self.assertLess(float(((comb - obj.abs()).abs() / obj.abs())[m].median()), 0.15)


if __name__ == "__main__":
    unittest.main()
