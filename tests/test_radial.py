import math
import unittest

import pytest
import torch

from mritorch.fft import fftc
from mritorch.recon.radial import Nufft2d, radial_trajectory, sense_cg, calibration_maps, calibration_nufft

KB_TOL = 1e-4   # max relative error of backend="kb" against "direct" (measured ~1e-5, plan step B1)


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

    def test_real_input_gives_complex_output(self):
        traj = radial_trajectory(8, 5, 111.246)
        x = torch.randn(2, 8, 8, dtype=torch.float32)
        for backend in ("direct", "kb"):
            nu = Nufft2d(traj, (8, 8), backend=backend)
            y = nu.forward(x)
            self.assertEqual(y.dtype, torch.complex64, backend)
            self.assertLess(_relerr(y, nu.forward(x.to(torch.complex64))), 1e-6, backend)
            self.assertEqual(nu.adjoint(y.real.double()).dtype, torch.complex128, backend)

    def test_kb_is_default_backend(self):
        self.assertEqual(Nufft2d(radial_trajectory(8, 3, 111.246), (8, 8)).backend, "kb")


def _relerr(a, b):
    return float((a - b).abs().max() / b.abs().max())


class TestKbNufft(unittest.TestCase):
    """backend="kb" (torchkbnufft) against the exact dense sums of backend="direct"."""

    CASES = [((16, 16), (16, 9)), ((32, 32), (32, 20)), ((40, 40), (40, 80)), ((24, 20), (24, 30))]

    def _check_matches_direct(self, dtype):
        torch.manual_seed(0)
        for shape, (n_samp, n_view) in self.CASES:
            traj = radial_trajectory(n_samp, n_view, 111.246)
            ref = Nufft2d(traj, shape, backend="direct")
            kb = Nufft2d(traj, shape, backend="kb")
            x = torch.randn(2, 3, *shape, dtype=torch.complex128)
            y = torch.randn(2, 3, ref.n_points, dtype=torch.complex128)
            fk, ak = kb.forward(x.to(dtype)), kb.adjoint(y.to(dtype))
            self.assertEqual(fk.dtype, dtype)
            self.assertEqual(ak.dtype, dtype)
            self.assertEqual(tuple(fk.shape), (2, 3, ref.n_points))
            self.assertEqual(tuple(ak.shape), (2, 3, *shape))
            self.assertLess(_relerr(fk.to(torch.complex128), ref.forward(x)), KB_TOL, (shape, dtype))
            self.assertLess(_relerr(ak.to(torch.complex128), ref.adjoint(y)), KB_TOL, (shape, dtype))

    def test_matches_direct_complex64(self):
        self._check_matches_direct(torch.complex64)

    def test_matches_direct_complex128(self):
        self._check_matches_direct(torch.complex128)

    def test_adjointness(self):
        traj = radial_trajectory(16, 9, 111.246)
        nu = Nufft2d(traj, (16, 16), backend="kb")
        x = torch.randn(2, 16, 16, dtype=torch.complex128)
        y = torch.randn(2, nu.n_points, dtype=torch.complex128)
        lhs = torch.sum(nu.forward(x).conj() * y)
        rhs = torch.sum(x.conj() * nu.adjoint(y))
        self.assertLess(abs(complex(lhs - rhs)), 1e-10 * abs(complex(lhs)))

    def test_gradcheck(self):
        traj = radial_trajectory(8, 5, 111.246)
        nu = Nufft2d(traj, (8, 8), backend="kb")
        x = torch.randn(2, 8, 8, dtype=torch.complex128, requires_grad=True)
        y = torch.randn(2, nu.n_points, dtype=torch.complex128, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(nu.forward, (x,)))
        self.assertTrue(torch.autograd.gradcheck(nu.adjoint, (y,)))

    def _check_device(self, device):
        traj = radial_trajectory(16, 9, 111.246)
        nu = Nufft2d(traj, (16, 16), backend="kb")
        x = torch.randn(2, 16, 16, dtype=torch.complex64, device=device)
        y = nu.forward(x)
        self.assertEqual(y.device, x.device)
        self.assertEqual(nu.adjoint(y).device, x.device)
        ref = Nufft2d(traj, (16, 16), backend="direct").forward(x.cpu().to(torch.complex128))
        self.assertLess(_relerr(y.cpu().to(torch.complex128), ref), KB_TOL)

    def test_cpu_device(self):
        self._check_device(torch.device("cpu"))

    @pytest.mark.gpu
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_cuda_device(self):
        self._check_device(torch.device("cuda"))


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

    def test_prebuilt_calibration_operator(self):
        obj, sens = _phantom(n=40, n_coil=6)
        n = 40
        nu = Nufft2d(radial_trajectory(n, 80, 111.246), (n, n), backend="direct")
        y = nu.forward(sens * obj).reshape(6, n, 80).permute(1, 2, 0)
        kw = dict(calib_reduction=2, espirit=dict(calib_size=12, kernel_size=4))
        ref = calibration_maps(y, 111.246, (n, n), **kw)
        op = calibration_nufft(n, 80, 111.246, (n, n), calib_reduction=2)
        self.assertTrue(torch.allclose(calibration_maps(y, 111.246, (n, n), nufft=op, **kw), ref, atol=1e-5))
        wrong = calibration_nufft(n, 80, 111.246, (n, n), calib_reduction=4)
        with self.assertRaises(ValueError):
            calibration_maps(y, 111.246, (n, n), nufft=wrong, **kw)
        other_angle = calibration_nufft(n, 80, 90.0, (n, n), calib_reduction=2)
        with self.assertRaises(ValueError):
            calibration_maps(y, 111.246, (n, n), nufft=other_angle, **kw)


if __name__ == "__main__":
    unittest.main()
