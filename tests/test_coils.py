import unittest

import numpy as np
import torch

from mritorch.coils import noise_covariance, prewhiten, sos, optimal_combine, ci4_combine, virtual_coil_combine
from mritorch.recon.corrections import drift_correction


def _correlated_noise(n_coil=4, n=200000, seed=0):
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(n_coil, n_coil, dtype=torch.cdouble, generator=g) * 0.3 + torch.eye(n_coil)
    white = torch.randn(n_coil, n, dtype=torch.cdouble, generator=g)
    return A @ white, A @ A.conj().T * 1.0  # complex normal has E|z|^2 = 1


def _coil_images(n_coil=4, shape=(1, 32, 32), seed=0):
    g = torch.Generator().manual_seed(seed)
    z, y, x = torch.meshgrid(*[torch.linspace(-1, 1, n) for n in shape], indexing="ij")
    obj = ((x ** 2 + y ** 2) < 0.6).to(torch.cfloat) * torch.exp(1j * 0.5 * x)
    sens = []
    for c in range(n_coil):
        ang = 2 * np.pi * c / n_coil
        cx, cy = np.cos(ang), np.sin(ang)
        mag = torch.exp(-((x - cx) ** 2 + (y - cy) ** 2))
        sens.append(mag * torch.exp(1j * (c + 0.3 * y)))
    sens = torch.stack(sens)  # (coil, z, y, x)
    return sens * obj, obj, sens


class TestNoise(unittest.TestCase):
    def test_covariance(self):
        noise, cov = _correlated_noise()
        est = noise_covariance(noise)
        self.assertTrue(torch.allclose(est, cov, atol=0.05))
        # matches numpy (rows are variables)
        self.assertTrue(np.allclose(est.numpy(), np.cov(noise.numpy())))

    def test_prewhiten(self):
        noise, _ = _correlated_noise()
        cov = noise_covariance(noise)
        white = prewhiten(noise, cov, coil_dim=0)
        self.assertTrue(torch.allclose(noise_covariance(white), torch.eye(4, dtype=cov.dtype), atol=1e-6))


class TestCombine(unittest.TestCase):
    def test_sos(self):
        im = torch.randn(3, 2, 4, 5, dtype=torch.cfloat)
        self.assertTrue(torch.allclose(sos(im), torch.sqrt((abs(im) ** 2).sum(-4))))

    def test_optimal_identity_cov_is_sos(self):
        coil_im, _, _ = _coil_images()
        out = optimal_combine(coil_im, torch.eye(4))
        self.assertTrue(torch.allclose(abs(out), sos(coil_im), atol=1e-5))

    def test_ci4_multi_echo_phase(self):
        coil_im, obj, _ = _coil_images()
        dphi = 0.7
        echoes = torch.stack([coil_im, coil_im * np.exp(1j * dphi)])  # (echo, coil, z, y, x)
        cov = torch.eye(4) + 0.1
        out = ci4_combine(echoes, cov, coil_dim=1, echo_dim=0)
        mask = abs(obj) > 0
        # echo 1 is phase-referenced to itself (zero phase); echo 2 carries dphi
        self.assertTrue(torch.allclose(torch.angle(out[0])[mask], torch.zeros(()), atol=1e-4))
        self.assertTrue(torch.allclose(torch.angle(out[1])[mask], torch.tensor(dphi), atol=1e-4))
        self.assertTrue(torch.allclose(abs(out[0]), abs(out[1]), atol=1e-5))

    def test_ci4_dynamic_phase(self):
        coil_im, obj, _ = _coil_images()
        phases = torch.tensor([0.0, 0.2, -0.4])
        dyn = coil_im[None] * torch.exp(1j * phases)[:, None, None, None, None]
        out = ci4_combine(dyn, torch.eye(4), coil_dim=1, time_dim=0)
        mask = abs(obj[0]) > 0
        for t, p in enumerate(phases):
            self.assertTrue(torch.allclose(torch.angle(out[t, 0])[mask], p, atol=1e-4))

    def test_virtual_coil_magnitude(self):
        coil_im, obj, _ = _coil_images(shape=(1, 48, 48))
        out = virtual_coil_combine(coil_im, torch.eye(4, dtype=torch.cfloat))
        mask = abs(obj) > 0
        self.assertGreater(float(np.corrcoef(abs(out[mask]), sos(coil_im)[mask])[0, 1]), 0.99)

    def test_virtual_coil_removes_smooth_phase(self):
        # vcCombinePH references phase to smoothed images, so a global phase
        # (or any slowly varying phase) does not reach the output
        coil_im, _, _ = _coil_images(shape=(1, 48, 48))
        cov = torch.eye(4, dtype=torch.cfloat)
        a = virtual_coil_combine(coil_im, cov)
        b = virtual_coil_combine(coil_im * np.exp(1j * 1.2), cov)
        self.assertTrue(torch.allclose(a, b, atol=1e-5))

    def test_virtual_coil_single_frame_dispatch(self):
        coil_im, _, _ = _coil_images()
        a = ci4_combine(coil_im, torch.eye(4, dtype=torch.cfloat))
        b = virtual_coil_combine(coil_im, torch.eye(4, dtype=torch.cfloat))
        self.assertTrue(torch.allclose(a, b))


class TestDrift(unittest.TestCase):
    def test_removes_global_phase(self):
        ks = torch.randn(1, 4, 8, 16, dtype=torch.cfloat).expand(10, 4, 8, 16).clone()  # (rep, coil, ky, kx)
        drift = torch.linspace(0, 1, 10)
        drifted = ks * torch.exp(1j * drift)[:, None, None, None]
        out = drift_correction(drifted, time_dim=0, average_dims=(1, 2, 3), n_baseline=1)
        self.assertTrue(torch.allclose(out, ks, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
