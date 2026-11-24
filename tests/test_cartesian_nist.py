"""Cartesian recon options added for the NIST qMRI port (plan Step 4, D22, D26)."""
import unittest

import torch

from mritorch.data import KSpace
from mritorch.fft import fftc, ifftc
from mritorch.geometry import Geometry

try:
    import ismrmrd  # noqa: F401
    HAS_IO = True
except ImportError:
    HAS_IO = False

if HAS_IO:
    from mritorch.io import make_header
    from mritorch.recon import reconstruct_cartesian
    from mritorch.recon.cartesian import imaging_noise_covariance, sort_slices
    from mritorch.coils import ci4_combine
    from mritorch.filters import apply_window

NC = 6


def _geom(y=0.0, ny=16, fov_y=160.0):
    return Geometry(position=(0.0, y, 0.0), read_dir=(1.0, 0.0, 0.0), phase_dir=(0.0, 0.0, 1.0),
                    slice_dir=(0.0, -1.0, 0.0), fov=(160.0, fov_y, 1.6), matrix=(16, ny, 1))


def _ks(data, noise=None, noise_dt=None, dt=4.0, enc_y=16, rec_y=16, slices=(0.0,)):
    header = make_header(encoded_matrix=(32, enc_y, 1), encoded_fov=(320.0, 160.0 * enc_y / rec_y, 1.6),
                         recon_matrix=(16, rec_y, 1), recon_fov=(160.0, 160.0, 1.6))
    geometry = {i: _geom(y, ny=rec_y) for i, y in enumerate(slices)}
    return KSpace(data=data, header=header, noise=noise, geometry=geometry, sample_time_us=dt,
                  noise_sample_time_us=noise_dt)


def _noise_ks(n_echo=3, v_img=2.0, dt=4.0, noise_dt=1.0, seed=0, n_slice=1):
    """Pure-noise k-space (complex variance v_img per sample) and a matching noise scan at another dwell."""
    g = torch.Generator().manual_seed(seed)
    A = torch.eye(NC, dtype=torch.cfloat) + 0.3 * torch.randn(NC, NC, dtype=torch.cfloat, generator=g)
    A = A / torch.linalg.norm(A, dim=1, keepdim=True)                  # rows unit norm: per-coil variance 1
    shape = (n_slice, 1, 1, n_echo, 1, 1, NC, 1, 16, 32)
    w = torch.randn(shape, dtype=torch.cfloat, generator=g)
    data = v_img ** 0.5 * (w.movedim(6, -1) @ A.T).movedim(-1, 6)       # coil mixing: n = A w
    v_noise = v_img * dt / noise_dt                                     # variance grows with bandwidth = 1/dwell
    noise = v_noise ** 0.5 * (A @ torch.randn(NC, 20000, dtype=torch.cfloat, generator=g))
    return data, noise, A


@unittest.skipUnless(HAS_IO, "requires ismrmrd")
class TestCartesianNist(unittest.TestCase):
    def test_ci4_without_noise_is_an_error(self):
        data, _, _ = _noise_ks()
        with self.assertRaises(ValueError):
            reconstruct_cartesian(_ks(data), combine="ci4")

    def test_optimal_without_noise_is_an_error(self):
        data, _, _ = _noise_ks()
        with self.assertRaises(ValueError):
            reconstruct_cartesian(_ks(data), combine="optimal")

    def test_noise_covariance_branches(self):
        data, noise, A = _noise_ks(v_img=2.0, dt=4.0, noise_dt=1.0)
        with self.assertRaises(ValueError):
            imaging_noise_covariance(_ks(data))
        unscaled = imaging_noise_covariance(_ks(data, noise, noise_dt=None))
        self.assertTrue(torch.allclose(unscaled, 8.0 * (A @ A.conj().T), atol=0.25))   # noise-scan variance

    def test_single_echo_ci4_combines_before_crop(self):
        # virtual-coil combine smooths in-plane: must act on the oversampled image, as MATLAB does
        from mritorch.coils import virtual_coil_combine
        from mritorch.utils.array import crop_to_size
        data, noise, _ = _noise_ks(n_echo=1, v_img=1.0)
        ks = _ks(data, noise, noise_dt=1.0)
        out = reconstruct_cartesian(ks, combine="ci4")
        full = ifftc(data, dim=(-3, -2, -1))
        ref = crop_to_size(virtual_coil_combine(full, imaging_noise_covariance(ks), coil_dim=6), (1, 16, 16))
        self.assertTrue(torch.allclose(out.data, ref, atol=1e-5))

    def test_noise_covariance_rescaled_to_imaging_dwell(self):
        data, noise, A = _noise_ks(v_img=2.0, dt=4.0, noise_dt=1.0)
        cov = imaging_noise_covariance(_ks(data, noise, noise_dt=1.0))
        expect = 2.0 * (A @ A.conj().T)
        self.assertTrue(torch.allclose(cov, expect, atol=0.06), (cov.diagonal(), expect.diagonal()))

    def test_ci4_pure_noise_has_unit_prewhitened_power(self):
        # CI4 (self-referenced) = prewhitened root-sum-of-squares: E|S|^2 = n_coils for pure noise,
        # only if the covariance is scaled to the imaging dwell
        data, noise, _ = _noise_ks(n_echo=3, v_img=2.0, dt=4.0, noise_dt=1.0, n_slice=4)
        im = reconstruct_cartesian(_ks(data, noise, noise_dt=1.0, slices=(0, 1, 2, 3)), combine="ci4")
        power = (abs(im.data) ** 2).mean().item()
        self.assertAlmostEqual(power / NC, 1.0, delta=0.05)

    def test_keep_phase_oversampling(self):
        data, noise, _ = _noise_ks(n_echo=1)
        data = torch.nn.functional.pad(data, (0, 0, 2, 2))             # 20 encoded ky lines, recon matrix 16
        ks = _ks(data, noise, noise_dt=1.0, enc_y=20, rec_y=16)
        im = reconstruct_cartesian(ks, combine="sos", crop_phase_oversampling=False)
        self.assertEqual(im.data.shape[-2], 20)
        g = im.slice_geometry[0]
        self.assertEqual(g.matrix, (16, 20, 1))
        self.assertAlmostEqual(g.fov[1], 200.0)                         # 160 mm * 20 / 16; voxel size unchanged
        self.assertAlmostEqual(g.resolution[1], 10.0)
        cropped = reconstruct_cartesian(ks, combine="sos")
        self.assertEqual(cropped.data.shape[-2], 16)

    def test_keep_phase_oversampling_with_zfi(self):
        data, noise, _ = _noise_ks(n_echo=1)
        data = torch.nn.functional.pad(data, (0, 0, 2, 2))
        ks = _ks(data, noise, noise_dt=1.0, enc_y=20, rec_y=16)
        im = reconstruct_cartesian(ks, combine="none", crop_phase_oversampling=False, matrix=(12, 15, 1))
        g = im.slice_geometry[0]
        self.assertEqual(tuple(im.data.shape[-3:]), (1, 15, 12))
        self.assertEqual(g.matrix, (12, 15, 1))
        self.assertAlmostEqual(g.fov[1], 200.0)                         # FOV kept through the interpolation

    def test_window_before_combine(self):
        data, noise, _ = _noise_ks(n_echo=3, v_img=1.0)
        ks = _ks(data, noise, noise_dt=1.0)
        out = reconstruct_cartesian(ks, combine="ci4", window="tukey", window_params=[None, 1.0, 1.0],
                                    window_before_combine=True)
        # manual: image, crop readout, window each coil, then CI4 with the dwell-scaled covariance
        im = ifftc(data, dim=(-3, -2, -1))[..., 8:24]
        im = ifftc(apply_window(fftc(im), "tukey", [None, 1.0, 1.0]))
        ref = ci4_combine(im, imaging_noise_covariance(ks), coil_dim=6, echo_dim=3, time_dim=4)
        self.assertTrue(torch.allclose(out.data, ref, atol=1e-5))
        after = reconstruct_cartesian(ks, combine="ci4", window="tukey", window_params=[None, 1.0, 1.0])
        self.assertFalse(torch.allclose(after.data, out.data, atol=1e-3))

    def test_sort_slices_by_position_along_slice_dir(self):
        data, noise, _ = _noise_ks(n_echo=1, n_slice=3)
        data = data * torch.arange(1, 4).view(3, *([1] * 9))           # tag slices
        ks = _ks(data, noise, noise_dt=1.0, slices=(-7.2, 47.2, 15.2))  # PCS y; slice_dir = -y
        im = sort_slices(reconstruct_cartesian(ks, combine="sos"))
        ys = [im.slice_geometry[i].position[1] for i in range(3)]
        self.assertEqual(ys, [47.2, 15.2, -7.2])                        # ascending along slice_dir (-y)
        tags = [abs(im.data[i]).mean().item() for i in range(3)]
        self.assertTrue(tags[1] > tags[0] > tags[2])                    # tags 2, 3, 1 (original slices 1, 2, 0)
        self.assertEqual(im.meta["slice_order"], [1, 2, 0])


if __name__ == "__main__":
    unittest.main()
