import inspect
import math
import unittest
from unittest import mock

import torch

from mritorch.io.mpme import MpmeKSpace, _pathways_from_header
import mritorch.recon.radial as radial
from mritorch.recon.radial import Nufft2d, calibration_maps, radial_trajectory, reconstruct_mpme

N, N_VIEW, N_COIL = 16, 40, 4


def _synthetic():
    """Two readout planes, pathways (1, 0, -1), two echoes: each image is a scaled disk; coils have smooth maps."""
    y, x = torch.meshgrid(torch.linspace(-1, 1, N), torch.linspace(-1, 1, N), indexing="ij")
    disk = ((x ** 2 + y ** 2) < 0.6).to(torch.cfloat)
    sens = torch.stack([torch.exp(-((x - math.cos(a)) ** 2 + (y - math.sin(a)) ** 2) / 1.5) * torch.exp(torch.tensor(1j * 0.4 * c))
                        for c, a in enumerate([2 * math.pi * k / N_COIL for k in range(N_COIL)])]).to(torch.cfloat)
    nu = Nufft2d(radial_trajectory(N, N_VIEW, 111.246), (N, N), backend="direct")
    amp = {(0, 0): 1.0, (1, 0): 0.5, (2, 0): 0.2, (0, 1): 0.8, (1, 1): 0.4, (2, 1): 0.1}     # (pathway, echo)
    data = torch.zeros(2, N, N_VIEW, N_COIL, 3, 2, dtype=torch.cfloat)
    truth = torch.zeros(2, N, N, 3, 2, dtype=torch.cfloat)
    for xi in range(2):
        for (p, e), a in amp.items():
            img = a * (1 + 0.5 * xi) * disk
            truth[xi, :, :, p, e] = img
            k = nu.forward(sens * img)                                    # (coil, samp*view)
            data[xi, :, :, :, p, e] = k.reshape(N_COIL, N, N_VIEW).permute(1, 2, 0)
    return MpmeKSpace(data, (1, 0, -1), 111.246), truth


class TestMpme(unittest.TestCase):
    def test_calibration_pathway_index(self):
        self.assertEqual(MpmeKSpace(torch.zeros(1, 1, 1, 1, 3, 1), (1, 0, -1)).calib_index, 1)
        self.assertEqual(MpmeKSpace(torch.zeros(1, 1, 1, 1, 1, 1), (0,)).calib_index, 0)

    def test_pathways_from_protocol_name(self):
        class H:
            class measurementInformation:
                protocolName = "MPME_XA30_FOV256_Mat160_4pwy_FA30"
        self.assertEqual(_pathways_from_header(H), (1, 0, -1, -2))
        H.measurementInformation.protocolName = "x_3pwy"
        self.assertEqual(_pathways_from_header(H), (1, 0, -1))
        H.measurementInformation.protocolName = "other"
        with self.assertRaises(ValueError):
            _pathways_from_header(H)

    def test_reconstruct_mpme_recovers_pathway_and_echo_images(self):
        mk, truth = _synthetic()
        rec = reconstruct_mpme(mk, shape=(N, N), reg=1e-4, n_iter=40, backend="direct",
                               espirit=dict(calib_size=8, kernel_size=3), calib_reduction=1)
        self.assertEqual(tuple(rec.shape), (2, N, N, 3, 2))
        for xi in range(2):
            m = truth[xi, :, :, 0, 0].abs() > 0
            for p in range(3):
                for e in range(2):
                    # compare magnitudes (each plane has an arbitrary global phase)
                    r = rec[xi, :, :, p, e].abs()[m] / truth[xi, :, :, p, e].abs()[m]
                    self.assertLess(float((r - 1).abs().median()), 0.1, (xi, p, e))

    def test_reconstruct_mpme_default_backend(self):
        """Same problem through the default (torchkbnufft) backend, with no backend argument."""
        for fn in (reconstruct_mpme, calibration_maps):
            self.assertEqual(inspect.signature(fn).parameters["backend"].default, "kb", fn.__name__)
        mk, truth = _synthetic()
        rec = reconstruct_mpme(mk, shape=(N, N), reg=1e-4, n_iter=40,
                               espirit=dict(calib_size=8, kernel_size=3), calib_reduction=1)
        self.assertEqual(tuple(rec.shape), (2, N, N, 3, 2))
        for xi in range(2):
            m = truth[xi, :, :, 0, 0].abs() > 0
            for p in range(3):
                for e in range(2):
                    r = rec[xi, :, :, p, e].abs()[m] / truth[xi, :, :, p, e].abs()[m]
                    self.assertLess(float((r - 1).abs().median()), 0.1, (xi, p, e))

    def test_operators_built_once_with_nufft_settings(self):
        """One SENSE and one calibration NUFFT for all planes, both with the requested numpoints/oversamp."""
        mk, _ = _synthetic()
        with mock.patch.object(radial, "Nufft2d", wraps=radial.Nufft2d) as spy:
            reconstruct_mpme(mk, shape=(N, N), n_iter=2, espirit=dict(calib_size=8, kernel_size=3),
                             calib_reduction=1, numpoints=4, oversamp=1.5)
        self.assertEqual(spy.call_count, 2)
        for call in spy.call_args_list:
            self.assertEqual((call.kwargs["numpoints"], call.kwargs["oversamp"]), (4, 1.5))


if __name__ == "__main__":
    unittest.main()
