import os
import tempfile
import unittest

import numpy as np
import torch

from mritorch.data import KSpace
from mritorch.fft import fftc
from mritorch.geometry import Geometry
from mritorch.coils import sos

try:
    import ismrmrd  # noqa: F401
    HAS_IO = True
except ImportError:
    HAS_IO = False

if HAS_IO:
    from mritorch.io import make_header, write_kspace, read_kspace
    from mritorch.recon import reconstruct_cartesian

NY, NX, OS = 32, 32, 2
N_COIL = 4


def _synthetic(n_echo=1, noise_std=1e-3, seed=0):
    """Two-echo single-slice phantom with readout oversampling and correlated noise."""
    g = torch.Generator().manual_seed(seed)
    y, x = torch.meshgrid(torch.linspace(-1, 1, NY), torch.linspace(-OS, OS, NX * OS), indexing="ij")
    obj = ((x ** 2 + y ** 2) < 0.5).to(torch.cfloat) * torch.exp(1j * 0.3 * y)
    sens = torch.stack([torch.exp(-((x - np.cos(a)) ** 2 + (y - np.sin(a)) ** 2)) * torch.exp(1j * (c + 0.2 * x))
                        for c, a in enumerate(np.linspace(0, 2 * np.pi, N_COIL, endpoint=False))])
    A = torch.eye(N_COIL, dtype=torch.cfloat) + 0.2
    echoes = []
    for e in range(n_echo):
        coil = (sens * obj * np.exp(1j * 0.5 * e))[:, None]  # (coil, z, y, x)
        ks = fftc(coil)
        ks = ks + noise_std * torch.einsum("ij,j...->i...", A, torch.randn(ks.shape, dtype=torch.cfloat, generator=g))
        echoes.append(ks)
    data = torch.stack(echoes)[None, None, None, :, None, None]  # (slice, set, phase, contrast, rep, avg, ...)
    noise = noise_std * (A @ torch.randn(N_COIL, 5000, dtype=torch.cfloat, generator=g))
    header = make_header(encoded_matrix=(NX * OS, NY, 1), encoded_fov=(200.0 * OS, 200.0, 5.0),
                         recon_matrix=(NX, NY, 1), recon_fov=(200.0, 200.0, 5.0))
    geom = Geometry(position=(0.0, 0.0, 0.0), read_dir=(1.0, 0.0, 0.0), phase_dir=(0.0, 1.0, 0.0),
                    slice_dir=(0.0, 0.0, 1.0), fov=(200.0, 200.0, 5.0), matrix=(NX, NY, 1))
    ks = KSpace(data=data, header=header, noise=noise, geometry={0: geom})
    truth = obj[:, NX * OS // 2 - NX // 2: NX * OS // 2 + NX // 2]
    truth_sos = sos(sens[:, None] * obj)[0, :, NX * OS // 2 - NX // 2: NX * OS // 2 + NX // 2]
    # noise-weighted root sum of squares, sqrt(s^H Psi^-1 s), with Psi ~ A A^H
    s = (sens * obj).movedim(0, -1)
    w = torch.einsum("...j,jk,...k->...", s.conj(), torch.linalg.inv(A @ A.conj().T), s)
    truth_opt = torch.sqrt(abs(w))[:, NX * OS // 2 - NX // 2: NX * OS // 2 + NX // 2]
    return ks, truth, truth_sos, truth_opt


@unittest.skipUnless(HAS_IO, "requires ismrmrd")
class TestCartesian(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def _roundtrip(self, ks):
        path = os.path.join(self.tmp.name, "ks.h5")
        write_kspace(path, ks)
        return read_kspace(path)

    def test_sos_matches_truth(self):
        ks, _, truth_sos, _ = _synthetic()
        im = reconstruct_cartesian(self._roundtrip(ks), combine="sos").squeeze()
        self.assertEqual(im.dims, ("z", "y", "x"))
        self.assertEqual(tuple(im.data.shape), (1, NY, NX))
        self.assertTrue(torch.allclose(abs(im.data[0]), truth_sos, atol=0.02))
        self.assertEqual(im.geometry.matrix, (NX, NY, 1))

    def test_ci4_multi_echo(self):
        ks, truth, _, truth_opt = _synthetic(n_echo=2)
        im = reconstruct_cartesian(self._roundtrip(ks), combine="ci4").squeeze()
        self.assertEqual(im.dims, ("contrast", "z", "y", "x"))
        mask = abs(truth) > 0
        r = np.corrcoef(abs(im.data[0, 0])[mask].numpy(), truth_opt[mask].numpy())[0, 1]
        self.assertGreater(r, 0.99)
        dphi = torch.angle(im.data[1, 0] * im.data[0, 0].conj())[mask]
        self.assertTrue(torch.allclose(dphi, torch.tensor(0.5), atol=0.02))

    def test_ci4_single_echo_virtual_coil(self):
        ks, truth, _, truth_opt = _synthetic(n_echo=1)
        im = reconstruct_cartesian(self._roundtrip(ks), combine="ci4").squeeze()
        mask = abs(truth) > 0
        r = np.corrcoef(abs(im.data[0])[mask].numpy(), truth_opt[mask].numpy())[0, 1]
        self.assertGreater(r, 0.99)

    def test_zero_fill_and_window(self):
        ks, _, _, _ = _synthetic()
        im = reconstruct_cartesian(ks, combine="sos", window="tukey", window_params=[None, 0.3, 0.3],
                                   matrix=(64, 64, 1)).squeeze()
        self.assertEqual(tuple(im.data.shape), (1, 64, 64))
        self.assertEqual(im.geometry.matrix, (64, 64, 1))
        self.assertEqual(im.geometry.resolution[:2], (200.0 / 64, 200.0 / 64))

    def test_no_coil_combine(self):
        ks, _, _, _ = _synthetic()
        im = reconstruct_cartesian(ks, combine="none").squeeze()
        self.assertEqual(im.dims, ("coil", "z", "y", "x"))
        self.assertEqual(tuple(im.data.shape), (N_COIL, 1, NY, NX))


if __name__ == "__main__":
    unittest.main()
