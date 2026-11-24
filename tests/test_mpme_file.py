"""`read_mpme` on synthetic ISMRMRD MPME files: geometry (and, later, reversed readouts)."""
import os
import tempfile
import unittest

import numpy as np
import torch

from mritorch.geometry import Geometry

try:
    import ismrmrd
    import h5py  # noqa: F401
    HAS_IO = True
except ImportError:
    HAS_IO = False

HEAD = {"position": (0.0, 20.0, 0.0), "read_dir": (0.0, 0.0, 1.0),
        "phase_dir": (0.0, 1.0, 0.0), "slice_dir": (1.0, 0.0, 0.0)}


def write_mpme_file(path, kdata, n_x, fov=(256.0, 256.0, 256.0), reverse_contrasts=(), head=HEAD):
    """
    Write a synthetic MPME ISMRMRD file.

    kdata : complex array ``(lin, par, contrast, coil, pathway, kx)`` -- each acquisition holds the pathway segments
    back to back. Acquisitions of `reverse_contrasts` are stored time-reversed with ACQ_IS_REVERSE set, as a bipolar
    readout would record them.
    """
    from mritorch.io import make_header
    n_lin, n_par, n_eco, n_coil, n_pwy, n_kx = kdata.shape
    hdr = make_header(encoded_matrix=(n_kx, n_lin, n_par), encoded_fov=(2 * fov[0], fov[1], fov[2]),
                      recon_matrix=(n_x, n_lin, n_lin), recon_fov=fov, TE=[2.0, 5.0, 8.0][:n_eco], TR=[12.0])
    from mritorch.io.header import header_to_xml
    dset = ismrmrd.Dataset(str(path), "dataset", create_if_needed=True)
    try:
        dset.write_xml_header(header_to_xml(hdr))
        for e in range(n_eco):
            for p in range(n_par):
                for l in range(n_lin):
                    samples = kdata[l, p, e].reshape(n_coil, n_pwy * n_kx)
                    if e in reverse_contrasts:
                        samples = samples[:, ::-1]
                    acq = ismrmrd.Acquisition.from_array(np.ascontiguousarray(samples.astype(np.complex64)))
                    if e in reverse_contrasts:
                        acq.set_flag(ismrmrd.ACQ_IS_REVERSE)
                    acq.idx.kspace_encode_step_1 = l
                    acq.idx.kspace_encode_step_2 = p
                    acq.idx.contrast = e
                    for k, v in head.items():
                        getattr(acq, k)[:] = v
                    dset.append_acquisition(acq)
    finally:
        dset.close()


@unittest.skipUnless(HAS_IO, "requires ismrmrd and h5py")
class TestReadMpmeFile(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "mpme.h5")
        rng = np.random.default_rng(0)
        self.k = (rng.standard_normal((4, 3, 3, 2, 3, 8)) + 1j * rng.standard_normal((4, 3, 3, 2, 3, 8)))

    def tearDown(self):
        self.tmp.cleanup()

    def test_geometry_attached(self):
        from mritorch.io import read_mpme
        write_mpme_file(self.path, self.k, n_x=4)
        mk = read_mpme(self.path, pathways=(1, 0, -1), n_x=4, readout_window=None)
        expect = Geometry.from_mpme_header(HEAD, fov=(256.0, 256.0, 256.0), matrix=(4, 4, 4))
        self.assertEqual(mk.geometry, expect)

    def test_planes_subset_has_no_geometry(self):
        from mritorch.io import read_mpme
        write_mpme_file(self.path, self.k, n_x=4)
        mk = read_mpme(self.path, pathways=(1, 0, -1), n_x=4, readout_window=None, planes=[1, 2])
        self.assertIsNone(mk.geometry)

    def test_readout_fov_from_encoding_when_n_x_differs(self):
        from mritorch.io import read_mpme
        write_mpme_file(self.path, self.k, n_x=4)                   # n_kx = 8, encoded FOV x = 512 mm
        mk = read_mpme(self.path, pathways=(1, 0, -1), n_x=2, readout_window=None)
        self.assertAlmostEqual(mk.geometry.fov[0], 128.0)           # 2 * 512 / 8: voxel size unchanged (64 mm)
        self.assertEqual(mk.geometry.matrix, (2, 4, 4))

    def test_mpme_image_rejects_mismatched_grid(self):
        from mritorch.recon.radial import mpme_image
        from mritorch.io.mpme import MpmeKSpace
        g = Geometry.from_mpme_header(HEAD, fov=(256.0, 256.0, 256.0), matrix=(4, 5, 6))
        mk = MpmeKSpace(torch.zeros(1), (1, 0, -1), geometry=g)
        with self.assertRaises(ValueError):
            mpme_image(torch.zeros(2, 5, 6, 3, 2, dtype=torch.cfloat), mk)

    def test_mpme_image_layout(self):
        from mritorch.recon.radial import mpme_image
        from mritorch.io.mpme import MpmeKSpace
        g = Geometry.from_mpme_header(HEAD, fov=(256.0, 256.0, 256.0), matrix=(4, 5, 6))
        mk = MpmeKSpace(torch.zeros(1), (1, 0, -1), geometry=g)
        im = torch.randn(4, 5, 6, 3, 2, dtype=torch.cfloat)        # (x, N1, N2, pathway, echo)
        img = mpme_image(im, mk)
        self.assertEqual(img.dims, ("pathway", "echo", "z", "y", "x"))
        self.assertEqual(tuple(img.data.shape), (3, 2, 6, 5, 4))
        self.assertTrue(torch.equal(img.data[2, 1, 5, 4, 3], im[3, 4, 5, 2, 1]))
        self.assertEqual(img.geometry, g)
        self.assertEqual(img.meta["pathways"], [1, 0, -1])


if __name__ == "__main__":
    unittest.main()
