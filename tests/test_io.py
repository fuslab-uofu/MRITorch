import os
import tempfile
import unittest

import numpy as np
import torch

from mritorch.data import KSpace, Image, KSPACE_DIMS, IMAGE_DIMS
from mritorch.geometry import Geometry

try:
    import ismrmrd  # noqa: F401
    import h5py  # noqa: F401
    HAS_IO = True
except ImportError:
    HAS_IO = False

if HAS_IO:
    from mritorch.io import (read_kspace, write_kspace, read_image, write_image, make_header,
                             encoding_info, sequence_parameters, user_parameters, to_nifti, load_mat)


def _geom(z=0.0):
    return Geometry(position=(1.0, 2.0, z), read_dir=(1.0, 0.0, 0.0), phase_dir=(0.0, 1.0, 0.0),
                    slice_dir=(0.0, 0.0, 1.0), fov=(100.0, 80.0, 5.0), matrix=(10, 8, 1))


@unittest.skipUnless(HAS_IO, "requires ismrmrd and h5py")
class TestIO(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "test.h5")

    def tearDown(self):
        self.tmp.cleanup()

    def _kspace(self, ny=8, pf_lines=0):
        header = make_header(encoded_matrix=(20, ny, 1), encoded_fov=(200.0, 80.0, 5.0),
                             recon_matrix=(10, 8, 1), recon_fov=(100.0, 80.0, 5.0),
                             TE=[5.0, 10.0], TR=[100.0], user_long={"sWipMemBlock.alFree[1]": 3})
        g = torch.Generator().manual_seed(0)
        shape = (2, 1, 1, 2, 1, 1, 3, 1, ny, 20)  # 2 slices, 2 contrasts, 3 coils
        data = torch.randn(shape, dtype=torch.cfloat, generator=g)
        data[..., :pf_lines, :] = 0
        noise = torch.randn(3, 64, dtype=torch.cfloat, generator=g)
        geometry = {0: _geom(0.0), 1: _geom(5.0)}
        return KSpace(data=data, header=header, noise=noise, geometry=geometry, sample_time_us=5.0)

    def test_kspace_roundtrip(self):
        ks = self._kspace()
        write_kspace(self.path, ks)
        out = read_kspace(self.path)
        self.assertEqual(out.dims, KSPACE_DIMS)
        self.assertEqual(out.data.shape, ks.data.shape)
        self.assertTrue(torch.allclose(out.data, ks.data))
        self.assertTrue(torch.allclose(out.noise, ks.noise))
        self.assertEqual(out.geometry[1].position, (1.0, 2.0, 5.0))
        self.assertEqual(out.geometry[0].matrix, (10, 8, 1))
        self.assertEqual(out.sample_time_us, 5.0)

    def test_partial_fourier_placement(self):
        ks = self._kspace(pf_lines=3)
        write_kspace(self.path, ks)
        out = read_kspace(self.path)
        self.assertEqual(out.data.shape, ks.data.shape)
        self.assertTrue(torch.all(out.data[..., :3, :] == 0))
        self.assertTrue(torch.allclose(out.data, ks.data))

    def test_header_helpers(self):
        ks = self._kspace()
        write_kspace(self.path, ks)
        out = read_kspace(self.path)
        info = encoding_info(out.header)
        self.assertEqual(info.encoded_matrix, (20, 8, 1))
        self.assertEqual(info.recon_fov, (100.0, 80.0, 5.0))
        self.assertEqual(info.trajectory, "cartesian")
        self.assertEqual(sequence_parameters(out.header)["TE"], [5.0, 10.0])
        self.assertEqual(user_parameters(out.header)["sWipMemBlock.alFree[1]"], 3)

    def test_image_roundtrip(self):
        data = torch.randn(2, 1, 1, 3, 1, 1, 1, 8, 10, dtype=torch.cfloat)
        dims = IMAGE_DIMS[:6] + ("z", "y", "x")
        img = Image(data=data, geometry=_geom(), dims=dims, slice_geometry={0: _geom(0.0), 1: _geom(5.0)})
        write_image(self.path, img)
        out = read_image(self.path)
        self.assertEqual(out.dims, IMAGE_DIMS)
        self.assertEqual(out.data.shape, (2, 1, 1, 3, 1, 1, 1, 1, 8, 10))
        self.assertTrue(torch.allclose(out.data[..., 0, :, :, :], data))
        self.assertEqual(out.slice_geometry[1], _geom(5.0))
        self.assertEqual(out.squeeze().dims, ("slice", "contrast", "z", "y", "x"))

    def test_nifti(self):
        import nibabel as nib
        img = Image(data=torch.rand(3, 8, 10), geometry=_geom(), dims=("z", "y", "x"))
        img = Image(data=img.data, geometry=Geometry(**{**_geom().__dict__, "matrix": (10, 8, 3)}),
                    dims=("z", "y", "x"))
        path = os.path.join(self.tmp.name, "t.nii.gz")
        to_nifti(img, path)
        nii = nib.load(path)
        self.assertEqual(nii.shape, (10, 8, 3))
        self.assertTrue(np.allclose(nii.get_fdata()[:, :, 1], img.data[1].numpy().T))
        self.assertTrue(np.allclose(nii.affine[:3, 3], [-img.affine[0, 3], -img.affine[1, 3], img.affine[2, 3]]))

    def test_load_mat_v5(self):
        from scipy.io import savemat
        path = os.path.join(self.tmp.name, "t.mat")
        savemat(path, {"im": np.arange(6).reshape(2, 3) + 1j})
        out = load_mat(path)
        self.assertTrue(np.allclose(out["im"], np.arange(6).reshape(2, 3) + 1j))


if __name__ == "__main__":
    unittest.main()
