import unittest

import torch

from mritorch.geometry import Geometry, voxel_coordinates, resample, fov_intersection_mask, lps_to_ras


def _geom(position=(10.0, -5.0, 3.0), matrix=(8, 6, 4), fov=(80.0, 60.0, 20.0)):
    return Geometry(position=position, read_dir=(1.0, 0.0, 0.0), phase_dir=(0.0, 1.0, 0.0),
                    slice_dir=(0.0, 0.0, 1.0), fov=fov, matrix=matrix)


class TestGeometry(unittest.TestCase):
    def test_center_voxel_at_position(self):
        g = _geom()
        center = torch.tensor([4.0, 3.0, 2.0, 1.0], dtype=torch.float64)
        self.assertTrue(torch.allclose((g.affine @ center)[:3], torch.tensor(g.position, dtype=torch.float64)))

    def test_resolution_and_coords(self):
        g = _geom()
        self.assertEqual(g.resolution, (10.0, 10.0, 5.0))
        c = voxel_coordinates(g)
        self.assertEqual(c.shape, (4, 6, 8, 3))
        self.assertTrue(torch.allclose(c[0, 0, 1] - c[0, 0, 0], torch.tensor([10.0, 0, 0], dtype=torch.float64)))
        self.assertTrue(torch.allclose(c[1, 0, 0] - c[0, 0, 0], torch.tensor([0, 0, 5.0], dtype=torch.float64)))

    def test_oblique_axes(self):
        s = 2 ** -0.5
        g = Geometry(position=(0, 0, 0), read_dir=(s, s, 0), phase_dir=(-s, s, 0), slice_dir=(0, 0, 1),
                     fov=(40, 40, 4), matrix=(4, 4, 1))
        c = voxel_coordinates(g)
        step = c[0, 0, 1] - c[0, 0, 0]
        self.assertTrue(torch.allclose(step, torch.tensor([10 * s, 10 * s, 0], dtype=torch.float64)))

    def test_lps_to_ras(self):
        A = lps_to_ras(_geom().affine)
        self.assertEqual(float(A[0, 3]), -_geom().affine[0, 3].item())

    def test_resample_identity(self):
        g = _geom()
        im = torch.randn(2, 4, 6, 8)
        self.assertTrue(torch.allclose(resample(im, g, g), im, atol=1e-5))

    def test_resample_shift(self):
        src = _geom()
        tgt = _geom(position=(20.0, -5.0, 3.0))  # shifted one voxel in +x
        im = torch.arange(8, dtype=torch.float32).expand(4, 6, 8).clone()
        out = resample(im, src, tgt)
        self.assertTrue(torch.allclose(out[..., :-1], im[..., 1:]))
        self.assertTrue(torch.allclose(out[..., -1], torch.zeros(()), atol=1e-5))

    def test_resample_complex_and_upsample(self):
        src = _geom()
        tgt = _geom(matrix=(16, 12, 8))
        im = torch.ones(4, 6, 8, dtype=torch.cfloat) * (1 + 2j)
        out = resample(im, src, tgt)
        self.assertEqual(out.shape, (8, 12, 16))
        inner = out[2:-2, 2:-2, 2:-2]
        self.assertTrue(torch.allclose(inner, torch.full_like(inner, 1 + 2j), atol=1e-5))

    def test_resample_2d_slice_into_3d(self):
        src = _geom(matrix=(8, 6, 1), fov=(80.0, 60.0, 5.0))
        tgt = _geom(matrix=(8, 6, 4), fov=(80.0, 60.0, 20.0))
        out = resample(torch.ones(1, 6, 8), src, tgt)
        # source slice is centered at z index 2 of the target
        self.assertTrue(torch.all(out[2] == 1))
        self.assertTrue(torch.all(out[0] == 0))

    def test_fov_intersection(self):
        a = _geom()
        b = _geom(position=(40.0, -5.0, 3.0))  # shifted 3 voxels in +x
        mask = fov_intersection_mask(a, b)
        self.assertEqual(mask.shape, (4, 6, 8))
        self.assertTrue(torch.all(mask[..., 3:]))
        self.assertFalse(torch.any(mask[..., :3]))


class TestPcsToDcs(unittest.TestCase):
    """Lab DCS = Siemens PCS (LPH+) of a head-first-prone patient (MedImgeReslicer calc_coords_RAW)."""

    def test_flips_per_patient_position(self):
        from mritorch.geometry import pcs_to_dcs
        p = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float64)
        expect = {"HFP": (1, 2, 3), "HFS": (-1, -2, 3), "FFP": (-1, 2, -3), "FFS": (1, -2, -3)}
        for pos, e in expect.items():
            self.assertTrue(torch.equal(pcs_to_dcs(p, pos), torch.tensor([e], dtype=torch.float64)), pos)

    def test_involution_and_shapes(self):
        from mritorch.geometry import pcs_to_dcs
        p = torch.randn(4, 5, 3, dtype=torch.float64)
        for pos in ("HFP", "HFS", "FFP", "FFS"):
            self.assertTrue(torch.allclose(pcs_to_dcs(pcs_to_dcs(p, pos), pos), p))

    def test_affine_input(self):
        from mritorch.geometry import pcs_to_dcs_affine
        A = torch.eye(4, dtype=torch.float64)
        A[:3, 3] = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        D = pcs_to_dcs_affine(A, "HFS")
        self.assertTrue(torch.equal(D[:3, 3], torch.tensor([-1.0, -2.0, 3.0], dtype=torch.float64)))
        self.assertTrue(torch.equal(torch.diag(D[:3, :3]), torch.tensor([-1.0, -1.0, 1.0], dtype=torch.float64)))

    def test_unknown_position(self):
        from mritorch.geometry import pcs_to_dcs
        with self.assertRaises(ValueError):
            pcs_to_dcs(torch.zeros(1, 3), "HFDL")


if __name__ == "__main__":
    unittest.main()
