"""Storage behaviour for the NIST/MPME port: noise groups, header preservation, image metadata, converter wrapper."""
import os
import subprocess
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from mritorch.data import KSpace, Image
from mritorch.geometry import Geometry

try:
    import ismrmrd
    import h5py
    HAS_IO = True
except ImportError:
    HAS_IO = False

if HAS_IO:
    from mritorch.io import read_kspace, write_kspace, read_image, write_image, make_header
    from mritorch.io.header import parse_header


def _geom(z=0.0):
    return Geometry(position=(1.0, 2.0, z), read_dir=(1.0, 0.0, 0.0), phase_dir=(0.0, 1.0, 0.0),
                    slice_dir=(0.0, 0.0, 1.0), fov=(100.0, 80.0, 5.0), matrix=(10, 8, 1))


def _header(te=5.0):
    return make_header(encoded_matrix=(20, 8, 1), encoded_fov=(200.0, 80.0, 5.0),
                       recon_matrix=(10, 8, 1), recon_fov=(100.0, 80.0, 5.0),
                       TE=[te], TR=[100.0])


def _te(path):
    with h5py.File(path, "r", locking=False) as f:
        return float(parse_header(f["dataset/xml"][0]).sequenceParameters.TE[0])


def _kspace(noise=True):
    g = torch.Generator().manual_seed(0)
    data = torch.randn((2, 1, 1, 1, 1, 1, 3, 1, 8, 20), dtype=torch.cfloat, generator=g)
    nz = torch.randn(3, 64, dtype=torch.cfloat, generator=g) if noise else None
    return KSpace(data=data, header=_header(), noise=nz, geometry={0: _geom(0.0), 1: _geom(5.0)},
                  sample_time_us=20.0)


def _write_noise_group(path, noise, sample_time_us, group="noise"):
    dset = ismrmrd.Dataset(str(path), group, create_if_needed=True)
    try:
        acq = ismrmrd.Acquisition.from_array(noise.numpy())
        acq.set_flag(ismrmrd.ACQ_IS_NOISE_MEASUREMENT)
        acq.sample_time_us = sample_time_us
        dset.append_acquisition(acq)
    finally:
        dset.close()


@unittest.skipUnless(HAS_IO, "requires ismrmrd and h5py")
class TestNoiseGroup(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "raw.h5")

    def tearDown(self):
        self.tmp.cleanup()

    def test_noise_read_from_separate_group(self):
        write_kspace(self.path, _kspace(noise=False))
        noise = torch.randn(3, 128, dtype=torch.cfloat)
        _write_noise_group(self.path, noise, sample_time_us=5.0)
        ks = read_kspace(self.path)
        self.assertIsNotNone(ks.noise)
        self.assertTrue(torch.allclose(ks.noise, noise))
        self.assertEqual(ks.noise_sample_time_us, 5.0)
        self.assertEqual(ks.sample_time_us, 20.0)

    def test_inline_noise_keeps_its_sample_time(self):
        write_kspace(self.path, _kspace(noise=True))
        ks = read_kspace(self.path)
        self.assertEqual(ks.noise.shape, (3, 64))
        self.assertEqual(ks.noise_sample_time_us, 20.0)

    def test_noise_dataset_none_ignores_noise_group(self):
        write_kspace(self.path, _kspace(noise=False))
        _write_noise_group(self.path, torch.randn(3, 16, dtype=torch.cfloat), sample_time_us=5.0)
        ks = read_kspace(self.path, noise_dataset=None)
        self.assertIsNone(ks.noise)

    def test_noise_group_without_noise_flags(self):
        write_kspace(self.path, _kspace(noise=False))
        dset = ismrmrd.Dataset(self.path, "noise", create_if_needed=True)
        try:
            dset.append_acquisition(ismrmrd.Acquisition.from_array(np.zeros((3, 16), np.complex64)))
        finally:
            dset.close()
        self.assertIsNone(read_kspace(self.path).noise)

    def test_no_noise_anywhere(self):
        write_kspace(self.path, _kspace(noise=False))
        ks = read_kspace(self.path)
        self.assertIsNone(ks.noise)
        self.assertIsNone(ks.noise_sample_time_us)


@unittest.skipUnless(HAS_IO, "requires ismrmrd and h5py")
class TestImageStore(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "recon.h5")

    def tearDown(self):
        self.tmp.cleanup()

    def _image(self, value=1.0, te=7.0):
        # non-loop lead dims (pathway, echo) must survive the round trip
        data = torch.arange(2 * 3 * 2 * 1 * 8 * 10, dtype=torch.float32).reshape(2, 3, 2, 1, 8, 10) * value
        return Image(data=data, dims=("pathway", "echo", "slice", "z", "y", "x"), header=_header(te),
                     geometry=_geom(0.0), slice_geometry={0: _geom(0.0), 1: _geom(5.0)},
                     meta={"units": "ms", "source_mids": [373, 374], "scale": 17.89, "description": "test"})

    def test_existing_xml_header_is_preserved(self):
        write_kspace(self.path, _kspace())
        write_image(self.path, self._image(te=7.0), group="recon")
        self.assertEqual(_te(self.path), 5.0)

    def test_header_written_when_absent(self):
        write_image(self.path, self._image(te=9.0), group="recon")
        self.assertEqual(_te(self.path), 9.0)

    def test_refuses_existing_group(self):
        write_image(self.path, self._image(), group="recon")
        with self.assertRaises(FileExistsError):
            write_image(self.path, self._image(), group="recon")

    def test_overwrite_replaces_group(self):
        write_image(self.path, self._image(1.0), group="recon")
        write_image(self.path, self._image(2.0), group="recon", overwrite=True)
        out = read_image(self.path, group="recon", restore=True)
        self.assertTrue(torch.allclose(out.data, self._image(2.0).data))

    def test_dims_meta_and_slice_geometry_roundtrip(self):
        img = self._image()
        write_image(self.path, img, group="recon")
        out = read_image(self.path, group="recon", restore=True)
        self.assertEqual(out.dims, img.dims)
        self.assertEqual(tuple(out.data.shape), tuple(img.data.shape))
        self.assertTrue(torch.allclose(out.data, img.data))
        self.assertEqual(out.meta["units"], "ms")
        self.assertEqual(out.meta["source_mids"], [373, 374])
        self.assertAlmostEqual(out.meta["scale"], 17.89)
        for k in (0, 1):
            self.assertEqual(out.slice_geometry[k], img.slice_geometry[k])
            self.assertTrue(torch.allclose(out.slice_geometry[k].affine, img.slice_geometry[k].affine))

    def test_complex_coil_image_roundtrip(self):
        data = torch.randn(3, 2, 1, 8, 10, dtype=torch.cfloat)
        img = Image(data=data, dims=("echo", "coil", "z", "y", "x"), geometry=_geom(0.0))
        write_image(self.path, img, group="coils")
        out = read_image(self.path, group="coils", restore=True)
        self.assertEqual(out.dims, img.dims)
        self.assertTrue(torch.allclose(out.data, data))


@unittest.skipUnless(HAS_IO, "requires ismrmrd and h5py")
class TestImageStoreEdges(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "recon.h5")

    def tearDown(self):
        self.tmp.cleanup()

    def test_coil_moved_next_to_spatial_dims(self):
        data = torch.randn(4, 3, 1, 8, 10, dtype=torch.cfloat)
        write_image(self.path, Image(data=data, dims=("coil", "echo", "z", "y", "x"), geometry=_geom()), group="g")
        out = read_image(self.path, group="g", restore=True)
        self.assertEqual(out.dims, ("echo", "coil", "z", "y", "x"))
        self.assertTrue(torch.allclose(out.data, data.permute(1, 0, 2, 3, 4)))

    def test_restore_falls_back_without_mritorch_meta(self):
        dset = ismrmrd.Dataset(self.path, "dataset", create_if_needed=True)
        try:
            im = ismrmrd.Image.from_array(np.ones((1, 1, 8, 10), np.float32), transpose=False)
            dset.append_image("plain", im)
        finally:
            dset.close()
        out = read_image(self.path, group="plain", restore=True)
        self.assertEqual(out.dims[-4:], ("coil", "z", "y", "x"))
        self.assertEqual(len(out.dims), 10)

    def test_meta_values_converted(self):
        meta = {"scale": np.float32(2.5), "mids": np.array([1, 2]), "t": torch.tensor(3.0), "bad": float("nan")}
        write_image(self.path, Image(data=torch.zeros(1, 8, 10), dims=("z", "y", "x"), geometry=_geom(), meta=meta),
                    group="g")
        out = read_image(self.path, group="g", restore=True).meta
        self.assertEqual(out, {"scale": 2.5, "mids": [1, 2], "t": 3.0, "bad": None})

    def test_meta_rejects_unknown_types(self):
        img = Image(data=torch.zeros(1, 8, 10), dims=("z", "y", "x"), geometry=_geom(), meta={"x": object()})
        with self.assertRaises(TypeError):
            write_image(self.path, img, group="g")

    def test_counter_layout_warns_on_shared_counters(self):
        img = Image(data=torch.zeros(3, 1, 8, 10), dims=("echo", "z", "y", "x"), geometry=_geom())
        write_image(self.path, img, group="g")
        with self.assertWarns(UserWarning):
            read_image(self.path, group="g")


class TestConverterWrapper(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dat = os.path.join(self.tmp.name, "meas.dat")
        open(self.dat, "wb").close()
        self.out = os.path.join(self.tmp.name, "meas.h5")

    def tearDown(self):
        self.tmp.cleanup()

    def _run(self, **kw):
        from mritorch.io.convert import siemens_to_ismrmrd
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append((cmd, kwargs))
            with open(cmd[cmd.index("-o") + 1], "ab") as f:
                f.write(b"x")
            return subprocess.CompletedProcess(cmd, 0, "", "")
        with mock.patch("mritorch.io.convert.find_converter", return_value="s2i"), \
                mock.patch("mritorch.io.convert.subprocess.run", side_effect=fake_run):
            siemens_to_ismrmrd(self.dat, self.out, **kw)
        return calls

    def test_refuses_existing_output(self):
        from mritorch.io.convert import siemens_to_ismrmrd
        open(self.out, "wb").close()
        with mock.patch("mritorch.io.convert.find_converter", return_value="s2i"):
            with self.assertRaises(FileExistsError):
                siemens_to_ismrmrd(self.dat, self.out)

    def test_disables_hdf5_locking_in_subprocess(self):
        before = os.environ.get("HDF5_USE_FILE_LOCKING")
        calls = self._run()
        self.assertEqual(calls[0][1]["env"]["HDF5_USE_FILE_LOCKING"], "FALSE")
        self.assertEqual(os.environ.get("HDF5_USE_FILE_LOCKING"), before)

    def test_overwrite_replaces_output_only_on_success(self):
        with open(self.out, "wb") as f:
            f.write(b"old")
        self._run(overwrite=True, noise=True)
        with open(self.out, "rb") as f:
            self.assertEqual(f.read(), b"xx")
        self.assertFalse(os.path.exists(self.out + ".partial"))

    def test_failed_noise_run_leaves_no_partial_and_keeps_old(self):
        from mritorch.io.convert import siemens_to_ismrmrd
        with open(self.out, "wb") as f:
            f.write(b"old")
        n = {"i": 0}

        def flaky(cmd, **kwargs):
            n["i"] += 1
            with open(cmd[cmd.index("-o") + 1], "ab") as f:
                f.write(b"x")
            return subprocess.CompletedProcess(cmd, 0 if n["i"] == 1 else 3, "", "boom")
        with mock.patch("mritorch.io.convert.find_converter", return_value="s2i"), \
                mock.patch("mritorch.io.convert.subprocess.run", side_effect=flaky):
            with self.assertRaises(RuntimeError):
                siemens_to_ismrmrd(self.dat, self.out, noise=True, overwrite=True)
        self.assertFalse(os.path.exists(self.out + ".partial"))
        with open(self.out, "rb") as f:
            self.assertEqual(f.read(), b"old")

    def test_noise_group_must_differ(self):
        with self.assertRaises(ValueError):
            self._run(noise=True, noise_group="dataset")

    def test_noise_measurement_into_noise_group(self):
        calls = self._run(noise=True)
        self.assertEqual(len(calls), 2)
        cmd = calls[1][0]
        self.assertEqual(cmd[cmd.index("-z") + 1], "1")
        self.assertEqual(cmd[cmd.index("-g") + 1], "noise")
        self.assertEqual(calls[0][0][calls[0][0].index("-z") + 1], "-1")


if __name__ == "__main__":
    unittest.main()
