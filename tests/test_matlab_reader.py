"""MATLAB reader for MRITorch ISMRMRD image groups (`matlab/read_ismrmrd_image.m`).

Marked ``matlab``: deselected unless ``--run-matlab`` is given. The MATLAB
executable is ``$MRITORCH_MATLAB`` if set, else ``matlab`` on PATH. The fixture is
written next to this file (on the repository's file system, e.g. /ceph) so the
test also exercises HDF5 file locking being disabled on NFS.
"""
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from mritorch.data import Image
from mritorch.geometry import Geometry

pytestmark = pytest.mark.matlab

REPO = Path(__file__).resolve().parents[1]


def _geom(z):
    return Geometry(position=(1.0, 2.0, z), read_dir=(1.0, 0.0, 0.0), phase_dir=(0.0, 1.0, 0.0),
                    slice_dir=(0.0, 0.0, 1.0), fov=(100.0, 80.0, 5.0), matrix=(10, 8, 1))


def _matlab_order(data: np.ndarray, n_spatial_lead: int) -> np.ndarray:
    """Python (lead..., [coil], z, y, x) -> MATLAB (x, y, z, [coil], lead...)."""
    nd = data.ndim
    lead = list(range(n_spatial_lead))
    rest = list(range(n_spatial_lead, nd - 3))
    return np.transpose(data, [nd - 1, nd - 2, nd - 3, *rest, *lead])


def test_matlab_reads_mritorch_image_groups(monkeypatch):
    pytest.importorskip("ismrmrd")
    scipy_io = pytest.importorskip("scipy.io")
    # MRITORCH_MATLAB overrides PATH lookup (some site launchers drop arguments)
    matlab = os.environ.get("MRITORCH_MATLAB") or shutil.which("matlab")
    if matlab is None:
        pytest.skip("matlab not found (set MRITORCH_MATLAB or put matlab on PATH)")
    import ismrmrd
    from mritorch.io import write_image, make_header
    # the fixture writer (ismrmrd package) opens files with default locking
    monkeypatch.setenv("HDF5_USE_FILE_LOCKING", "FALSE")

    header = make_header(encoded_matrix=(20, 8, 1), encoded_fov=(200.0, 80.0, 5.0), TE=[5.0])
    recon = torch.arange(2 * 3 * 2 * 1 * 8 * 10, dtype=torch.float32).reshape(2, 3, 2, 1, 8, 10)
    img = Image(data=recon, dims=("pathway", "echo", "slice", "z", "y", "x"), header=header,
                geometry=_geom(0.0), slice_geometry={0: _geom(0.0), 1: _geom(5.0)},
                meta={"units": "ms", "source_mids": [373, 374]})
    g = torch.Generator().manual_seed(1)
    coils = torch.randn(3, 4, 1, 8, 10, dtype=torch.cfloat, generator=g)
    cimg = Image(data=coils, dims=("echo", "coil", "z", "y", "x"), geometry=_geom(0.0))

    with tempfile.TemporaryDirectory(dir=REPO / "tests") as tmp:
        h5 = os.path.join(tmp, "fixture.h5")
        write_image(h5, img, group="recon")
        write_image(h5, cimg, group="coils")
        # single 2D image without lead dims (nz = 1)
        single = torch.arange(80, dtype=torch.float32).reshape(1, 8, 10)
        write_image(h5, Image(data=single, dims=("z", "y", "x"), geometry=_geom(0.0)), group="single")
        # group without MRITorch metadata
        dset = ismrmrd.Dataset(h5, "dataset", create_if_needed=True)
        try:
            dset.append_image("plain", ismrmrd.Image.from_array(single.numpy()[None], transpose=False))
        finally:
            dset.close()
        expected = {
            "recon": _matlab_order(recon.numpy(), 3),
            "recon_dims": np.array(["x", "y", "z", "pathway", "echo", "slice"], dtype=object),
            "coils": _matlab_order(coils.numpy(), 1),
            "coils_dims": np.array(["x", "y", "z", "coil", "echo"], dtype=object),
            "single": single.numpy()[0].T,
            "affines": np.stack([_geom(0.0).affine.numpy(), _geom(5.0).affine.numpy()], axis=-1),
        }
        mat = os.path.join(tmp, "expected.mat")
        scipy_io.savemat(mat, expected)
        cmd = (f"addpath('{REPO / 'matlab'}'); "
               f"test_read_ismrmrd_image('{h5}', '{mat}')")
        # MATLAB must be launched with the variable set: a setenv inside MATLAB
        # comes too late (MATLAB launched without it hung on this /ceph fixture
        # until killed, 2026-11-23), so read_ismrmrd_image documents this
        env = {**os.environ, "HDF5_USE_FILE_LOCKING": "FALSE"}
        res = subprocess.run([matlab, "-batch", cmd], capture_output=True, text=True, env=env, timeout=600)
        assert res.returncode == 0, res.stdout + res.stderr
        assert "test_read_ismrmrd_image: OK" in res.stdout
