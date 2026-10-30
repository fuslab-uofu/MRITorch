"""
Export images to NIfTI and read legacy MATLAB .mat files.
"""
from __future__ import annotations
from typing import Any
from pathlib import Path

import numpy as np

from mritorch.data import Image
from mritorch.geometry import lps_to_ras


def to_nifti(image: Image, path: str | Path, magnitude: bool=True) -> None:
    """
    Save an `Image` as NIfTI.

    The ``(..., z, y, x)`` data is written as ``(x, y, z, ...)`` with the RAS
    affine from `image.geometry`.

    Parameters
    ----------
    image : Image
        Image with `geometry` set.
    path : str or Path
        Output .nii or .nii.gz path.
    magnitude : bool, optional
        Write the magnitude of complex data (default). If False, complex data
        is written as complex64.
    """
    import nibabel as nib
    data = image.data.detach().cpu().numpy()
    if np.iscomplexobj(data):
        data = np.abs(data).astype(np.float32) if magnitude else data.astype(np.complex64)
    n_lead = data.ndim - 3
    data = np.transpose(data, (data.ndim - 1, data.ndim - 2, data.ndim - 3, *range(n_lead)))
    affine = lps_to_ras(image.affine).numpy()
    nib.save(nib.Nifti1Image(data, affine), str(path))


def load_mat(path: str | Path) -> dict[str, Any]:
    """
    Load a MATLAB .mat file (v5 via scipy, v7.3 via h5py).

    v7.3 arrays are returned in MATLAB (column-major) index order, matching
    the v5 path.
    """
    try:
        from scipy.io import loadmat
        return {k: v for k, v in loadmat(str(path), squeeze_me=True, struct_as_record=False).items()
                if not k.startswith("__")}
    except NotImplementedError:
        pass

    import h5py

    def convert(obj):
        if isinstance(obj, h5py.Dataset):
            arr = obj[()]
            if arr.dtype.names and {"real", "imag"} <= set(arr.dtype.names):
                arr = arr["real"] + 1j * arr["imag"]
            return np.transpose(arr) if isinstance(arr, np.ndarray) else arr
        return {k: convert(v) for k, v in obj.items() if not k.startswith("#")}

    with h5py.File(str(path), "r") as f:
        return {k: convert(v) for k, v in f.items() if not k.startswith("#")}
