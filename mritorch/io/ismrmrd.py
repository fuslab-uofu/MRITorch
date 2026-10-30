"""
Read and write ISMRMRD (v1 HDF5) k-space and images.

Reading uses `h5py` directly on the ISMRMRD layout, which is much faster than
looping over `ismrmrd.Dataset.read_acquisition`. Writing uses the `ismrmrd`
package.

K-space is sorted by `EncodingCounters` into the `mritorch.data.KSPACE_DIMS`
layout. Cartesian lines are placed so that the k-space center from
`encodingLimits` (and each readout's `center_sample`) lands at index
``N // 2``; partial-Fourier and asymmetric-echo data are therefore
zero-filled in place.
"""
from __future__ import annotations
from typing import Optional, Sequence
from pathlib import Path

import numpy as np
import torch

from mritorch.data import KSpace, Image, KSPACE_DIMS, LOOP_DIMS, IMAGE_DIMS
from mritorch.geometry import Geometry
from mritorch.io.header import parse_header, header_to_xml, encoding_info

# idx field name for each loop dim, in KSPACE_DIMS order
_LOOP_IDX = {"slice": "slice", "set": "set", "phase": "phase", "contrast": "contrast",
             "repetition": "repetition", "average": "average"}

# Acquisitions that are not imaging data
_SKIP_FLAGS = (
    "ACQ_IS_NOISE_MEASUREMENT", "ACQ_IS_NAVIGATION_DATA", "ACQ_IS_PHASECORR_DATA",
    "ACQ_IS_DUMMYSCAN_DATA", "ACQ_IS_RTFEEDBACK_DATA", "ACQ_IS_HPFEEDBACK_DATA",
    "ACQ_IS_SURFACECOILCORRECTIONSCAN_DATA", "ACQ_IS_PHASE_STABILIZATION",
    "ACQ_IS_PHASE_STABILIZATION_REFERENCE",
)


def _flag_mask(flags: np.ndarray, name: str) -> np.ndarray:
    import ismrmrd
    bit = np.uint64(1) << np.uint64(getattr(ismrmrd, name) - 1)
    return (flags & bit) != 0


def _place(idx: np.ndarray, n_encoded: int, center: Optional[int]) -> tuple[int, int]:
    """Matrix size and offset so that `center` lands at ``n // 2``."""
    lo, hi = int(idx.min()), int(idx.max())
    if center is None:
        return max(n_encoded, hi + 1), 0
    n = max(n_encoded, 1)
    while True:
        off = n // 2 - center
        if lo + off >= 0 and hi + off < n:
            return n, off
        n += 1


def read_kspace(
        path: str | Path,
        dataset: str="dataset",
        encoding: int=0,
        keep_calibration: bool=False,
        dtype: torch.dtype=torch.complex64,
        ) -> KSpace:
    """
    Read and sort k-space from an ISMRMRD HDF5 file.

    Parameters
    ----------
    path : str or Path
        ISMRMRD HDF5 file.
    dataset : str, optional
        HDF5 group name. Defaults to 'dataset'.
    encoding : int, optional
        Encoding space to read (`encoding_space_ref`). Defaults to 0.
    keep_calibration : bool, optional
        Keep calibration-only lines (`ACQ_IS_PARALLEL_CALIBRATION`).
        Defaults to False.
    dtype : torch.dtype, optional
        Complex dtype of the output. Defaults to `torch.complex64`.

    Returns
    -------
    KSpace
        Sorted k-space with header, noise scans, per-slice geometry and (for
        non-Cartesian data) trajectory.
    """
    import h5py
    with h5py.File(path, "r") as f:
        g = f[dataset]
        header = parse_header(g["xml"][0])
        raw = g["data"][:]

    head = raw["head"]
    flags = head["flags"]
    noise_mask = _flag_mask(flags, "ACQ_IS_NOISE_MEASUREMENT")
    skip = np.zeros(len(head), dtype=bool)
    for name in _SKIP_FLAGS:
        skip |= _flag_mask(flags, name)
    if not keep_calibration:
        skip |= _flag_mask(flags, "ACQ_IS_PARALLEL_CALIBRATION")
    skip |= head["encoding_space_ref"] != encoding
    keep = ~skip

    noise = None
    if noise_mask.any():
        nh = head[noise_mask]
        blocks = [np.asarray(d).view(np.complex64).reshape(c, s)
                  for d, c, s in zip(raw["data"][noise_mask], nh["active_channels"], nh["number_of_samples"])]
        noise = torch.from_numpy(np.concatenate(blocks, axis=1)).to(dtype)

    h = head[keep]
    if len(h) == 0:
        raise ValueError(f"No imaging acquisitions in {path}")
    n_ch = np.unique(h["active_channels"])
    n_s = np.unique(h["number_of_samples"])
    if len(n_ch) > 1 or len(n_s) > 1:
        raise ValueError("Acquisitions with varying channel or sample counts are not supported")
    n_ch, n_s = int(n_ch[0]), int(n_s[0])
    samples = np.stack(raw["data"][keep]).view(np.complex64).reshape(len(h), n_ch, n_s)
    reverse = _flag_mask(h["flags"], "ACQ_IS_REVERSE")
    samples[reverse] = samples[reverse, :, ::-1]

    info = encoding_info(header, encoding)
    cartesian = info.trajectory.lower() == "cartesian"
    idx = h["idx"]
    e1, e2 = idx["kspace_encode_step_1"].astype(np.int64), idx["kspace_encode_step_2"].astype(np.int64)

    if cartesian:
        c0 = int(h["center_sample"][0])
        nx = max(info.encoded_matrix[0], n_s)
        ox = nx // 2 - c0 if c0 > 0 else (nx - n_s) // 2
        ox = min(max(ox, 0), nx - n_s)
        ny, oy = _place(e1, info.encoded_matrix[1], info.center_ky)
        nz, oz = _place(e2, info.encoded_matrix[2], info.center_kz)
    else:
        nx, ox = n_s, 0
        ny, oy = int(e1.max()) + 1, 0
        nz, oz = int(e2.max()) + 1, 0

    loops = [idx[_LOOP_IDX[d]].astype(np.int64) for d in LOOP_DIMS]
    loop_sizes = [int(l.max()) + 1 for l in loops]
    out = np.zeros((*loop_sizes, n_ch, nz, ny, nx), dtype=np.complex64)
    out[(*loops, slice(None), e2 + oz, e1 + oy, slice(ox, ox + n_s))] = samples

    traj = None
    n_dim = int(h["trajectory_dimensions"][0])
    if n_dim > 0:
        t = np.stack(raw["traj"][keep]).reshape(len(h), n_s, n_dim)
        tout = np.zeros((*loop_sizes, nz, ny, nx, n_dim), dtype=np.float32)
        tout[(*loops, e2 + oz, e1 + oy, slice(ox, ox + n_s))] = t
        traj = torch.from_numpy(tout)

    geometry: dict[int, Geometry] = {}
    slices = idx["slice"]
    for s in np.unique(slices):
        first = h[np.argmax(slices == s)]
        geometry[int(s)] = Geometry.from_header(
            first, fov=info.recon_fov, matrix=info.recon_matrix)

    return KSpace(
        data=torch.from_numpy(out).to(dtype), header=header, dims=KSPACE_DIMS,
        traj=traj, noise=noise, geometry=geometry,
        sample_time_us=float(h["sample_time_us"][0]),
    )


def _set_geometry(hdr, geom: Optional[Geometry]) -> None:
    if geom is None:
        return
    hdr.position[:] = geom.position
    hdr.read_dir[:] = geom.read_dir
    hdr.phase_dir[:] = geom.phase_dir
    hdr.slice_dir[:] = geom.slice_dir


def write_kspace(path: str | Path, ks: KSpace, dataset: str="dataset") -> None:
    """
    Write k-space to an ISMRMRD HDF5 file.

    Every readout line that contains a nonzero sample is written, with its
    encoding counters taken from its position in the tensor. `ks.dims` must
    be `KSPACE_DIMS` and `ks.header` must be set.
    """
    import ismrmrd
    if tuple(ks.dims) != KSPACE_DIMS:
        raise ValueError(f"write_kspace expects dims {KSPACE_DIMS}, got {ks.dims}")
    if ks.header is None:
        raise ValueError("write_kspace requires an ISMRMRD header")

    data = ks.data.detach().cpu().to(torch.complex64).numpy()
    nx = data.shape[-1]
    dset = ismrmrd.Dataset(str(path), dataset, create_if_needed=True)
    try:
        dset.write_xml_header(header_to_xml(ks.header))
        if ks.noise is not None:
            acq = ismrmrd.Acquisition.from_array(ks.noise.cpu().to(torch.complex64).numpy())
            acq.set_flag(ismrmrd.ACQ_IS_NOISE_MEASUREMENT)
            dset.append_acquisition(acq)

        lines = np.argwhere(np.abs(data).max(axis=(KSPACE_DIMS.index("coil"), -1)) > 0)
        for counter, pos in enumerate(lines):
            *loop, kz, ky = (int(p) for p in pos)
            acq = ismrmrd.Acquisition.from_array(np.ascontiguousarray(data[(*loop, slice(None), kz, ky)]))
            for name, val in zip(LOOP_DIMS, loop):
                setattr(acq.idx, _LOOP_IDX[name], val)
            acq.idx.kspace_encode_step_1 = ky
            acq.idx.kspace_encode_step_2 = kz
            acq.center_sample = nx // 2
            acq.scan_counter = counter
            if ks.sample_time_us is not None:
                acq.sample_time_us = ks.sample_time_us
            _set_geometry(acq, ks.geometry.get(loop[0]))
            if ks.traj is not None:
                t = ks.traj[(*loop, kz, ky)].cpu().numpy()
                acq.resize(nx, acq.active_channels, t.shape[-1])
                acq.data[:] = data[(*loop, slice(None), kz, ky)]
                acq.traj[:] = t
            dset.append_acquisition(acq)
    finally:
        dset.close()


def write_image(
        path: str | Path,
        image: Image,
        group: str="image_0",
        dataset: str="dataset",
        series_index: int=0,
        ) -> None:
    """
    Write an `Image` as ISMRMRD images.

    Leading dims named in `LOOP_DIMS` set the matching image header counters;
    a 'coil' dim becomes the image channel axis. All other leading dims are
    flattened into `image_index`. Geometry comes from `image.slice_geometry`
    (by slice index) or `image.geometry`.
    """
    import ismrmrd
    img = image
    if "coil" in img.dims:
        order = [d for d in img.dims if d != "coil" and d not in ("z", "y", "x")] + ["coil", "z", "y", "x"]
        img = img.permute(order)
    data = img.data.detach().cpu().numpy()
    has_coil = "coil" in img.dims
    n_lead = img.data.ndim - (4 if has_coil else 3)
    lead_names = img.dims[:n_lead]
    lead_shape = data.shape[:n_lead]

    dset = ismrmrd.Dataset(str(path), dataset, create_if_needed=True)
    try:
        if image.header is not None:
            dset.write_xml_header(header_to_xml(image.header))
        for i, pos in enumerate(np.ndindex(*lead_shape) if lead_shape else [()]):
            block = data[pos]
            if not has_coil:
                block = block[None]
            im = ismrmrd.Image.from_array(np.ascontiguousarray(block), transpose=False)
            counters = dict(zip(lead_names, pos))
            for name in LOOP_DIMS:
                if name in counters:
                    setattr(im, name, int(counters[name]))
            geom = image.slice_geometry.get(int(counters.get("slice", 0)), image.geometry)
            _set_geometry(im, geom)
            if geom is not None:
                im.field_of_view[:] = geom.fov
            im.image_index = i
            im.image_series_index = series_index
            im.image_type = ismrmrd.IMTYPE_COMPLEX if np.iscomplexobj(block) else ismrmrd.IMTYPE_MAGNITUDE
            dset.append_image(group, im)
    finally:
        dset.close()


def read_image(path: str | Path, group: str="image_0", dataset: str="dataset") -> Image:
    """
    Read ISMRMRD images into an `Image` with `IMAGE_DIMS` layout.

    Images are placed by their header counters; singleton loop dims are
    kept (call `.squeeze()` to drop them). Geometry is taken per slice from
    the image headers.
    """
    import h5py
    with h5py.File(path, "r") as f:
        g = f[dataset]
        header = parse_header(g["xml"][0]) if "xml" in g else None
        heads = g[group]["header"][:]
        raw = g[group]["data"][:]
    if raw.dtype.names is not None:
        raw = raw["real"] + 1j * raw["imag"]
    # raw: (n_images, coil, z, y, x)
    loops = [heads[d].astype(np.int64) for d in LOOP_DIMS]
    loop_sizes = [int(l.max()) + 1 for l in loops]
    out = np.zeros((*loop_sizes, *raw.shape[1:]), dtype=raw.dtype)
    out[tuple(loops)] = raw

    slice_geometry: dict[int, Geometry] = {}
    for s in np.unique(heads["slice"]):
        h = heads[np.argmax(heads["slice"] == s)]
        mx, my, mz = (int(v) for v in h["matrix_size"])
        slice_geometry[int(s)] = Geometry.from_header(h, fov=h["field_of_view"], matrix=(mx, my, mz))
    first = next(iter(slice_geometry.values()))
    return Image(
        data=torch.from_numpy(out), geometry=first, dims=IMAGE_DIMS,
        header=header, slice_geometry=slice_geometry,
    )
