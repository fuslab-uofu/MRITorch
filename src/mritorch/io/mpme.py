"""
Reader for multipathway multi-echo (MPME) acquisitions in ISMRMRD format.

Acquisition layout (`FUSLab_SA/recon_mpme.m`): every acquisition is one readout of ``n_pathways * n_kx`` samples
(the pathway segments in sequence, each ``n_kx`` samples including the readout oversampling) for the phase-encode
line ``kspace_encode_step_1`` (``lin``, the samples along a radial spoke) of the spoke ``kspace_encode_step_2``
(``par``, the view), echo window ``contrast``. The spoke direction is rotated by the golden angle (default
111.246 deg) per view in the phase-encode plane.

`read_mpme` streams the file in chunks, applies the readout window to every pathway segment, transforms the readout
dimension to image space and crops the oversampling, so that the returned array holds one set of radial k-space
samples for every readout plane ``x``.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch

from mritorch.fft import ifftc
from mritorch.filters import fermi
from mritorch.utils.array import crop_to_size
from mritorch.io.header import parse_header

GOLDEN_ANGLE_DEG = 111.246


@dataclass
class MpmeKSpace:
    """
    Radial k-space of an MPME scan with the readout dimension transformed to image space.

    ``data`` is ``(x, sample, view, coil, pathway, echo)``: for each readout plane ``x`` the ``sample`` points along each
    of the ``view`` spokes of the phase-encode plane.
    """
    data: torch.Tensor
    pathways: tuple[int, ...]
    angle_increment: float = GOLDEN_ANGLE_DEG
    header: object = None
    meta: dict = field(default_factory=dict)

    @property
    def calib_index(self) -> int:
        """Index of the pathway used to calibrate the coil sensitivities (p = 0, or the only one)."""
        return self.pathways.index(0) if 0 in self.pathways and len(self.pathways) > 1 else 0


def read_mpme(
        path: str | Path,
        pathways: Optional[Sequence[int]]=None,
        n_x: Optional[int]=None,
        readout_window: Optional[tuple[str, float]]=("fermi", 1 / 16),
        angle_increment: float=GOLDEN_ANGLE_DEG,
        max_views: Optional[int]=None,
        planes: Optional[Sequence[int]]=None,
        norm: str="ortho",
        chunk: int=2000,
        dataset: str="dataset",
        ) -> MpmeKSpace:
    """
    Read an MPME scan.

    Parameters
    ----------
    path : str or Path
        ISMRMRD file from `siemens_to_ismrmrd`.
    pathways : sequence of int, optional
        Coherence pathway orders in readout order. Defaults to the set encoded in the protocol name (``4pwy`` ->
        +1, 0, -1, -2; ``3pwy`` -> +1, 0, -1).
    n_x : int, optional
        Number of readout planes after cropping the oversampling; defaults to the recon matrix of the header.
    readout_window : (str, float) or None, optional
        ``('fermi', r)`` window applied to every pathway segment (default as in `recon_mpme.m`).
    angle_increment : float, optional
        Angle between consecutive views in degrees.
    max_views : int, optional
        Use only the first views.
    planes : sequence of int, optional
        Keep only these readout planes (applied after the readout transform, to save memory).
    norm : {'ortho', 'backward'}, optional
        Normalization of the readout FFT. 'backward' (1/N) reproduces `recon_mpme.m` (MATLAB `ifft`); images are then
        ``sqrt(n_kx)`` (17.89 for 320 samples) smaller than with 'ortho'.
    chunk : int, optional
        Number of acquisitions read at a time.
    """
    import h5py
    with h5py.File(path, "r") as f:
        g = f[dataset]
        header = parse_header(g["xml"][0])
        d = g["data"]
        head = d["head"]
        n_acq = len(head)
        n_samp = int(head["number_of_samples"][0])
        n_coil = int(head["active_channels"][0])
        idx = head["idx"]
        lin, par, eco = (np.asarray(idx[k]) for k in ("kspace_encode_step_1", "kspace_encode_step_2", "contrast"))
        flags = np.asarray(head["flags"])
        noise = (flags & (np.uint64(1) << np.uint64(18))) != 0
        use = ~noise
        n_lin, n_par, n_eco = int(lin[use].max()) + 1, int(par[use].max()) + 1, int(eco[use].max()) + 1
        if max_views is not None:
            n_par = min(n_par, int(max_views))
        if pathways is None:
            pathways = _pathways_from_header(header)
        pathways = tuple(int(p) for p in pathways)
        n_pwy = len(pathways)
        n_kx = n_samp // n_pwy
        if n_kx * n_pwy != n_samp:
            raise ValueError(f"{n_samp} samples per readout is not a multiple of {n_pwy} pathways")
        if n_x is None:
            n_x = int(header.encoding[0].reconSpace.matrixSize.x) if hasattr(header, "encoding") else n_kx // 2
        win = None
        if readout_window is not None:
            if readout_window[0] != "fermi":
                raise ValueError("only the fermi readout window is supported")
            win = fermi(n_kx, readout_window[1], dtype=torch.float32)
        planes = list(range(n_x)) if planes is None else [int(i) for i in planes]
        n_keep = len(planes)
        sel_x = torch.tensor(planes, dtype=torch.long)
        out = torch.zeros(n_keep, n_lin, n_par, n_coil, n_pwy, n_eco, dtype=torch.complex64)

        for i0 in range(0, n_acq, chunk):
            i1 = min(i0 + chunk, n_acq)
            sel = np.nonzero(use[i0:i1] & (par[i0:i1] < n_par))[0]
            if sel.size == 0:
                continue
            rows = d[i0:i1]
            blocks = np.stack([np.asarray(rows["data"][j]).view(np.complex64).reshape(n_coil, n_samp) for j in sel])
            x = torch.from_numpy(blocks).reshape(len(sel), n_coil, n_pwy, n_kx)
            if win is not None:
                x = x * win
            x = ifftc(x, dim=-1, norm=norm)
            x = crop_to_size(x, (n_x,))[..., sel_x]                      # (acq, coil, pwy, x)
            x = x.permute(0, 3, 1, 2)                                   # (acq, x, coil, pwy)
            j = sel + i0
            tl, tp, te = (torch.from_numpy(a[j].astype(np.int64)) for a in (lin, par, eco))
            out[:, tl, tp, :, :, te] = x                                # advanced indices are separated by slices, so they lead: (acq, x, coil, pwy)
    return MpmeKSpace(out, pathways, angle_increment, header,
                      meta=dict(n_samp=n_samp, n_kx=n_kx, n_x=n_x, path=str(path), planes=planes))


def _pathways_from_header(header) -> tuple[int, ...]:
    """Pathways from the protocol name (``..._4pwy_...``)."""
    name = ""
    try:
        name = str(header.measurementInformation.protocolName)
    except AttributeError:
        pass
    for n, p in ((4, (1, 0, -1, -2)), (3, (1, 0, -1)), (2, (0, -1))):
        if f"{n}pwy" in name:
            return p
    raise ValueError(f"cannot infer pathways from protocol name {name!r}; pass pathways=")
