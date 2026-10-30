"""
Cartesian reconstruction.

Replaces MATLAB `kspace_to_imspace_cartesian` + `crop_to_size` + k-space
windowing + `zfi_to_size`. Partial-Fourier zero-filling happens when the
k-space is read (`mritorch.io.read_kspace` places lines about the k-space
center).
"""
from __future__ import annotations
from typing import Literal, Optional, Sequence
import warnings

import torch

from mritorch.data import KSpace, Image, LOOP_DIMS
from mritorch.fft import ifftc, fftc
from mritorch.filters import apply_window
from mritorch.utils.array import crop_to_size, resize_center
from mritorch.coils import noise_covariance, sos, ci4_combine, optimal_combine
from mritorch.recon.corrections import drift_correction
from mritorch.io.header import encoding_info

Combine = Literal["ci4", "optimal", "sos", "none"]


def reconstruct_cartesian(
        ks: KSpace,
        combine: Combine="ci4",
        t_ave: tuple[int, int]=(0, 1),
        drift_baseline: int=0,
        window: Optional[str]=None,
        window_params: Optional[Sequence[Optional[float]]]=None,
        matrix: Optional[Sequence[int]]=None,
        ) -> Image:
    """
    Reconstruct fully sampled (or zero-filled) Cartesian k-space.

    Steps: optional drift correction, inverse FFT over (kz, ky, kx), coil
    combine, crop to the recon matrix (removes readout oversampling),
    optional k-space window, optional zero-filled interpolation.

    Parameters
    ----------
    ks : KSpace
        K-space in `KSPACE_DIMS` layout, from `mritorch.io.read_kspace`.
    combine : {'ci4', 'optimal', 'sos', 'none'}, optional
        Coil combination. 'ci4' reproduces `CreateImg4` mode 3 (contrast is
        the echo axis, repetition the dynamic axis); 'optimal' uses each
        image as its own reference; 'none' keeps the coil axis. Falls back to
        'sos' when there is no noise scan. Defaults to 'ci4'.
    t_ave : tuple of int, optional
        Half-open range of repetitions used as the 'ci4' reference.
    drift_baseline : int, optional
        If > 0, apply `drift_correction` over repetitions with this many
        baseline dynamics.
    window : str, optional
        Window name ('tukey', 'hann', 'fermi') applied to the k-space of the
        cropped image.
    window_params : sequence of float or None, optional
        Window parameter for (z, y, x); None skips an axis.
    matrix : sequence of int, optional
        Output matrix (x, y, z) for zero-filled interpolation. Defaults to the
        recon matrix.

    Returns
    -------
    Image
        Image with dims ``LOOP_DIMS + ([coil,] z, y, x)``, per-slice geometry
        and ``meta['scale']`` noting the orthonormal FFT convention.
    """
    if tuple(ks.dims[:len(LOOP_DIMS)]) != LOOP_DIMS:
        raise ValueError("reconstruct_cartesian expects k-space in KSPACE_DIMS layout")
    data = ks.data
    coil_dim = ks.dim("coil")

    if drift_baseline > 0:
        data = drift_correction(
            data, time_dim=ks.dim("repetition"),
            average_dims=(ks.dim("kx"), ks.dim("kz"), coil_dim), n_baseline=drift_baseline)

    im = ifftc(data, dim=(-3, -2, -1))

    if combine != "none" and ks.noise is None and combine != "sos":
        warnings.warn("No noise scan; using root-sum-of-squares coil combine")
        combine = "sos"
    if combine == "sos":
        im = sos(im, coil_dim=coil_dim).to(im.dtype)
    elif combine == "optimal":
        invcov = torch.linalg.inv(noise_covariance(ks.noise.to(im.device)))  # type: ignore
        im = optimal_combine(im, invcov, coil_dim=coil_dim)
    elif combine == "ci4":
        im = ci4_combine(
            im, noise_covariance(ks.noise.to(im.device)), coil_dim=coil_dim,  # type: ignore
            echo_dim=ks.dim("contrast"), time_dim=ks.dim("repetition"), t_ave=t_ave)
    elif combine != "none":
        raise ValueError(f"Unknown combine option {combine!r}")

    # Crop to the recon matrix (x: readout oversampling; z: slice oversampling)
    info = encoding_info(ks.header) if ks.header is not None else None
    if info is not None:
        rx, ry, rz = info.recon_matrix
        nz = im.shape[-3]
        im = crop_to_size(im, (rz if nz > 1 else 1, ry, rx))

    if window is not None:
        params = window_params if window_params is not None else [None, 0.5, 0.5]
        im = ifftc(apply_window(fftc(im), window, params))

    if matrix is not None:
        mx, my, mz = matrix
        k = fftc(im, norm="forward")
        im = ifftc(resize_center(k, (mz, my, mx)), norm="forward")

    out_matrix = (im.shape[-1], im.shape[-2], im.shape[-3])
    slice_geometry = {s: g.with_matrix(out_matrix) for s, g in ks.geometry.items()}
    dims = LOOP_DIMS + (("coil",) if combine == "none" else ()) + ("z", "y", "x")
    return Image(
        data=im, geometry=slice_geometry.get(0), dims=dims, header=ks.header,
        slice_geometry=slice_geometry, meta={"fft_norm": "ortho"},
    )
