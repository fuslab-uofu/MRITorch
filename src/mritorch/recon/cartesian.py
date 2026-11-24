"""
Cartesian reconstruction.

Replaces MATLAB `kspace_to_imspace_cartesian` + `crop_to_size` + k-space
windowing + `zfi_to_size`. Partial-Fourier zero-filling happens when the
k-space is read (`mritorch.io.read_kspace` places lines about the k-space
center).
"""
from __future__ import annotations
from typing import Literal, Optional, Sequence

import torch

from mritorch.data import KSpace, Image, LOOP_DIMS
from mritorch.fft import ifftc, fftc
from mritorch.filters import apply_window
from mritorch.utils.array import crop_to_size, resize_center
from mritorch.coils import noise_covariance, sos, ci4_combine, optimal_combine, espirit_maps, sense_combine
from mritorch.recon.corrections import drift_correction
from mritorch.io.header import encoding_info

Combine = Literal["ci4", "optimal", "sos", "espirit", "none"]


def reconstruct_cartesian(
        ks: KSpace,
        combine: Combine="ci4",
        t_ave: tuple[int, int]=(0, 1),
        drift_baseline: int=0,
        window: Optional[str]=None,
        window_params: Optional[Sequence[Optional[float]]]=None,
        matrix: Optional[Sequence[int]]=None,
        espirit: Optional[dict]=None,
        crop_phase_oversampling: bool=True,
        window_before_combine: bool=False,
        ) -> Image:
    """
    Reconstruct fully sampled (or zero-filled) Cartesian k-space.

    Steps: optional drift correction, inverse FFT over (kz, ky, kx), coil
    combine, crop to the recon matrix (removes readout oversampling; MATLAB
    order), optional k-space window (after the combine by default, as MATLAB
    does), optional zero-filled interpolation. With combine 'none'/'espirit'
    or `window_before_combine` the crop (and window) come before the combine.

    Parameters
    ----------
    ks : KSpace
        K-space in `KSPACE_DIMS` layout, from `mritorch.io.read_kspace`.
    combine : {'ci4', 'optimal', 'sos', 'espirit', 'none'}, optional
        Coil combination. 'ci4' reproduces `CreateImg4` mode 3 (contrast is
        the echo axis, repetition the dynamic axis); 'optimal' uses each
        image as its own reference; 'espirit' estimates ESPIRiT maps from the
        first contrast/repetition/average of each slice (BART ``ecalib`` options
        of the MPME reconstruction by default, see `espirit`) and combines with
        `sense_combine` (as BART ``pics`` does for fully sampled data); the
        combination acts on the image cropped to the recon matrix; 'none' keeps
        the coil axis. 'ci4' and 'optimal' fall back to 'sos' when there is no
        noise scan. Defaults to 'ci4'.
    t_ave : tuple of int, optional
        Half-open range of repetitions used as the 'ci4' reference.
    drift_baseline : int, optional
        If > 0, apply `drift_correction` over repetitions with this many
        baseline dynamics.
    espirit : dict, optional
        Options for `espirit_maps` (calib_size, kernel_size, threshold, crop, n_maps) and ``reg`` for
        `sense_combine` (default 0). Defaults: 24, 6, 1e-3, 0.8, 1 and 0 (the MPME reconstruction uses
        ``reg = 5e-4``, BART ``pics -r``).
    window : str, optional
        Window name ('tukey', 'hann', 'fermi') applied to the k-space of the
        cropped image.
    window_params : sequence of float or None, optional
        Window parameter for (z, y, x); None skips an axis.
    matrix : sequence of int, optional
        Output matrix (x, y, z) for zero-filled interpolation. Defaults to the
        recon matrix.
    crop_phase_oversampling : bool, optional
        Crop y to the recon matrix (default). False keeps every encoded
        phase-encode line, as MATLAB ``crop_to_size(im, [lBaseResolution,
        lPhaseEncodingLines])`` does; the geometry's y FOV grows accordingly.
    window_before_combine : bool, optional
        Apply `window` to each coil image before the combine instead of after
        it (the coil images are then cropped before the combine too, which
        changes the single-echo 'ci4' virtual-coil smoothing kernel). A window after a non-linear combine (ci4/optimal/sos) keeps the
        noise-floor mean but shrinks its variance, so magnitude noise is no
        longer non-central chi; windowing the (linear) coil images first keeps
        it non-central chi with sigma scaled by the window RMS.

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

    if combine in ("ci4", "optimal") and ks.noise is None:
        raise ValueError(f"combine={combine!r} needs a noise scan (KSpace.noise is None); "
                         "convert with siemens_to_ismrmrd(noise=True) or use combine='sos'")
    if combine not in ("ci4", "optimal", "sos", "espirit", "none"):
        raise ValueError(f"Unknown combine option {combine!r}")

    # Crop to the recon matrix (x: readout oversampling; y: phase oversampling unless kept; z: slice oversampling).
    # MATLAB combines on the oversampled image and crops afterwards. The single-echo 'ci4' path (virtual-coil
    # combine) smooths in-plane with a kernel tied to the matrix size, so the crop is not interchangeable with it:
    # crop after the combine, except where the coil images are needed (no combine, ESPIRiT, per-coil window).
    info = encoding_info(ks.header) if ks.header is not None else None
    ny_enc = im.shape[-2]
    ry = ny_enc

    def _crop(x):
        if info is None:
            return x
        rx_, ry_, rz_ = info.recon_matrix
        ny_out = ry_ if crop_phase_oversampling else ny_enc
        return crop_to_size(x, (rz_ if x.shape[-3] > 1 else 1, ny_out, rx_))

    def _window(x):
        params = window_params if window_params is not None else [None, 0.5, 0.5]
        return ifftc(apply_window(fftc(x), window, params))

    if info is not None:
        ry = info.recon_matrix[1]
    crop_first = combine in ("none", "espirit") or (window is not None and window_before_combine)
    if crop_first:
        im = _crop(im)
    if window is not None and window_before_combine:
        im = _window(im)

    if combine == "sos":
        im = sos(im, coil_dim=coil_dim).to(im.dtype)
    elif combine == "optimal":
        invcov = torch.linalg.inv(imaging_noise_covariance(ks).to(im.device))
        im = optimal_combine(im, invcov, coil_dim=coil_dim)
    elif combine == "ci4":
        im = ci4_combine(
            im, imaging_noise_covariance(ks).to(im.device), coil_dim=coil_dim,
            echo_dim=ks.dim("contrast"), time_dim=ks.dim("repetition"), t_ave=t_ave)
    elif combine == "espirit":
        im = _espirit_combine(im, ks, espirit or {})

    if not crop_first:
        im = _crop(im)
    if window is not None and not window_before_combine:
        im = _window(im)

    if matrix is not None:
        mx, my, mz = matrix
        k = fftc(im, norm="forward")
        im = ifftc(resize_center(k, (mz, my, mx)), norm="forward")

    out_matrix = (im.shape[-1], im.shape[-2], im.shape[-3])
    kept_y = (not crop_phase_oversampling) and info is not None and ny_enc > ry
    slice_geometry = {}
    for s_, g in ks.geometry.items():
        if kept_y:
            g = g.with_fov((g.fov[0], g.fov[1] * ny_enc / ry, g.fov[2]), matrix=(g.matrix[0], ny_enc, g.matrix[2]))
        slice_geometry[s_] = g.with_matrix(out_matrix)
    dims = LOOP_DIMS + (("coil",) if combine == "none" else ()) + ("z", "y", "x")
    return Image(
        data=im, geometry=slice_geometry.get(0), dims=dims, header=ks.header,
        slice_geometry=slice_geometry, meta={"fft_norm": "ortho"},
    )


def imaging_noise_covariance(ks: KSpace) -> torch.Tensor:
    """
    Coil noise covariance of the imaging readout.

    The noise scan's covariance (`noise_covariance`), rescaled from its dwell
    time to the imaging dwell time: thermal noise variance is proportional to
    the bandwidth, ``1 / dwell``. Siemens noise-adjust scans often use another
    dwell (e.g. 5 us vs 1.6-20 us for imaging). Unscaled if either dwell is
    unknown.
    """
    if ks.noise is None:
        raise ValueError("KSpace has no noise scan")
    cov = noise_covariance(ks.noise)
    if ks.noise_sample_time_us and ks.sample_time_us:
        cov = cov * (float(ks.noise_sample_time_us) / float(ks.sample_time_us))
    return cov


def sort_slices(image: Image) -> Image:
    """
    Order the 'slice' axis by position along the slice normal (ascending).

    Siemens stores interleaved multi-slice data in acquisition order. Uses
    ``position . slice_dir`` of each slice's geometry; the permutation (new
    index -> original slice index) is stored in ``meta['slice_order']``.
    Assumes `slice_geometry` keys 0..n-1 match the slice axis (as returned by
    `reconstruct_cartesian`); other per-slice metadata is not permuted.
    """
    import numpy as np
    sd = image.dim("slice")
    n = image.data.shape[sd]
    proj = [float(np.dot(image.slice_geometry[i].position, image.slice_geometry[i].slice_dir)) for i in range(n)]
    order = [int(i) for i in np.argsort(proj, kind="stable")]
    data = image.data.index_select(sd, torch.tensor(order, device=image.data.device))
    geometry = {new: image.slice_geometry[old] for new, old in enumerate(order)}
    meta = {**image.meta, "slice_order": order}
    return Image(data=data, geometry=geometry.get(0), dims=image.dims, header=image.header,
                 slice_geometry=geometry, meta=meta)


def _espirit_combine(im: torch.Tensor, ks: KSpace, options: dict) -> torch.Tensor:
    """ESPIRiT maps from the first set/phase/contrast/repetition/average of each slice; SENSE combine of everything.

    `im` has the dims of `ks` (slice, set, phase, contrast, repetition, average, coil, z, y, x); the coil axis is
    removed, as for the other combine modes.
    """
    opts = dict(calib_size=24, kernel_size=6, threshold=1e-3, crop=0.8, n_maps=1)
    reg = float(options.get("reg", 0.0))
    opts.update({k: v for k, v in options.items() if k != "reg"})
    slice_dim = ks.dims.index("slice")
    out = []
    for sl in range(im.shape[slice_dim]):
        sub = im.select(slice_dim, sl)                    # (set, phase, contrast, repetition, average, coil, z, y, x)
        n_loop = sub.ndim - 4
        ref = sub[(0,) * n_loop]                          # (coil, z, y, x)
        k_ref = fftc(ref, dim=(-3, -2, -1), norm="ortho")
        maps, _ = espirit_maps(k_ref, **opts)             # (n_maps, coil, z, y, x)
        m = maps[0].reshape((1,) * n_loop + tuple(maps.shape[1:]))
        out.append(sense_combine(sub, m, coil_dim=n_loop, reg=reg))
    return torch.stack(out, dim=slice_dim)
