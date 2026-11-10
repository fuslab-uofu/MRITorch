"""
ESPIRiT coil sensitivity estimation.

Port of the calibration of BART's ``ecalib`` for fully sampled (Cartesian) calibration data, following
Uecker et al., "ESPIRiT: an eigenvalue approach to autocalibrating parallel MRI", Magn Reson Med 2014.
Steps:

1. A calibration matrix is built from all ``kernel`` sized patches of the central ``calib_size`` region of
   k-space (rows: patch positions; columns: coil x kernel position).
2. The eigenvectors of the calibration covariance (``A^H A``) with eigenvalues above ``threshold`` (relative to
   the largest) span the signal space.
3. In image space, each signal-space vector ``v_i`` gives, at every pixel x, the coil vector
   ``vhat_i(x) = sum_delta v_i(delta) exp(-i delta x)``; the pixel-wise matrix
   ``G(x) = (1 / prod(kernel)) sum_i vhat_i(x) vhat_i(x)^H`` has the coil sensitivity vector as an eigenvector
   with eigenvalue 1.
4. The complex conjugates of the ``n_maps`` dominant eigenvectors of G(x) are the sensitivity maps (unit norm over the coils, global phase
   referenced to the first principal component, see ``phase_reference``); pixels whose eigenvalue is below
   ``crop`` are set to zero.

The defaults are those of the MPME reconstruction (``ecalib -d0 -t1e-3 -r24 -k6 -m1``).

Arrays are ``(coil, *spatial)`` with 1 to 3 spatial dimensions (the trailing dims of the input); dimensions of
size 1 are left alone.
"""
from __future__ import annotations
import itertools
from typing import Optional, Sequence

import torch

from mritorch.fft import fftc, ifftc


def _spatial_dims(x: torch.Tensor, n_spatial: Optional[int]) -> list[int]:
    """Indices of the spatial dims of a ``(coil, *spatial)`` tensor."""
    n = x.ndim - 1 if n_spatial is None else n_spatial
    return list(range(x.ndim - n, x.ndim))


def _per_dim(value: int | Sequence[int], sizes: Sequence[int]) -> list[int]:
    v = [int(value)] * len(sizes) if isinstance(value, int) else [int(i) for i in value]
    return [min(a, n) for a, n in zip(v, sizes)]


def calibration_region(ksp: torch.Tensor, calib_size: int | Sequence[int]=24, n_spatial: Optional[int]=None) -> torch.Tensor:
    """Central ``calib_size`` block of ``(coil, *spatial)`` k-space (index ``N // 2`` is the center)."""
    dims = _spatial_dims(ksp, n_spatial)
    sizes = _per_dim(calib_size, [ksp.shape[d] for d in dims])
    out = ksp
    for d, m in zip(dims, sizes):
        n = out.shape[d]
        start = n // 2 - m // 2
        out = out.narrow(d, start, m)
    return out


def calibration_matrix(calib: torch.Tensor, kernel_size: int | Sequence[int]=6, n_spatial: Optional[int]=None) -> torch.Tensor:
    """
    Calibration matrix of a ``(coil, *spatial)`` calibration region.

    Returns ``(n_patches, coil * prod(kernel))``; the column order is (coil, kernel position in C order).
    """
    dims = _spatial_dims(calib, n_spatial)
    sizes = [calib.shape[d] for d in dims]
    kernel = _per_dim(kernel_size, sizes)
    n_pos = [n - k + 1 for n, k in zip(sizes, kernel)]
    cols = []
    for off in itertools.product(*[range(k) for k in kernel]):
        block = calib
        for d, o, n in zip(dims, off, n_pos):
            block = block.narrow(d, o, n)
        cols.append(block.reshape(calib.shape[0], -1))            # (coil, n_patches)
    mat = torch.stack(cols, dim=1)                                # (coil, n_offsets, n_patches)
    return mat.reshape(-1, mat.shape[-1]).T                        # (n_patches, coil * n_offsets)


def signal_space(calib: torch.Tensor, kernel_size: int | Sequence[int]=6, threshold: float=1e-3,
                 threshold_mode: str="eigenvalue", n_spatial: Optional[int]=None):
    """
    Signal-space vectors of the calibration matrix.

    ``threshold_mode`` 'eigenvalue' keeps eigenvectors of ``A^H A`` whose eigenvalue exceeds ``threshold`` times the
    largest; 'singular' compares the singular values (square roots) instead.

    Returns ``(vectors, kernel)``: ``vectors`` is ``(n_keep, coil, *kernel)``.
    """
    dims = _spatial_dims(calib, n_spatial)
    kernel = _per_dim(kernel_size, [calib.shape[d] for d in dims])
    A = calibration_matrix(calib, kernel, n_spatial)
    cov = A.conj().T @ A
    evals, evecs = torch.linalg.eigh(cov)                          # ascending
    ref = evals[-1]
    keep = evals > (threshold * ref if threshold_mode == "eigenvalue" else (threshold ** 2) * ref)
    vec = evecs[:, keep].T                                         # (n_keep, coil * n_offsets)
    return vec.reshape(vec.shape[0], calib.shape[0], *kernel), kernel


def espirit_maps(
        ksp: torch.Tensor,
        calib_size: int | Sequence[int]=24,
        kernel_size: int | Sequence[int]=6,
        threshold: float=1e-3,
        crop: float=0.8,
        n_maps: int=1,
        threshold_mode: str="eigenvalue",
        phase_reference: str="principal",
        n_spatial: Optional[int]=None,
        ) -> tuple[torch.Tensor, torch.Tensor]:
    """
    ESPIRiT sensitivity maps of fully sampled, centered k-space.

    Parameters
    ----------
    ksp : torch.Tensor
        ``(coil, *spatial)`` centered k-space.
    calib_size, kernel_size : int or sequence of int
        Calibration region and kernel size (BART ``-r`` and ``-k``).
    threshold : float
        Signal-space threshold (BART ``-t``), see `signal_space`.
    crop : float
        Maps are zero where the eigenvalue is below this value (BART ``-c``).
    n_maps : int
        Number of maps (BART ``-m``).
    phase_reference : {'principal', 'dominant', 'none'}
        Global phase convention of each pixel's vector. 'principal' (default, BART) rotates each vector so that
        its projection onto the dominant eigenvector of the map covariance over the object is real; 'dominant'
        makes the largest-magnitude coil real and positive; 'none' keeps the eigenvector phase (arbitrary per
        pixel). Magnitudes and the SENSE combination's magnitude do not depend on this choice.
    n_spatial : int, optional
        Number of trailing spatial dims; defaults to all dims after the coil axis.

    Returns
    -------
    maps : torch.Tensor
        ``(n_maps, coil, *spatial)`` unit-norm sensitivity maps.
    eigenvalues : torch.Tensor
        ``(n_maps, *spatial)`` eigenvalues of G(x).
    """
    dims = _spatial_dims(ksp, n_spatial)
    shape = [ksp.shape[d] for d in dims]
    calib = calibration_region(ksp, calib_size, n_spatial)
    vecs, kernel = signal_space(calib, kernel_size, threshold, threshold_mode, n_spatial)
    n_keep, n_coil = vecs.shape[0], vecs.shape[1]
    n_pix = int(torch.tensor(shape).prod())
    k_vol = float(torch.tensor(kernel).prod())

    # kernels to image space: vhat_i(x) = sum_delta v_i(delta) exp(-i delta x) = fft(zero-padded kernel). The sign
    # matches the transform of the calibration k-space to the image (checked against BART ecalib)
    pad = torch.zeros(n_keep, n_coil, *shape, dtype=vecs.dtype, device=vecs.device)
    pad[(slice(None), slice(None)) + tuple(slice(0, k) for k in kernel)] = vecs
    sp_axes = tuple(range(2, 2 + len(shape)))
    vhat = torch.fft.fftn(pad, dim=sp_axes)                       # unnormalized forward transform
    vhat = torch.fft.fftshift(vhat, dim=sp_axes)                  # image origin at N // 2, as ifftc of centered k-space
    vhat = vhat.reshape(n_keep, n_coil, n_pix).permute(2, 1, 0)   # (pix, coil, keep)

    G = (vhat @ vhat.conj().transpose(-1, -2)) / k_vol            # (pix, coil, coil)
    evals, evecs = torch.linalg.eigh(G)                            # ascending
    out_maps, out_ev = [], []
    for m in range(n_maps):
        ev = evals[:, -1 - m]
        vec = evecs[:, :, -1 - m].conj()                            # (pix, coil), unit norm; the conjugate of the G
                                                                    # eigenvector is the sensitivity (checked vs BART)
        inside = ev >= crop
        if phase_reference == "principal":
            M = vec[inside]
            _, w_all = torch.linalg.eigh(M.conj().T @ M)
            w = w_all[:, -1]
            w = w * (w.sum().conj() / w.sum().abs().clamp_min(1e-30))   # eigenvector phase is arbitrary: pin it (sum of components real, >= 0)
            proj = vec @ w
            vec = vec * (proj.conj() / proj.abs().clamp_min(1e-30))[:, None]
        elif phase_reference == "dominant":
            idx = vec.abs().argmax(dim=1, keepdim=True)
            ph = torch.gather(vec, 1, idx)
            vec = vec * (ph.conj() / ph.abs().clamp_min(1e-30))
        elif phase_reference != "none":
            raise ValueError(f"unknown phase_reference {phase_reference!r}")
        vec = torch.where(inside[:, None], vec, torch.zeros_like(vec))
        out_maps.append(vec.T.reshape(n_coil, *shape))
        out_ev.append(ev.reshape(*shape))
    return torch.stack(out_maps), torch.stack(out_ev)
