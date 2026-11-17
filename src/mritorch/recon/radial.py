"""
Radial (golden-angle) 2D reconstruction of MPME readout planes: NUFFT, ESPIRiT calibration from a low-resolution
image and a regularized SENSE (CG) solution, following `FUSLab_SA/recon_mpme.m` and the BART calls it makes:

    nufft -a -d40:40:1 -t trajOS ksOS        (low-resolution calibration image, density-compensated)
    ecalib -d0 -t1e-3 -r24 -k6 -m1           (ESPIRiT maps, `mritorch.coils.espirit_maps`)
    pics -d0 -S -r5e-4 -t traj ks sens        (L2-regularized SENSE: (A^H A + lambda I) x = A^H y)

Trajectories are in units of pixel^-1 times the matrix size (BART convention): sample ``rho`` in
``[-N/2, N/2)`` along spokes rotated by ``angle_increment`` per view. Operators are normalized with ``1/sqrt(prod(shape))``
(as the unitary FFT), so that a fully sampled Cartesian trajectory reproduces `fftc` with ``norm='ortho'``.
"""
from __future__ import annotations
import math
from typing import Callable, Optional, Sequence

import numpy as np
import torch

from mritorch.coils.espirit import espirit_maps
from mritorch.fft import fftc, ifftc
from mritorch.utils.array import resize_center, pad_to_size


def radial_trajectory(n_samp: int, n_view: int, angle_increment: float, init_angle: float=0.0) -> torch.Tensor:
    """``(2, n_samp, n_view)`` trajectory in pixel^-1 * N units; ``rho = -n_samp/2 ... n_samp/2 - 1`` along each spoke."""
    rho = torch.arange(-(n_samp // 2), n_samp - n_samp // 2, dtype=torch.float64)
    ang = (torch.arange(n_view, dtype=torch.float64) * angle_increment + init_angle) * math.pi / 180
    return torch.stack([rho[:, None] * torch.cos(ang)[None], rho[:, None] * torch.sin(ang)[None]])


class Nufft2d:
    """
    2D non-uniform FFT between images ``(..., N1, N2)`` (origin at index ``N // 2``) and samples ``(..., n_points)``.

    ``forward`` is ``sum_n x_n exp(-2 pi i k n / N)`` and ``adjoint`` its conjugate transpose, both times
    ``1 / sqrt(N1 N2)``. ``backend='finufft'`` is the production (CPU, multi-threaded) implementation;
    ``'direct'`` evaluates the sums densely (tests, small problems).
    """

    def __init__(self, traj: torch.Tensor, shape: Sequence[int], backend: str="finufft", eps: float=1e-6,
                 nthreads: int=0):
        self.shape = tuple(int(n) for n in shape)
        self.traj = traj.reshape(2, -1).to(torch.float64)
        self.n_points = self.traj.shape[1]
        self.backend = backend
        self.eps = eps
        self.nthreads = nthreads
        self.scale = 1.0 / math.sqrt(self.shape[0] * self.shape[1])
        self._plans: dict = {}
        # phase argument in radians: 2 pi k / N
        self.omega = torch.stack([2 * math.pi * self.traj[0] / self.shape[0], 2 * math.pi * self.traj[1] / self.shape[1]])

    def _plan(self, kind: int, n_trans: int):
        import finufft
        key = (kind, n_trans)
        if key not in self._plans:
            p = finufft.Plan(kind, self.shape, n_trans=n_trans, eps=self.eps, isign=-1 if kind == 2 else 1,
                             dtype="complex128", nthreads=self.nthreads)
            p.setpts(self.omega[0].numpy(), self.omega[1].numpy())
            self._plans[key] = p
        return self._plans[key]

    def _wrap_ok(self) -> bool:
        return bool(torch.all(self.omega.abs() <= math.pi + 1e-9))

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        lead = img.shape[:-2]
        x = img.reshape(-1, *self.shape)
        if self.backend == "direct":
            n1 = torch.arange(self.shape[0], dtype=torch.float64) - self.shape[0] // 2
            n2 = torch.arange(self.shape[1], dtype=torch.float64) - self.shape[1] // 2
            e1 = torch.exp(-1j * self.omega[0][:, None] * n1[None])           # (pts, N1)
            e2 = torch.exp(-1j * self.omega[1][:, None] * n2[None])           # (pts, N2)
            out = torch.einsum("bnm,pn,pm->bp", x.to(torch.complex128), e1, e2)
        else:
            plan = self._plan(2, x.shape[0])
            out = torch.from_numpy(np.asarray(plan.execute(np.ascontiguousarray(x.numpy().astype(np.complex128)))))
            out = out.reshape(x.shape[0], self.n_points)
        return (out * self.scale).to(img.dtype).reshape(*lead, self.n_points)

    def adjoint(self, ks: torch.Tensor) -> torch.Tensor:
        lead = ks.shape[:-1]
        y = ks.reshape(-1, self.n_points)
        if self.backend == "direct":
            n1 = torch.arange(self.shape[0], dtype=torch.float64) - self.shape[0] // 2
            n2 = torch.arange(self.shape[1], dtype=torch.float64) - self.shape[1] // 2
            e1 = torch.exp(1j * self.omega[0][:, None] * n1[None])
            e2 = torch.exp(1j * self.omega[1][:, None] * n2[None])
            out = torch.einsum("bp,pn,pm->bnm", y.to(torch.complex128), e1, e2)
        else:
            plan = self._plan(1, y.shape[0])
            out = torch.from_numpy(np.asarray(plan.execute(np.ascontiguousarray(y.numpy().astype(np.complex128)))))
            out = out.reshape(y.shape[0], *self.shape)
        return (out * self.scale).to(ks.dtype).reshape(*lead, *self.shape)


def calibration_maps(
        k_calib: torch.Tensor,
        angle_increment: float,
        shape: Sequence[int],
        calib_reduction: int=4,
        os_factor: int=2,
        espirit: Optional[dict]=None,
        backend: str="finufft",
        ) -> torch.Tensor:
    """
    Sensitivity maps from the radial k-space of the calibration pathway (`recon_mpme.m`, ``bart_radial_ecalib_pics``).

    Parameters
    ----------
    k_calib : torch.Tensor
        ``(n_samp, n_view, coil)`` k-space samples along the spokes (one readout plane, calibration pathway, first echo).
    shape : (int, int)
        Image matrix.
    calib_reduction : int
        The calibration image has ``ceil(N / calib_reduction)`` pixels per side.
    os_factor : int
        Oversampling of the samples along each spoke (Fourier interpolation) before gridding.

    Returns
    -------
    torch.Tensor
        ``(coil, N1, N2)`` unit-norm ESPIRiT maps.
    """
    n_samp, n_view, n_coil = k_calib.shape
    # Fourier interpolation of the samples along each spoke (zero-fill in the projection domain)
    k = k_calib.movedim(0, -1)                                            # (view, coil, samp)
    k = fftc(k, dim=-1, norm="forward")
    k = resize_center(k, (n_samp * os_factor,))
    k = ifftc(k, dim=-1, norm="forward").movedim(-1, 0)                   # (samp*os, view, coil)
    traj = radial_trajectory(n_samp * os_factor, n_view, angle_increment) / os_factor
    k = k * (torch.sqrt((traj ** 2).sum(0)) / (n_samp / 2))[..., None]          # ramp density compensation
    n_cal = tuple(math.ceil(n / calib_reduction) for n in shape)
    # samples outside the low-resolution grid are discarded (BART nufft does not wrap them)
    inside = (traj[0].abs() < n_cal[0] / 2) & (traj[1].abs() < n_cal[1] / 2)
    t = traj[:, inside]                                                   # (2, n_in)
    nu = Nufft2d(t, n_cal, backend=backend)
    img = nu.adjoint(k[inside].T.contiguous())                            # (coil, n1, n2)
    k_lr = fftc(img, dim=(-2, -1), norm="ortho")
    k_zfi = pad_to_size(k_lr, tuple(shape))
    opts = dict(calib_size=24, kernel_size=6, threshold=1e-3, crop=0.8, n_maps=1)
    opts.update(espirit or {})
    maps, _ = espirit_maps(k_zfi, **opts)
    return maps[0]


def sense_cg(
        ks: torch.Tensor,
        nufft: Nufft2d,
        maps: torch.Tensor,
        reg: float=5e-4,
        n_iter: int=50,
        tol: float=0.0,
        ) -> torch.Tensor:
    """
    Solve ``(A^H A + reg I) x = A^H y`` by conjugate gradient, ``A = NUFFT * (coil maps)``.

    Parameters
    ----------
    ks : torch.Tensor
        ``(coil, n_points, *batch)``-shaped samples given as ``(*batch, coil, n_points)``; the solution is independent per
        batch entry.
    maps : torch.Tensor
        ``(coil, N1, N2)``.

    Returns
    -------
    torch.Tensor
        ``(*batch, N1, N2)``.
    """
    c = maps

    def A(x):                                   # (*b, N1, N2) -> (*b, coil, pts)
        return nufft.forward(x[..., None, :, :] * c)

    def AH(y):                                  # (*b, coil, pts) -> (*b, N1, N2)
        return (nufft.adjoint(y) * c.conj()).sum(-3)

    def N(x):
        return AH(A(x)) + reg * x

    b = AH(ks)
    x = torch.zeros_like(b)
    r = b.clone()
    p = r.clone()
    rs = _vdot(r, r)
    rs0 = rs.clone()
    for _ in range(n_iter):
        Ap = N(p)
        alpha = rs / _vdot(p, Ap).clamp_min(_TINY)
        x = x + alpha[(...,) + (None,) * 2] * p
        r = r - alpha[(...,) + (None,) * 2] * Ap
        rs_new = _vdot(r, r)
        if tol > 0 and bool(torch.all(rs_new <= (tol ** 2) * rs0)):
            break
        beta = rs_new / rs.clamp_min(_TINY)
        p = r + beta[(...,) + (None,) * 2] * p
        rs = rs_new
    return x


_TINY = torch.finfo(torch.float32).tiny


def _vdot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Real inner product per batch entry (sum over the last two dims)."""
    return (a.conj() * b).sum((-2, -1)).real


def reconstruct_mpme(
        mk,
        shape: Optional[Sequence[int]]=None,
        reg: float=5e-4,
        n_iter: int=50,
        calib_reduction: int=4,
        espirit: Optional[dict]=None,
        planes: Optional[Sequence[int]]=None,
        backend: str="finufft",
        nthreads: int=0,
        eps: float=1e-6,
        progress: Optional[Callable[[int, int], None]]=None,
        return_maps: bool=False,
        ):
    """
    Reconstruct every readout plane of an `mritorch.io.mpme.MpmeKSpace` (calibration from the p = 0 pathway of the
    first echo window, solution for all pathways and echoes at once).

    Returns
    -------
    torch.Tensor
        ``(x, N1, N2, pathway, echo)`` complex64 (the layout of `im_MPME_*.mat`); also the maps ``(x, coil, N1, N2)``
        if ``return_maps``.
    """
    data = mk.data                                                        # (x, samp, view, coil, pwy, echo)
    n_x, n_samp, n_view, n_coil, n_pwy, n_eco = data.shape
    shape = tuple(shape) if shape is not None else (n_samp, n_samp)
    traj = radial_trajectory(n_samp, n_view, mk.angle_increment)
    nu = Nufft2d(traj, shape, backend=backend, eps=eps, nthreads=nthreads)
    planes = list(range(n_x)) if planes is None else list(planes)
    out = torch.zeros(len(planes), *shape, n_pwy, n_eco, dtype=torch.complex64)
    all_maps = torch.zeros(len(planes), n_coil, *shape, dtype=torch.complex64) if return_maps else None
    for i, x in enumerate(planes):
        kr = data[x]                                                      # (samp, view, coil, pwy, echo)
        maps = calibration_maps(kr[..., mk.calib_index, 0], mk.angle_increment, shape, calib_reduction,
                                espirit=espirit, backend=backend)
        y = kr.permute(4, 3, 2, 0, 1)                                    # (echo, pwy, coil, samp, view)
        y = y.reshape(n_eco * n_pwy, n_coil, n_samp * n_view)            # flatten (samp, view): matches traj.reshape(2, -1)
        x_img = sense_cg(y, nu, maps, reg=reg, n_iter=n_iter)            # (echo*pwy, N1, N2)
        out[i] = x_img.reshape(n_eco, n_pwy, *shape).permute(2, 3, 1, 0).to(torch.complex64)
        if all_maps is not None:
            all_maps[i] = maps
        if progress is not None:
            progress(i + 1, len(planes))
    return (out, all_maps) if return_maps else out
