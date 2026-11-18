"""
Radial (golden-angle) 2D reconstruction of MPME readout planes: NUFFT, ESPIRiT calibration from a low-resolution
image and a regularized SENSE (CG) solution, following `FUSLab_SA/recon_mpme.m` and the BART calls it makes:

    nufft -a -d40:40:1 -t trajOS ksOS        (low-resolution calibration image, density-compensated)
    ecalib -d0 -t1e-3 -r24 -k6 -m1           (ESPIRiT maps, `mritorch.coils.espirit_maps`)
    pics -d0 -S -r5e-4 -t traj ks sens        (L2-regularized SENSE: (A^H A + lambda I) x = A^H y)

Trajectories are in units of pixel^-1 times the matrix size (BART convention): sample ``rho`` in
``[-N/2, N/2)`` along spokes rotated by ``angle_increment`` per view. Operators are normalized with ``1/sqrt(prod(shape))``
(as the unitary FFT), so that a fully sampled Cartesian trajectory reproduces `fftc` with ``norm='ortho'``.

The NUFFT (`Nufft2d`) uses `torchkbnufft` by default, so the operator itself is differentiable and runs on the
device of its input (GPU use is untested). `reconstruct_mpme` still assembles its outputs on the CPU.
"""
from __future__ import annotations
import math
from typing import Callable, Optional, Sequence

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
    ``1 / sqrt(N1 N2)``.

    Backends:

    - ``'kb'`` (default): Kaiser-Bessel gridding with `torchkbnufft` and a precomputed sparse interpolation matrix,
      built once per device on first use. Differentiable (autograd) with respect to the image or samples; runs on
      the device of its input, in the input's precision (complex64 or complex128). Relative error against
      ``'direct'`` is about 1e-5 with the defaults ``numpoints=6, oversamp=2.0``. CPU threads follow
      `torch.set_num_threads`. GPU use is untested.
    - ``'direct'``: evaluates the sums densely in complex128 (tests, small problems). It keeps two complex128
      phase tables of ``n_points * (N1 + N2)`` entries per direction for the object's lifetime.

    Outputs are complex in the precision of the input: complex64 for float32/complex64 input, complex128 for
    float64/complex128 input.

    Parameters
    ----------
    traj : torch.Tensor
        ``(2, ...)`` sample locations in pixel^-1 * N units (see `radial_trajectory`); flattened to ``(2, n_points)``.
    shape : (int, int)
        Image matrix ``(N1, N2)``.
    backend : {'kb', 'direct'}
    numpoints : int
        Kaiser-Bessel kernel width in grid points (``'kb'`` only).
    oversamp : float
        Grid oversampling factor (``'kb'`` only).
    """

    def __init__(self, traj: torch.Tensor, shape: Sequence[int], backend: str="kb", numpoints: int=6,
                 oversamp: float=2.0):
        if backend not in ("kb", "direct"):
            raise ValueError(f"backend must be 'kb' or 'direct', got {backend!r}")
        self.shape = tuple(int(n) for n in shape)
        self.traj = traj.reshape(2, -1).to(torch.float64)
        self.n_points = self.traj.shape[1]
        self.backend = backend
        self.numpoints = int(numpoints)
        self.oversamp = float(oversamp)
        self.grid_size = tuple(int(math.ceil(self.oversamp * n)) for n in self.shape)
        self.scale = 1.0 / math.sqrt(self.shape[0] * self.shape[1])
        # phase argument in radians: 2 pi k / N
        self.omega = torch.stack([2 * math.pi * self.traj[0] / self.shape[0], 2 * math.pi * self.traj[1] / self.shape[1]])
        self._kb_cache: dict = {}
        self._direct_cache: dict = {}
        # geometry of a calibration operator, set by `calibration_nufft` and checked by `calibration_maps`
        self.calibration_key: Optional[tuple] = None

    def _kb(self, dtype: torch.dtype, device: torch.device):
        """``(forward op, adjoint op, omega, interp_mats)`` for a complex dtype and device, built on first use."""
        key = (dtype, device)
        if key not in self._kb_cache:
            import torchkbnufft as tkbn
            real = torch.float64 if dtype == torch.complex128 else torch.float32
            kw = dict(im_size=self.shape, grid_size=self.grid_size, numpoints=self.numpoints)
            fwd = tkbn.KbNufft(**kw, dtype=real).to(device)
            adj = tkbn.KbNufftAdjoint(**kw, dtype=real).to(device)
            omega = self.omega.to(device=device, dtype=real)
            mats = tkbn.calc_tensor_spmatrix(omega, **kw)
            self._kb_cache[key] = (fwd, adj, omega, mats)
        return self._kb_cache[key]

    # torchkbnufft's norm="ortho" divides by sqrt(prod(grid_size)); rescale to the 1/sqrt(N1 N2) convention
    def _kb_scale(self) -> float:
        return math.sqrt(math.prod(self.grid_size) / math.prod(self.shape))

    def _direct_phases(self, sign: int):
        """``exp(sign i omega n)`` tables ``(pts, N1)`` and ``(pts, N2)``, computed once per sign."""
        if sign not in self._direct_cache:
            n1 = torch.arange(self.shape[0], dtype=torch.float64) - self.shape[0] // 2
            n2 = torch.arange(self.shape[1], dtype=torch.float64) - self.shape[1] // 2
            e1 = torch.exp(sign * 1j * self.omega[0][:, None] * n1[None])
            e2 = torch.exp(sign * 1j * self.omega[1][:, None] * n2[None])
            self._direct_cache[sign] = (e1, e2)
        return self._direct_cache[sign]

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        lead = img.shape[:-2]
        x = img.reshape(-1, *self.shape)
        if self.backend == "direct":
            e1, e2 = self._direct_phases(-1)
            out = torch.einsum("bnm,pn,pm->bp", x.to(torch.complex128), e1, e2) * self.scale
            return out.to(_as_complex(x).dtype).reshape(*lead, self.n_points)
        x = _as_complex(x)
        fwd, _, omega, mats = self._kb(x.dtype, x.device)
        out = fwd(x[None], omega, interp_mats=mats, norm="ortho")[0] * self._kb_scale()
        return out.reshape(*lead, self.n_points)

    def adjoint(self, ks: torch.Tensor) -> torch.Tensor:
        lead = ks.shape[:-1]
        y = ks.reshape(-1, self.n_points)
        if self.backend == "direct":
            e1, e2 = self._direct_phases(1)
            out = torch.einsum("bp,pn,pm->bnm", y.to(torch.complex128), e1, e2) * self.scale
            return out.to(_as_complex(y).dtype).reshape(*lead, *self.shape)
        y = _as_complex(y)
        _, adj, omega, mats = self._kb(y.dtype, y.device)
        out = adj(y[None], omega, interp_mats=mats, norm="ortho")[0] * self._kb_scale()
        return out.reshape(*lead, *self.shape)


def _as_complex(x: torch.Tensor) -> torch.Tensor:
    """Complex view of ``x`` in matching precision (complex128 for float64/complex128, else complex64)."""
    if x.dtype in (torch.complex64, torch.complex128):
        return x
    return x.to(torch.complex128 if x.dtype == torch.float64 else torch.complex64)


def _calibration_geometry(n_samp: int, n_view: int, angle_increment: float, shape: Sequence[int],
                          calib_reduction: int, os_factor: int):
    """Oversampled spoke trajectory ``(2, n_samp*os, n_view)``, mask of samples inside the calibration grid, grid."""
    traj = radial_trajectory(n_samp * os_factor, n_view, angle_increment) / os_factor
    n_cal = tuple(math.ceil(n / calib_reduction) for n in shape)
    # samples outside the low-resolution grid are discarded (BART nufft does not wrap them)
    inside = (traj[0].abs() < n_cal[0] / 2) & (traj[1].abs() < n_cal[1] / 2)
    return traj, inside, n_cal


def _calibration_key(n_samp, n_view, angle_increment, shape, calib_reduction, os_factor) -> tuple:
    return (int(n_samp), int(n_view), float(angle_increment), tuple(int(n) for n in shape), int(calib_reduction),
            int(os_factor))


def calibration_nufft(
        n_samp: int,
        n_view: int,
        angle_increment: float,
        shape: Sequence[int],
        calib_reduction: int=4,
        os_factor: int=2,
        backend: str="kb",
        numpoints: int=6,
        oversamp: float=2.0,
        ) -> Nufft2d:
    """
    The low-resolution NUFFT used by `calibration_maps`, for reuse across readout planes that share a trajectory.
    Arguments are those of `calibration_maps` (and ``n_samp, n_view`` of its ``k_calib``).
    """
    traj, inside, n_cal = _calibration_geometry(n_samp, n_view, angle_increment, shape, calib_reduction, os_factor)
    op = Nufft2d(traj[:, inside], n_cal, backend=backend, numpoints=numpoints, oversamp=oversamp)
    op.calibration_key = _calibration_key(n_samp, n_view, angle_increment, shape, calib_reduction, os_factor)
    return op


def calibration_maps(
        k_calib: torch.Tensor,
        angle_increment: float,
        shape: Sequence[int],
        calib_reduction: int=4,
        os_factor: int=2,
        espirit: Optional[dict]=None,
        backend: Optional[str]=None,
        numpoints: Optional[int]=None,
        oversamp: Optional[float]=None,
        nufft: Optional[Nufft2d]=None,
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
    backend, numpoints, oversamp
        `Nufft2d` settings of the calibration NUFFT; ``None`` means the `calibration_nufft` defaults
        (``'kb'``, 6, 2.0). Must not be given together with ``nufft``.
    nufft : Nufft2d, optional
        Prebuilt operator from `calibration_nufft`, called with the same geometry arguments, to avoid rebuilding it
        for every plane. Its own NUFFT settings are used.

    Returns
    -------
    torch.Tensor
        ``(coil, N1, N2)`` unit-norm ESPIRiT maps.
    """
    n_samp, n_view, n_coil = k_calib.shape
    traj, inside, n_cal = _calibration_geometry(n_samp, n_view, angle_increment, shape, calib_reduction, os_factor)
    settings = {k: v for k, v in dict(backend=backend, numpoints=numpoints, oversamp=oversamp).items()
                if v is not None}
    if nufft is None:
        nufft = calibration_nufft(n_samp, n_view, angle_increment, shape, calib_reduction, os_factor, **settings)
    else:
        if settings:
            raise ValueError(f"pass either nufft or NUFFT settings, not both (got {sorted(settings)})")
        key = _calibration_key(n_samp, n_view, angle_increment, shape, calib_reduction, os_factor)
        if getattr(nufft, "calibration_key", None) != key:
            raise ValueError("nufft does not match this calibration geometry; build it with calibration_nufft and "
                             "the same n_samp, n_view, angle_increment, shape, calib_reduction and os_factor")
    # Fourier interpolation of the samples along each spoke (zero-fill in the projection domain)
    k = k_calib.movedim(0, -1)                                            # (view, coil, samp)
    k = fftc(k, dim=-1, norm="forward")
    k = resize_center(k, (n_samp * os_factor,))
    k = ifftc(k, dim=-1, norm="forward").movedim(-1, 0)                   # (samp*os, view, coil)
    ramp = (torch.sqrt((traj ** 2).sum(0)) / (n_samp / 2)).to(k.real.dtype)  # ramp density compensation, in k's precision
    k = k * ramp[..., None]
    img = nufft.adjoint(k[inside].T.contiguous())                         # (coil, n1, n2)
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
        backend: str="kb",
        numpoints: int=6,
        oversamp: float=2.0,
        progress: Optional[Callable[[int, int], None]]=None,
        return_maps: bool=False,
        ):
    """
    Reconstruct every readout plane of an `mritorch.io.mpme.MpmeKSpace` (calibration from the p = 0 pathway of the
    first echo window, solution for all pathways and echoes at once). ``backend``, ``numpoints`` and ``oversamp``
    configure both NUFFTs (see `Nufft2d`); each is built once and reused for every plane.

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
    nu = Nufft2d(traj, shape, backend=backend, numpoints=numpoints, oversamp=oversamp)
    nu_cal = calibration_nufft(n_samp, n_view, mk.angle_increment, shape, calib_reduction,
                               backend=backend, numpoints=numpoints, oversamp=oversamp)
    planes = list(range(n_x)) if planes is None else list(planes)
    out = torch.zeros(len(planes), *shape, n_pwy, n_eco, dtype=torch.complex64)
    all_maps = torch.zeros(len(planes), n_coil, *shape, dtype=torch.complex64) if return_maps else None
    for i, x in enumerate(planes):
        kr = data[x]                                                      # (samp, view, coil, pwy, echo)
        maps = calibration_maps(kr[..., mk.calib_index, 0], mk.angle_increment, shape, calib_reduction,
                                espirit=espirit, nufft=nu_cal)
        y = kr.permute(4, 3, 2, 0, 1)                                    # (echo, pwy, coil, samp, view)
        y = y.reshape(n_eco * n_pwy, n_coil, n_samp * n_view)            # flatten (samp, view): matches traj.reshape(2, -1)
        x_img = sense_cg(y, nu, maps, reg=reg, n_iter=n_iter)            # (echo*pwy, N1, N2)
        out[i] = x_img.reshape(n_eco, n_pwy, *shape).permute(2, 3, 1, 0).to(torch.complex64)
        if all_maps is not None:
            all_maps[i] = maps
        if progress is not None:
            progress(i + 1, len(planes))
    return (out, all_maps) if return_maps else out
