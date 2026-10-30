"""
Coil combination.

Images are ``(..., coil, z, y, x)`` by default (``coil_dim=-4``). Ports of
the combine modes in the legacy MATLAB `CreateImg4`/`coil_combine_ci4`:

- `sos`: root sum of squares (mode 1).
- `optimal_combine`: noise-weighted combine against a reference image (mode 3
  building block).
- `virtual_coil_combine`: phase-preserving combine for single-frame data
  (`vcCombinePH`, mode 3 fallback).
- `ci4_combine`: mode 3 dispatch with the same reference choices as
  `CreateImg4`.
"""
from __future__ import annotations
from typing import Optional

import torch

from mritorch.filters import hann


def sos(im: torch.Tensor, coil_dim: int=-4) -> torch.Tensor:
    """Root sum of squares over `coil_dim`."""
    return torch.sqrt(torch.sum(torch.abs(im) ** 2, dim=coil_dim))


def _quad(a: torch.Tensor, invcov: torch.Tensor, b: torch.Tensor, coil_dim: int) -> torch.Tensor:
    """Per-voxel ``a^H Psi^-1 b`` contracted over the coil axis."""
    a = a.movedim(coil_dim, -1)
    b = b.movedim(coil_dim, -1)
    invcov = invcov.to(dtype=b.dtype, device=b.device)
    return torch.einsum("...j,jk,...k->...", a.conj(), invcov, b)


def optimal_combine(
        im: torch.Tensor,
        invcov: torch.Tensor,
        ref: Optional[torch.Tensor]=None,
        phase_ref: Optional[torch.Tensor]=None,
        coil_dim: int=-4,
        ) -> torch.Tensor:
    """
    Noise-optimal coil combine against reference images.

    Magnitude is ``sqrt(|ref^H Psi^-1 im|)`` and phase is
    ``angle(phase_ref^H Psi^-1 im)``, both summed over coils. With ``ref = im``
    the magnitude is the noise-weighted root sum of squares.

    Parameters
    ----------
    im : torch.Tensor
        Complex coil images.
    invcov : torch.Tensor
        Inverse noise covariance ``Psi^-1``, shape ``(coil, coil)``.
    ref : torch.Tensor, optional
        Reference coil images for the magnitude, broadcastable to `im`.
        Defaults to `im`.
    phase_ref : torch.Tensor, optional
        Reference coil images for the phase, broadcastable to `im`.
        Defaults to `ref`.
    coil_dim : int, optional
        Coil axis. Defaults to -4.

    Returns
    -------
    torch.Tensor
        Combined complex image with the coil axis removed.
    """
    if ref is None:
        ref = im
    if phase_ref is None:
        phase_ref = ref
    ref, im_b = torch.broadcast_tensors(ref, im)
    mag = torch.sqrt(torch.abs(_quad(ref, invcov, im_b, coil_dim)))
    phase_ref, im_b = torch.broadcast_tensors(phase_ref, im)
    phase = torch.angle(_quad(phase_ref, invcov, im_b, coil_dim))
    return torch.polar(mag, phase)


def _dlpsmooth(x: torch.Tensor, half_width: int) -> torch.Tensor:
    """Low-pass filter over the last two dims (port of `dlpsmooth`, type 2)."""
    ny, nx = x.shape[-2:]
    width = 1 + 2 * half_width
    w1 = hann(width, dtype=x.real.dtype, device=x.device)
    kernel = torch.zeros(ny, nx, dtype=x.real.dtype, device=x.device)
    y0, x0 = max((ny - width) // 2, 0), max((nx - width) // 2, 0)
    wy, wx = w1[:ny - y0], w1[:nx - x0]
    kernel[y0:y0 + wy.numel(), x0:x0 + wx.numel()] = wy[:, None] * wx[None, :]
    kernel = torch.fft.fftshift(kernel, dim=(-2, -1))
    return torch.fft.ifft2(torch.fft.fft2(x) * kernel)


def virtual_coil_combine(
        im: torch.Tensor,
        covariance: torch.Tensor,
        coil_dim: int=-4,
        time_dim: Optional[int]=None,
        ) -> torch.Tensor:
    """
    Phase-preserving optimal combine using a virtual reference coil.

    Port of Dennis Parker's `vcCombinePH`, used by `CreateImg4` mode 3 for
    single-echo, single-dynamic data. Smoothing acts on the in-plane
    ``(y, x)`` axes.

    Parameters
    ----------
    im : torch.Tensor
        Complex coil images ``(..., coil, z, y, x)``.
    covariance : torch.Tensor
        Noise covariance ``Psi`` (not its inverse), shape ``(coil, coil)``.
    coil_dim : int, optional
        Coil axis. Defaults to -4.
    time_dim : int, optional
        Dynamic axis averaged to build the reference. If None, no averaging.

    Returns
    -------
    torch.Tensor
        Combined complex image with the coil axis removed.
    """
    nx = im.shape[-1]
    # Remove smooth coil phase, then build a magnitude-weighted reference
    smooth = _dlpsmooth(im, nx // 24)
    cor = im * torch.exp(-1j * torch.angle(smooth))
    refcoil = cor.mean(dim=time_dim, keepdim=True) if time_dim is not None else cor
    total = torch.sum(torch.abs(refcoil), dim=coil_dim, keepdim=True)
    denom = torch.clamp(total, min=1e-6 * float(total.max()))
    virtual = torch.sum(refcoil / denom, dim=coil_dim, keepdim=True)

    # Per-coil phase relative to the virtual coil, smoothed
    prod_smooth = _dlpsmooth(virtual * im.conj(), 12)
    # conj(s_j) * exp(i*angle(s_j * exp(i*angle(prod)))) == |s_j| exp(i*angle(prod))
    weighted = torch.abs(im) * torch.exp(1j * torch.angle(prod_smooth))
    invcov = torch.linalg.inv(covariance)
    # sum_ij s_i invcov_ij w_j with MATLAB's invcov, which is (Psi^-1)^T
    total = torch.einsum("...i,ij,...j->...",
                         im.movedim(coil_dim, -1),
                         invcov.T.to(dtype=im.dtype, device=im.device),
                         weighted.movedim(coil_dim, -1).to(im.dtype))
    return torch.polar(torch.sqrt(torch.abs(total)), torch.angle(total))


def ci4_combine(
        im: torch.Tensor,
        covariance: torch.Tensor,
        coil_dim: int=-4,
        echo_dim: Optional[int]=None,
        time_dim: Optional[int]=None,
        t_ave: tuple[int, int]=(0, 1),
        ) -> torch.Tensor:
    """
    Optimal combine with `CreateImg4` mode 3 reference selection.

    - Multi-echo: magnitude reference is the time-averaged image of the same
      echo; phase reference is the time-averaged first echo.
    - Single-echo, multi-dynamic: both references are the time-averaged
      image.
    - Single-echo, single-dynamic: `virtual_coil_combine`.

    Parameters
    ----------
    im : torch.Tensor
        Complex coil images.
    covariance : torch.Tensor
        Noise covariance ``Psi``, shape ``(coil, coil)``.
    coil_dim : int, optional
        Coil axis. Defaults to -4.
    echo_dim, time_dim : int, optional
        Echo and dynamic axes, if present.
    t_ave : tuple of int, optional
        Half-open range of dynamics averaged for the reference. Defaults to
        ``(0, 1)``, the first dynamic (MATLAB ``tAve = [1, 1]``).

    Returns
    -------
    torch.Tensor
        Combined complex image with the coil axis removed.
    """
    n_echo = im.shape[echo_dim] if echo_dim is not None else 1
    n_time = im.shape[time_dim] if time_dim is not None else 1
    if n_echo == 1 and n_time == 1:
        return virtual_coil_combine(im, covariance, coil_dim=coil_dim)

    invcov = torch.linalg.inv(covariance)
    ref = im
    if time_dim is not None:
        ref = im.narrow(time_dim, t_ave[0], t_ave[1] - t_ave[0]).mean(dim=time_dim, keepdim=True)
    phase_ref = ref.narrow(echo_dim, 0, 1) if (echo_dim is not None and n_echo > 1) else ref
    return optimal_combine(im, invcov, ref=ref, phase_ref=phase_ref, coil_dim=coil_dim)
