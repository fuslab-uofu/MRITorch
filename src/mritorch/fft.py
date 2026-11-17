"""
Centered Fourier transforms.

Conventions follow ISMRMRD/Gadgetron: k-space -> image is a centered inverse
FFT (`ifftc`), image -> k-space is a centered forward FFT (`fftc`). The DC
sample sits at index ``N // 2`` (the `fftshift` convention). Transforms are
orthonormal by default so that `fftc` and `ifftc` are exact adjoints.

Note that the legacy MATLAB Cartesian pipeline (`CreateImg4`) used a forward
FFT for k-space -> image. Images from that pipeline are index-reversed
(about ``N // 2``) relative to `ifftc`.
"""
from __future__ import annotations
from typing import Sequence, Union, Optional

import torch

Dims = Union[int, Sequence[int]]


def _as_tuple(dim: Dims) -> tuple[int, ...]:
    return (dim,) if isinstance(dim, int) else tuple(dim)


def fftc(x: torch.Tensor, dim: Dims=(-3, -2, -1), norm: Optional[str]="ortho") -> torch.Tensor:
    """
    Centered forward FFT along `dim` (image -> k-space).

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    dim : int or sequence of int, optional
        Dimensions to transform. Defaults to the last three (z, y, x).
    norm : str or None, optional
        Normalization passed to `torch.fft.fftn`. Defaults to 'ortho'.

    Returns
    -------
    torch.Tensor
        Centered FFT of `x`, same shape as `x`.
    """
    dim = _as_tuple(dim)
    x = torch.fft.ifftshift(x, dim=dim)
    x = torch.fft.fftn(x, dim=dim, norm=norm)
    return torch.fft.fftshift(x, dim=dim)


def ifftc(x: torch.Tensor, dim: Dims=(-3, -2, -1), norm: Optional[str]="ortho") -> torch.Tensor:
    """
    Centered inverse FFT along `dim` (k-space -> image).

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    dim : int or sequence of int, optional
        Dimensions to transform. Defaults to the last three (kz, ky, kx).
    norm : str or None, optional
        Normalization passed to `torch.fft.ifftn`. Defaults to 'ortho'.

    Returns
    -------
    torch.Tensor
        Centered inverse FFT of `x`, same shape as `x`.
    """
    dim = _as_tuple(dim)
    x = torch.fft.ifftshift(x, dim=dim)
    x = torch.fft.ifftn(x, dim=dim, norm=norm)
    return torch.fft.fftshift(x, dim=dim)


def kspace_to_image(ks: torch.Tensor, dim: Dims=(-3, -2, -1)) -> torch.Tensor:
    """Alias for `ifftc` with orthonormal scaling."""
    return ifftc(ks, dim=dim)


def image_to_kspace(im: torch.Tensor, dim: Dims=(-3, -2, -1)) -> torch.Tensor:
    """Alias for `fftc` with orthonormal scaling."""
    return fftc(im, dim=dim)
