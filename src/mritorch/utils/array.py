"""
Center-preserving pad, crop, and zero-filled interpolation.

Sizes are given for the trailing dimensions of the input: ``shape=(ny, nx)``
acts on the last two dimensions. An entry of ``None`` (or -1) keeps that
dimension's size.

"Center" means the `fftshift` center, index ``N // 2``. Padding and cropping
keep that index aligned, so k-space DC stays at the center and image-space
operations introduce no half-pixel shift. This matches MATLAB
`pad_to_size`/`crop_to_size` except when an even-length dimension changes by
an odd amount, where the MATLAB versions shift the center by one sample.
"""
from __future__ import annotations
from typing import Sequence, Optional

import torch

from mritorch.fft import fftc, ifftc

Shape = Sequence[Optional[int]]


def _resolve_shape(x: torch.Tensor, shape: Shape) -> tuple[list[int], list[int]]:
    """Return (dims, sizes) for the trailing dims addressed by `shape`."""
    if len(shape) > x.ndim:
        raise ValueError(f"shape has {len(shape)} entries but input has {x.ndim} dims")
    dims = list(range(x.ndim - len(shape), x.ndim))
    sizes = [x.shape[d] if (s is None or s < 0) else int(s) for d, s in zip(dims, shape)]
    return dims, sizes


def resize_center(x: torch.Tensor, shape: Shape, value: complex=0) -> torch.Tensor:
    """
    Pad and/or crop the trailing dimensions of `x` about their centers.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    shape : sequence of int or None
        Output sizes for the trailing ``len(shape)`` dimensions.
    value : complex, optional
        Fill value for padded regions. Defaults to 0.

    Returns
    -------
    torch.Tensor
        Resized tensor. The sample at index ``N // 2`` of each input dimension
        lands at index ``M // 2`` of the output.
    """
    dims, sizes = _resolve_shape(x, shape)
    out_shape = list(x.shape)
    src = [slice(None)] * x.ndim
    dst = [slice(None)] * x.ndim
    for d, m in zip(dims, sizes):
        n = x.shape[d]
        out_shape[d] = m
        if m >= n:
            start = m // 2 - n // 2
            dst[d] = slice(start, start + n)
        else:
            start = n // 2 - m // 2
            src[d] = slice(start, start + m)
    out = torch.full(out_shape, value, dtype=x.dtype, device=x.device)
    out[tuple(dst)] = x[tuple(src)]
    return out


def pad_to_size(x: torch.Tensor, shape: Shape, value: complex=0) -> torch.Tensor:
    """
    Center-pad the trailing dimensions of `x` to `shape`.

    Dimensions already larger than requested are left unchanged (as in the
    MATLAB `pad_to_size`). See `resize_center` for parameters.
    """
    dims, sizes = _resolve_shape(x, shape)
    sizes = [max(m, x.shape[d]) for d, m in zip(dims, sizes)]
    return resize_center(x, sizes, value)


def crop_to_size(x: torch.Tensor, shape: Shape) -> torch.Tensor:
    """
    Center-crop the trailing dimensions of `x` to `shape`.

    Dimensions already smaller than requested are left unchanged (as in the
    MATLAB `crop_to_size`). See `resize_center` for parameters.
    """
    dims, sizes = _resolve_shape(x, shape)
    sizes = [min(m, x.shape[d]) for d, m in zip(dims, sizes)]
    return resize_center(x, sizes)


def zfi_to_size(x: torch.Tensor, shape: Shape, preserve_intensity: bool=True) -> torch.Tensor:
    """
    Zero-filled (Fourier) interpolation of the trailing dimensions of `x`.

    Each dimension whose size changes is transformed to k-space, padded or
    cropped about its center, and transformed back.

    Parameters
    ----------
    x : torch.Tensor
        Image-space input.
    shape : sequence of int or None
        Output sizes for the trailing ``len(shape)`` dimensions.
    preserve_intensity : bool, optional
        If True (default), a constant image keeps its value. If False, the
        output is scaled by ``prod(N_in / N_out)`` to match MATLAB
        `zfi_to_size`, which used unnormalized `fft`/`ifft`.

    Returns
    -------
    torch.Tensor
        Interpolated tensor.
    """
    dims, sizes = _resolve_shape(x, shape)
    changed = [d for d, m in zip(dims, sizes) if m != x.shape[d]]
    if not changed:
        return x.clone()
    if not torch.is_complex(x):
        raise TypeError("zfi_to_size expects a complex tensor; use x.to(torch.cfloat) first")
    n_in = [x.shape[d] for d in changed]
    ks = fftc(x, dim=changed, norm="forward")
    ks = resize_center(ks, sizes)
    out = ifftc(ks, dim=changed, norm="forward")
    if not preserve_intensity:
        scale = 1.0
        for d, n in zip(changed, n_in):
            scale *= n / out.shape[d]
        out = out * scale
    return out
