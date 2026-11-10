"""
SENSE coil combination with sensitivity maps.

For fully sampled data and unit-norm maps ``c`` (``sum_i |c_i|^2 = 1`` inside the object) the SENSE
solution ``x = (sum_i conj(c_i) y_i) / (sum_i |c_i|^2 + lambda)`` is the matched-filter combination. Without noise
its magnitude equals the root sum of squares of the coil images; unlike the root sum of squares it is linear
(so it commutes with image filters), preserves phase, and is not biased upward where the signal is below the noise.
This is the combination performed by BART ``pics`` (``-r`` is the weight ``lambda`` of the L2 regularizer) for fully
sampled Cartesian data, which is how the MPME images are reconstructed.
"""
from __future__ import annotations
from typing import Optional

import torch


def sense_combine(im: torch.Tensor, maps: torch.Tensor, coil_dim: int=-3, reg: float=0.0,
                  mask_threshold: Optional[float]=None) -> torch.Tensor:
    """
    Matched-filter SENSE combination of coil images.

    Parameters
    ----------
    im : torch.Tensor
        Complex coil images, coil axis at `coil_dim`.
    maps : torch.Tensor
        Sensitivity maps broadcastable to `im` (same coil axis).
    coil_dim : int, optional
        Coil axis of `im` (and `maps`). Defaults to -3 (``(coil, y, x)`` images).
    reg : float, optional
        Tikhonov weight added to ``sum |c|^2`` (BART ``pics -r``).
    mask_threshold : float, optional
        Pixels where ``sum |c|^2`` is below this value are set to zero (cropped maps).

    Returns
    -------
    torch.Tensor
        Combined image with the coil axis removed.
    """
    num = torch.sum(maps.conj() * im, dim=coil_dim)
    den = torch.sum(maps.abs() ** 2, dim=coil_dim) + reg
    out = num / den.clamp_min(1e-12 if reg == 0.0 else 0.0)
    if mask_threshold is not None:
        out = torch.where(torch.sum(maps.abs() ** 2, dim=coil_dim) >= mask_threshold, out, torch.zeros_like(out))
    return out
