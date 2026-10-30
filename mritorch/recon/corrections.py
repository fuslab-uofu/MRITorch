"""
K-space corrections.
"""
from __future__ import annotations
from typing import Sequence

import torch


def drift_correction(
        ks: torch.Tensor,
        time_dim: int,
        average_dims: Sequence[int],
        n_baseline: int=5,
        ) -> torch.Tensor:
    """
    Remove global B0 drift phase from dynamic k-space.

    Port of MATLAB `ks_drift_correction` (Parker et al., MRM 2022). The
    phase of each dynamic relative to the baseline mean, averaged over
    `average_dims`, is removed.

    Parameters
    ----------
    ks : torch.Tensor
        Complex k-space.
    time_dim : int
        Dynamic (repetition) axis.
    average_dims : sequence of int
        Axes averaged to estimate the drift phase, typically kx, kz and coil.
    n_baseline : int, optional
        Number of initial dynamics forming the reference. Defaults to 5.

    Returns
    -------
    torch.Tensor
        Drift-corrected k-space.
    """
    ref = ks.narrow(time_dim, 0, n_baseline).mean(dim=time_dim, keepdim=True)
    phase = torch.angle(torch.mean(ks * ref.conj(), dim=tuple(average_dims), keepdim=True))
    return ks * torch.exp(-1j * phase)
