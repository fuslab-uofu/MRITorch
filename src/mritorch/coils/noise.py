"""
Coil noise statistics.

The noise covariance uses the standard convention
``Psi[j, k] = E[n_j * conj(n_k)]``. (MATLAB's `cov` returns the complex
conjugate of this matrix, so MATLAB `invcov` equals ``Psi^-1`` transposed.)
"""
from __future__ import annotations

import torch


def noise_covariance(noise: torch.Tensor, coil_dim: int=0) -> torch.Tensor:
    """
    Unbiased coil noise covariance from noise samples.

    Port of MATLAB `coil_covariance`.

    Parameters
    ----------
    noise : torch.Tensor
        Complex noise samples with a coil axis at `coil_dim`; all other axes
        are treated as samples.
    coil_dim : int, optional
        Coil axis. Defaults to 0.

    Returns
    -------
    torch.Tensor
        Covariance matrix of shape ``(coil, coil)``.
    """
    x = noise.movedim(coil_dim, 0).reshape(noise.shape[coil_dim], -1)
    return torch.cov(x)


def prewhiten(data: torch.Tensor, covariance: torch.Tensor, coil_dim: int=-4) -> torch.Tensor:
    """
    Decorrelate coils so the noise covariance becomes the identity.

    Applies ``L^-1`` along `coil_dim`, where ``covariance = L L^H``.

    Parameters
    ----------
    data : torch.Tensor
        Complex data with a coil axis at `coil_dim`.
    covariance : torch.Tensor
        Noise covariance ``(coil, coil)`` from `noise_covariance`.
    coil_dim : int, optional
        Coil axis. Defaults to -4, the coil axis of ``(..., coil, z, y, x)``.

    Returns
    -------
    torch.Tensor
        Prewhitened data, same shape as `data`.
    """
    L = torch.linalg.cholesky(covariance.to(dtype=data.dtype, device=data.device))
    W = torch.linalg.inv(L)
    x = data.movedim(coil_dim, -1)
    x = x @ W.T
    return x.movedim(-1, coil_dim)
