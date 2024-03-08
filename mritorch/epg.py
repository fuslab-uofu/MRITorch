from __future__ import annotations
from typing import Tuple, List, Union

import torch

Numeric = Union[int, float, torch.Tensor]

def excitation_operator(flip_angle: Numeric, phase_angle: Numeric=0.0) -> torch.Tensor:
    """Operator for mixing magnetization between EPG states due to excitation.

    Args:
        flip_angle (Numeric): (N,1) or (1,) array of flip angles in degrees.
        phase_angle (Numeric, optional): (N,1) or (1,) array of phase angles in degrees. Defaults to 0.0.

    Returns:
        torch.Tensor: (N,3,3) or (3,3) array of excitation operators. Use matrix multiplication to apply to EPG states.
    """
    fa, phase = deg2rad(flip_angle), deg2rad(phase_angle)

    # Excitation operator
    if fa.ndim == 0 or fa.shape[0] == 1:
        T = torch.zeros(3, 3, dtype=torch.cfloat)
    else:
        T = torch.zeros(fa.shape[0], 3, 3, dtype=torch.cfloat)

    # Populate excitation entries
    T[..., 0, 0] = torch.cos(fa / 2.0) ** 2
    T[..., 0, 1] = torch.sin(fa / 2.0) ** 2 * torch.exp(2j * phase)
    T[..., 0, 2] = -1j * torch.exp(1j * phase) * torch.sin(fa)
    T[..., 1, 0] = torch.sin(fa / 2.0) ** 2 * torch.exp(-2j * phase)
    T[..., 1, 1] = torch.cos(fa / 2.0) ** 2
    T[..., 1, 2] = 1j * torch.exp(-1j * phase) * torch.sin(fa)
    T[..., 2, 0] = -0.5j * torch.exp(-1j * phase) * torch.sin(fa)
    T[..., 2, 1] = 0.5j * torch.exp(1j * phase) * torch.sin(fa)
    T[..., 2, 2] = torch.cos(fa)

    return T

def deg2rad(deg: Numeric) -> torch.Tensor:
    """Convert degrees to radians."""
    if isinstance(deg, (int, float)):
        return torch.Tensor([deg * torch.pi / 180.0])
    else:
        return torch.Tensor(deg) * (torch.pi / 180.0)
