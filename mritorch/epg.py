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

    # Preallocate outputs
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

def relaxation_operator(
        dt: Numeric,
        T1: Numeric = torch.inf,
        T2: Numeric = torch.inf,
        M0: Numeric = 1.0
        ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Operator for decaying and restoring magnetization due to relaxation.

    Args:
        dt (Numeric): (N,1) or (1,) array of time step in seconds.
        T1 (Numeric, optional): (N,1) or (1,) array of T1 relaxation times. Defaults to inf.
        T2 (Numeric, optional): (N,1) or (1,) array of T2 relaxation times. Defaults to inf.
        M0 (Numeric, optional): (N,1) or (1,) array of equilibrium magnetization. Defaults to 1.0.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Each entry has shape (N,3) or (3,) and represents the elements of the diagonal operators for relaxation and recovery.
    """
    # Determine number of entries
    N = max(dt.shape[0] if torch.is_tensor(dt) else 0,  # type: ignore
            T1.shape[0] if torch.is_tensor(T1) else 0,  # type: ignore
            T2.shape[0] if torch.is_tensor(T2) else 0,  # type: ignore
            M0.shape[0] if torch.is_tensor(M0) else 0)  # type: ignore
    
    # Preallocate outputs
    if N <= 1:
        Erelax = torch.zeros(3, dtype=torch.float)
        Erecovery = torch.zeros(3, dtype=torch.float)
    else:
        Erelax = torch.zeros(N, 3, dtype=torch.float)
        Erecovery = torch.zeros(N, 3, dtype=torch.float)

    # Compute decay proportions for T1 and T2
    if T1 == torch.inf:
        E1 = totensor(1.0)
    else:
        E1 = torch.exp(-totensor(dt) / totensor(T1))
    if T2 == torch.inf:
        E2 = totensor(1.0)
    else:
        E2 = torch.exp(-totensor(dt) / totensor(T2))
    
    Erelax[..., 0] = E2
    Erelax[..., 1] = E2
    Erelax[..., 2] = E1

    Erecovery[..., 2] = totensor(M0) * (1 - E1)

    return Erelax, Erecovery

def totensor(x: Numeric) -> torch.Tensor:
    """Convert input to tensor."""
    if isinstance(x, (int, float)):
        return torch.Tensor([x])
    else:
        return torch.Tensor(x)

def deg2rad(deg: Numeric) -> torch.Tensor:
    """Convert degrees to radians."""
    if isinstance(deg, (int, float)):
        return torch.Tensor([deg * torch.pi / 180.0])
    else:
        return torch.Tensor(deg) * (torch.pi / 180.0)
