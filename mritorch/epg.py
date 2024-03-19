from __future__ import annotations
from typing import Tuple, List, Union

import torch

Numeric = Union[int, float, torch.Tensor]

def matrix_to_vectors(S: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Converts EPG state matrix to state vectors.

    Args:
        S (torch.Tensor): (N,3,K) or (3,K) matrix of EPG states.
    
    Returns:
        (torch.Tensor, torch.Tensor): ((N,2*K-1), (N, K)) or ((2*K-1,), (K,)) 
        arrays of (F+, Z) EPG states. The first entry represents F states and 
        the second represents Z states. 

        If N = 2,
        Vector format:
        F[0] == F_0
        F[1] == F_{+1}
        F[2] == F_{+2}
        F[-1] == F[K-1] == F_{-1}
        F[-2] == F[K-2] == F_{-2}

        Matrix format:
        [[     F[0] ,      F[ 1],       F[ 2] , ...,      F[K-1]],
         [conj(F[0]), conj(F[-1]), conj(F[-2]), ..., conj(F[-K])],
         [     Z[0] ,      Z[ 1],       Z[ 2] , ...,      Z[K-1]]
    """
    K = S.shape[-1]
    # Preallocate outputs
    if S.ndim > 2:
        N = S.shape[0]
        F = torch.zeros(N, 2 * K - 1, dtype=torch.cfloat)
        Z = torch.zeros(N, K, dtype=torch.cfloat)
    else:
        F = torch.zeros(2 * K - 1, dtype=torch.cfloat)
        Z = torch.zeros(K, dtype=torch.cfloat)
    
    # Check that the 0th state is correctly defined
    if torch.any(S[..., 1, 0] != torch.conj(S[..., 0, 0])):
        raise ValueError("The 0th state is not correctly defined. S[...,0,0] != conj(S[...,1,0])")

    # Populate vectors
    F[..., 0:K] = S[..., 0, 0:K]
    F[..., -K+1:] = torch.flip(torch.conj(S[..., 1, 1:K]), dims=[-1])
    Z[..., :] = S[..., 2, :]

    return F, Z

def vectors_to_matrix(F: torch.Tensor, Z: torch.Tensor) -> torch.Tensor:
    """Converts EPG state vectors to a state matrix.

    Args:
        F (torch.Tensor): _description_
        Z (torch.Tensor): _description_
    
    Returns:
        torch.Tensor: _description_
    """
    K = (F.shape[-1] + 1) // 2
    # Preallocate output
    if F.ndim > 1:
        N = F.shape[0]
        S = torch.zeros(N, 3, K, dtype=torch.cfloat)
    else:
        S = torch.zeros(3, K, dtype=torch.cfloat)
    
    # Populate matrix
    S[..., 0, :] = F[..., 0:K]
    S[..., 1, 0] = torch.conj(F[..., 0])
    S[..., 1, 1:] = torch.flip(torch.conj(F[..., -K+1:]), dims=[-1])
    S[..., 2, :] = Z

    return S

def dephase_vector(F_states: torch.Tensor, k: int=1) -> torch.Tensor:
    """Dephase EPG state vector by k indices.

    Args:
        F (torch.Tensor): (N,2*K-1) or (2*K-1,) array of EPG states.
        k (int, optional): Number of state indices to dephase by. Defaults to 1.

    Returns:
        torch.Tensor: (N,2*K-1) or (2*K-1,) array of dephased EPG states.
    """
    new_states = F_states.clone()
    if k == 0:
        return new_states

    K = (F_states.shape[-1] + 1) // 2

    # Zero out states that wrap around
    if k > 0:
        new_states[..., K-k:K] = 0
    elif k < 0:
        new_states[..., K:K-k] = 0

    # Apply circular shift to dephase states
    new_states = torch.roll(new_states, shifts=k, dims=-1)

    return new_states

def dephase_matrix(state: torch.Tensor, k: int=1) -> torch.Tensor:
    """Dephase EPG state matrix by k indices.

    S(dk) : F_k -> F_{k+dk}
            Z_k -> Z_k

    Args:
        state (torch.Tensor): (N,3,K) or (3,K) array of EPG states.
        k (int, optional): Number of state indices to dephase by. Defaults to 1.

    Returns:
        torch.Tensor: (N,3,K) or (3,K) array of dephased EPG states.
    """
    new_state = state.clone()

    ### Dephase transverse magnetization (rows 1 and 2)
    # Get the states that will pass through the coherent state; zero out states
    # that wrap around
    if k > 0:
        tmp = state[..., 1:2, 1:k]
        new_state[..., 1:2, 0:k] = 0
    elif k < 0:
        tmp = state[..., 0:1, 1:(-k)]
        new_state[..., 0:1, 0:(-k)] = 0
    else:
        return new_state
    
    shiftdim = 1 if state.ndim < 3 else 2

    # First row is dephasing magnetization F+; shift away from coherent state
    new_state[..., 0:1, :] = torch.roll(new_state[..., 0:1, :], shifts=k, dims=shiftdim)
    # Second row is rephasing magnetization F-; shift towards coherent state
    new_state[..., 1:2, :] = torch.roll(new_state[..., 1:2, :], shifts=-k, dims=shiftdim)

    # Update states that pass through coherence or become coherent
    tmp = torch.conj(torch.flip(tmp, dims=[shiftdim]))
    if k > 0:
        new_state[..., 0:1, 1:k] = tmp  # Add new dephasing states
        new_state[..., 0, 0] = torch.conj(new_state[..., 1, 0])  # Coherence
    elif k < 0:
        new_state[..., 1:2, 1:(-k)] = tmp  # Add new rephasing states
        new_state[..., 1, 0] = torch.conj(new_state[..., 0, 0])  # Coherence
    
    # index 2 is longitudinal magnetization Z_k; no change
        
    return new_state

def excitation_operator(flip_angle: Numeric, phase_angle: Numeric=0.0) -> torch.Tensor:
    """Operator for mixing magnetization between EPG states due to excitation.

    Cross-reference Weigel 2015, "Extended phase graphs: Dephasing, RF pulses, and echoes - Pure and simple" Eq. 15.

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
    T[..., 0, 0] =                                  torch.cos(fa / 2.0) ** 2
    T[..., 0, 1] =         torch.exp(+2j * phase) * torch.sin(fa / 2.0) ** 2
    T[..., 0, 2] = -1.0j * torch.exp(+1j * phase) * torch.sin(fa)
    T[..., 1, 0] =         torch.exp(-2j * phase) * torch.sin(fa / 2.0) ** 2
    T[..., 1, 1] =                                  torch.cos(fa / 2.0) ** 2
    T[..., 1, 2] = +1.0j * torch.exp(-1j * phase) * torch.sin(fa)
    T[..., 2, 0] = -0.5j * torch.exp(-1j * phase) * torch.sin(fa)
    T[..., 2, 1] = +0.5j * torch.exp(+1j * phase) * torch.sin(fa)
    T[..., 2, 2] =                                  torch.cos(fa)

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
    E1 = torch.exp(-totensor(dt) / totensor(T1))
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
