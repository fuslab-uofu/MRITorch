from __future__ import annotations
from typing import Tuple, List, Union

import torch

Numeric = Union[int, float, torch.Tensor]
NumSeq = Union[List[Numeric], Tuple[Numeric], Numeric]

def simulate_vector(
        fa_steps: NumSeq=0.0,
        phase_steps: NumSeq=0.0,
        time_steps: NumSeq=0.0,
        dephase_steps: NumSeq=1,
        save_steps: NumSeq=1,
        T1: Numeric=torch.inf,
        T2: Numeric=torch.inf,
        M0: Numeric=1.0,
        initial_F: Numeric=0.0,
        initial_Z: Numeric=1.0,
        ) -> tuple[torch.Tensor, torch.Tensor]:
    """Simulate EPG states for a sequence of RF pulses.

    Arguments are either sequential (list of scalars) or batch parameters. Sequential paramters can be either a single scalar or length `numSteps`. Batch parameters can be single scalars or have 0th dimension with N entries (N,). Initial states should have shape *******.

    For each entry in the sequence:
    1. Excitation (flip_angles, phase_angles)
    -. [Diffusion - not currently implemented]
    2. Relaxation (time_steps, T1, T2, M0)
    -. [Flow - not currently implemented]
    3. Dephasing (dephase_steps)
    4. Save out the state (save_steps)
    Repeat in order for each entry in sequence.

    Args:
        fa_steps (NumSeq, optional): _description_. Defaults to 0.0.
        phase_steps (NumSeq, optional): _description_. Defaults to 0.0.
        time_steps (NumSeq, optional): _description_. Defaults to 0.0.
        dephase_steps (NumSeq, optional): _description_. Defaults to 1.
        save_steps (NumSeq, optional): _description_. Defaults to 1.
        T1 (Numeric, optional): _description_. Defaults to torch.inf.
        T2 (Numeric, optional): _description_. Defaults to torch.inf.
        M0 (Numeric, optional): _description_. Defaults to 1.0.
        initial_F (Numeric, optional): _description_. Defaults to 0.0.
        initial_Z (Numeric, optional): _description_. Defaults to 1.0.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: _description_
    """
    pass

def simulate_matrix(
        fa_steps: NumSeq=0.0,
        phase_steps: NumSeq=0.0,
        time_steps: NumSeq=0.0,
        dephase_steps: NumSeq=1,
        save_steps: NumSeq=1,
        T1: Numeric=torch.inf,
        T2: Numeric=torch.inf,
        M0: Numeric=1.0,
        initial_state: Numeric=torch.tensor([[0., 0., 1.]])
        ) -> torch.Tensor:
    """Simulate EPG states for a sequence of RF pulses.

    Arguments are either sequential (list of scalars) or batch parameters. Sequential paramters can be either a single scalar or length `numSteps`. Batch parameters can be single scalars or have 0th dimension with N entries (N,). Initial states should have shape *******.

    For each entry in the sequence:
    1. Excitation (flip_angles, phase_angles)
    -. [Diffusion - not currently implemented]
    2. Relaxation (time_steps, T1, T2, M0)
    -. [Flow - not currently implemented]
    3. Dephasing (dephase_steps)
    4. Save out the state (save_steps)
    Repeat in order for each entry in sequence.

    Args:
        fa_steps (NumSeq, optional): _description_. Defaults to 0.0.
        phase_steps (NumSeq, optional): _description_. Defaults to 0.0.
        time_steps (NumSeq, optional): _description_. Defaults to 0.0.
        dephase_steps (NumSeq, optional): _description_. Defaults to 1.
        save_steps (NumSeq, optional): _description_. Defaults to 1.
        T1 (Numeric, optional): _description_. Defaults to torch.inf.
        T2 (Numeric, optional): _description_. Defaults to torch.inf.
        M0 (Numeric, optional): _description_. Defaults to 1.0.
        initial_state (Numeric, optional): _description_. Defaults to torch.tensor([0., 0., 1.]).

    Returns:
        torch.Tensor: _description_
    """
    FAs = totensor(fa_steps)
    phases = totensor(phase_steps)
    dts = totensor(time_steps)
    dks = totensor(dephase_steps)
    saves = totensor(save_steps)
    T1vals = totensor(T1)
    T2vals = totensor(T2)
    M0vals = totensor(M0)
    initial = totensor(initial_state)

    # Determine number of steps
    steps = max(FAs.shape[0], phases.shape[0], dts.shape[0], dks.shape[0], saves.shape[0])

    sh = (steps, )
    if FAs.ndim == 0 or FAs.shape[0] == 1:
        FAs = FAs.repeat(steps).reshape(sh)
    if phases.ndim == 0 or phases.shape[0] == 1:
        phases = phases.repeat(steps).reshape(sh)
    if dts.ndim == 0 or dts.shape[0] == 1:
        dts = dts.repeat(steps).reshape(sh)
    if dks.ndim == 0 or dks.shape[0] == 1:
        dks = dks.repeat(steps).reshape(sh)
    if saves.ndim == 0 or saves.shape[0] == 1:
        saves = saves.repeat(steps).reshape(sh)

    numSaves = torch.sum(saves)

    K = torch.max(torch.cumsum(dks, dim=0)) + 1
    
    # Determine number of entries
    N = max(T1vals.shape[0], T2vals.shape[0],
            M0vals.shape[0], initial.shape[0])
    
    sh = (N, 1)
    if T1vals.ndim == 0 or T1vals.shape[0] == 1:
        T1vals = T1vals.repeat(N).reshape(sh)
    if T2vals.ndim == 0 or T2vals.shape[0] == 1:
        T2vals = T2vals.repeat(N).reshape(sh)
    if M0vals.ndim == 0 or M0vals.shape[0] == 1:
        M0vals = M0vals.repeat(N).reshape(sh)
    
    state = torch.zeros(N, 3, K, dtype=torch.cfloat)
    if initial.ndim == 2:
        state[:, :, 0] = initial
    else:
        state[:, :, :] = initial

    # Preallocate outputs
    output = torch.zeros(N, 3, K, numSaves, dtype=torch.cfloat)

    # Loop through sequence
    for iStep in range(steps):
        # Excitation
        Texcite = excitation_operator(FAs[iStep], phases[iStep])
        state = torch.matmul(Texcite, state)

        # Relaxation
        Erelax, Erecovery = relaxation_operator(dts[iStep], T1vals, T2vals, M0vals)
        state = Erelax[..., None] * state
        state[..., :, 0] = state[..., :, 0] + Erecovery

        # Dephasing
        state = dephase_matrix(state, int(dks[iStep]))

        # Save state
        if saves[iStep]:
            output[..., :, :, torch.sum(saves[:iStep])] = state
    
    return output

def matrix_to_vectors(S: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Converts EPG state matrix to state vectors.

    Args:
        S (torch.Tensor): (...,3,K) matrix of EPG states.
    
    Returns:
        (torch.Tensor, torch.Tensor): ((...,2*K-1), (..., K)) arrays of (F+, Z) EPG states. 

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
    bshape = S.shape[:-2]
    K = S.shape[-1]
    # Preallocate outputs
    F = torch.zeros(*bshape, 2 * K - 1, dtype=torch.cfloat)
    Z = torch.zeros(*bshape, K, dtype=torch.cfloat)
    
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
    bshape = F.shape[:-1]
    K = (F.shape[-1] + 1) // 2

    S = torch.zeros(*bshape, 3, K, dtype=torch.cfloat)
    
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
        state (torch.Tensor): (...,3,K) array of EPG states.
        k (int, optional): Number of state indices to dephase by. Defaults to 1.

    Returns:
        torch.Tensor: (...,3,K) array of dephased EPG states.
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

    # First row is dephasing magnetization F+; shift away from coherent state
    new_state[..., 0:1, :] = torch.roll(new_state[..., 0:1, :], shifts=k, dims=-1)
    # Second row is rephasing magnetization F-; shift towards coherent state
    new_state[..., 1:2, :] = torch.roll(new_state[..., 1:2, :], shifts=-k, dims=-1)

    # Update states that pass through coherence or become coherent
    tmp = torch.conj(torch.flip(tmp, dims=[-1]))
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
        flip_angle (Numeric): array of flip angles in degrees.
        phase_angle (Numeric, optional): array of phase angles in degrees. Defaults to 0.0.

    Returns:
        torch.Tensor: (...,3,3) array of excitation operators, where `...` is the broadcastable shape of all input arguments. Use matrix multiplication to apply to EPG state matrix.
    """
    fa, phase = torch.broadcast_tensors(deg2rad(flip_angle), deg2rad(phase_angle))

    # Preallocate outputs
    T = torch.zeros(fa.shape + (3, 3), dtype=torch.cfloat)

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
        dt (Numeric): array of time step in seconds.
        T1 (Numeric, optional): array of T1 relaxation times. Defaults to inf.
        T2 (Numeric, optional): array of T2 relaxation times. Defaults to inf.
        M0 (Numeric, optional): array of equilibrium magnetization. Defaults to 1.0.

    Returns:
        (Erelax, Erecover) Tuple[torch.Tensor, torch.Tensor]: Each entry has shape (..., 3), where ... is the broadcastable shape across all input arguments, and the entries along the final dimension are the elements of the diagonal operators for relaxation and recovery.
    """

    dt_b, T1_b, T2_b, M0_b = torch.broadcast_tensors(
        totensor(dt),
        totensor(T1),
        totensor(T2),
        totensor(M0)
        )

    # Compute decay proportions for T1 and T2
    E1 = torch.exp(-dt_b / T1_b)
    E2 = torch.exp(-dt_b / T2_b)
    
    Erelax = torch.stack((E2, E2, E1), dim=-1)

    Erecover = torch.zeros_like(Erelax)
    Erecover[..., 2] = totensor(M0_b) * (1 - E1)

    return Erelax, Erecover

def totensor(x: Numeric | NumSeq) -> torch.Tensor:
    """Convert input to tensor."""
    if torch.is_tensor(x):
        return x # type: ignore
    else:
        return torch.tensor(x)

def deg2rad(deg: Numeric | NumSeq) -> torch.Tensor:
    """Convert degrees to radians."""
    return totensor(deg) * torch.pi / 180.0
