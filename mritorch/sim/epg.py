from __future__ import annotations
from typing import Tuple, List, Union, Literal, overload

import torch

Numeric = Union[int, float, torch.Tensor]
NumSeq = Union[List[Numeric], Tuple[Numeric], Numeric]

def _simulate_events_matrix(
        fa_steps: torch.Tensor,
        phase_steps: torch.Tensor,
        time_steps: torch.Tensor,
        dephase_steps: torch.Tensor,
        save_steps: torch.Tensor,
        B1: torch.Tensor,
        T1: torch.Tensor,
        T2: torch.Tensor,
        M0: torch.Tensor,
        initial_state: torch.Tensor
        ) -> torch.Tensor:
    num_steps = fa_steps.shape[-1]
    num_saves = int(torch.sum(save_steps))
    state = initial_state.clone()
    # Preallocate output
    output = torch.zeros(*state.shape, num_saves, dtype=torch.cfloat) # type: ignore

    for i in range(num_steps):
        # Excitation
        T_ex = excitation_operator(
            torch.abs(B1) * fa_steps[i],
            phase_steps[i] + rad2deg(torch.angle(B1))
            )
        state = T_ex @ state

        # Relaxation
        E_relax, E_recover = relaxation_operator(time_steps[i], T1, T2, M0)
        state = E_relax[..., None] * state + E_recover[..., None]

        # Dephasing
        state = dephase_matrix(state, int(dephase_steps[i]))

        # Save
        if save_steps[i]:
            save_idx = int(torch.sum(save_steps[:i+1]) - 1)
            output[..., save_idx] = state
    
    return output

def _simulate_events_vector(
        fa_steps: torch.Tensor,
        phase_steps: torch.Tensor,
        time_steps: torch.Tensor,
        dephase_steps: torch.Tensor,
        save_steps: torch.Tensor,
        B1: torch.Tensor,
        T1: torch.Tensor,
        T2: torch.Tensor,
        M0: torch.Tensor,
        initial_F: torch.Tensor,
        initial_Z: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor]:
    num_steps = fa_steps.shape[-1]
    num_saves = int(torch.sum(save_steps))
    F, Z = initial_F.clone(), initial_Z.clone()
    
    k_max = Z.shape[-1] - 1

    # Preallocate output
    F_output = torch.zeros(*F.shape, num_saves, dtype=torch.cfloat) # type: ignore
    Z_output = torch.zeros(*Z.shape, num_saves, dtype=torch.cfloat) # type: ignore

    for i in range(num_steps):
        # Excitation
        T_ex = excitation_operator(
            torch.abs(B1) * fa_steps[i],
            phase_steps[i] + rad2deg(torch.angle(B1))
            )
        
        F, Z = matrix_to_vectors(T_ex @ vectors_to_matrix(F, Z))

        # Fp, Fm = F[..., :k_max+1], F[..., -k_max:]
        # F[..., :k_max+1] = T_ex[..., 0, 0] * Fp + T_ex[..., 0, 1] * torch.conj(Fm) + T_ex[..., 0, 2] * Z
        # F[..., -k_max:] = torch.conj( T_ex[..., 1, 0] * Fp + T_ex[..., 1, 1] * torch.conj(Fm) + T_ex[..., 1, 2] * Z )
        # Z = T_ex[..., 2, 0] * Fp + T_ex[..., 2, 1] * torch.conj(Fm) + T_ex[..., 2, 2] * Z

        # Relaxation
        E_relax, E_recover = relaxation_operator(time_steps[i], T1, T2, M0)

        F = E_relax[..., 0] * F
        Z[..., :] = E_relax[..., 2] * Z[..., :]
        Z[..., 0] = Z[..., 0] + E_recover[..., 2]

        # Dephasing
        F = dephase_vector(F, int(dephase_steps[i]))

        # Save
        if save_steps[i]:
            save_idx = int(torch.sum(save_steps[:i+1]) - 1)
            F_output[..., save_idx] = F
            Z_output[..., save_idx] = Z
    
    return F_output, Z_output

def simulate_events(
    fa_steps: NumSeq=0.0,
    phase_steps: NumSeq=0.0,
    time_steps: NumSeq=0.0,
    dephase_steps: NumSeq=1,
    save_steps: NumSeq=1,
    num_steps: int | None=None,
    B1: Numeric=1.0,
    T1: Numeric=torch.inf,
    T2: Numeric=torch.inf,
    M0: Numeric=1.0,
    state_representation: Literal['matrix', 'vector']='matrix',
    initial_state: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None=None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Simulate EPG states for a sequence of RF pulses.

    Arguments are either sequential (list of scalars) or batch parameters. Sequential paramters must be broadcastable to the same shape, where the final dimension is the number of steps to iterate over. Batch parameters can be single scalars or have 0th dimension with N entries (N,). Initial states should have shape *******.

    For each entry in the sequence:
    1. Excitation (flip_angles, phase_angles)
    -. [Diffusion - not currently implemented]
    2. Relaxation (time_steps, T1, T2, M0)
    -. [Flow - not currently implemented]
    3. Dephasing (dephase_steps)
    4. Save out the state (save_steps)
    Repeat in order for each entry in sequence.

    Args:
    Sequence parameters:
        fa_steps (NumSeq, optional): Sequence of nominal flip angles for the sequence with length `num_steps`. Defaults to 0.0.
        phase_steps (NumSeq, optional): Sequence of nominal phase angles for the sequence with length `num_steps`. Defaults to 0.0.
        time_steps (NumSeq, optional): Sequence of relaxation time delays for the sequence with length `num_steps`. Defaults to 0.0.
        dephase_steps (NumSeq, optional): Sequence of epg dephasing steps with length `num_steps`. Defaults to 1.
        save_steps (NumSeq, optional): Sequence defining whether or not to save out the final state data for each  `num_steps`. Defaults to 1.
    Voxel parameters:
        B1 (Numeric, optional): Local magnitude and phase of the excitation (real or complex scale and/or phase factor). Defaults to 1.0.
        T1 (Numeric, optional): _description_. Defaults to torch.inf.
        T2 (Numeric, optional): _description_. Defaults to torch.inf.
        M0 (Numeric, optional): _description_. Defaults to 1.0.
    Simulation parameters:
        initial_state (Numeric | None, optional): _description_. Defaults to None.
        state_representation (Literal['matrix', 'vector'], optional): _description_. Defaults to 'matrix'.

    Returns:
        torch.Tensor: _description_
    """
    # Convert sequence parameters to tensors
    fas, phases, dts, dks, saves = map(totensor, (fa_steps, phase_steps, time_steps, dephase_steps, save_steps))
    # Operating mode - repeat pattern N times, or apply series of uniquely specified events
    if num_steps is not None:
        if num_steps < 1:
            raise ValueError("num_steps must be a positive integer.")
    else:
        num_steps = max(
            fas.shape[-1] if fas.ndim > 0 else 1,
            phases.shape[-1] if phases.ndim > 0 else 1,
            dts.shape[-1] if dts.ndim > 0 else 1,
            dks.shape[-1] if dks.ndim > 0 else 1,
            saves.shape[-1] if saves.ndim > 0 else 1
            )
    # Broadcast sequence parameters to the same shape with num_steps entries
    fas = torch.broadcast_to(fas, (num_steps,))
    phases = torch.broadcast_to(phases, (num_steps,))
    dts = torch.broadcast_to(dts, (num_steps,))
    dks = torch.broadcast_to(dks, (num_steps,))
    saves = torch.broadcast_to(saves, (num_steps,))

    # Maximum total dephasing moment
    max_dephase = torch.max(torch.cumsum(dks, dim=-1))

    # Convert voxel parameters to tensors
    b1, t1, t2, m0 = map(totensor, (B1, T1, T2, M0))

    # Get the broadcastable shape for voxel parameters
    vox_param_shape = torch.broadcast_shapes(b1.shape, t1.shape, t2.shape, m0.shape)

    # Branch for different representations of EPG states
    if state_representation == 'matrix':
        if initial_state is None:
            matrix_shape = vox_param_shape + (3, max_dephase + 1)

            initial_state = torch.zeros(*matrix_shape, dtype=torch.cfloat) # type: ignore
            initial_state[..., 2, 0] = m0.clone() # type: ignore
        else:
            initial_state = totensor(initial_state) # type: ignore

        return _simulate_events_matrix(
            fas, phases, dts, dks, saves, b1, t1, t2, m0,
            initial_state # type: ignore
        )
    
    else:
        if initial_state is None:
            F_shape = vox_param_shape + (2*max_dephase + 1,)
            Z_shape = vox_param_shape + (max_dephase + 1,)

            F = torch.zeros(*F_shape, dtype=torch.cfloat) # type: ignore
            Z = torch.zeros(*Z_shape, dtype=torch.cfloat) # type: ignore
            Z[..., 0] = m0.clone()

        else:
            F = totensor(initial_state[0])
            Z = totensor(initial_state[1])
        
        return _simulate_events_vector(
            fas, phases, dts, dks, saves, b1, t1, t2, m0, F, Z
        )

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

def totensor(x: NumSeq) -> torch.Tensor:
    """Convert input to tensor."""
    if torch.is_tensor(x):
        return x # type: ignore
    else:
        return torch.tensor(x)

def deg2rad(deg: NumSeq) -> torch.Tensor:
    """Convert degrees to radians."""
    return totensor(deg) * torch.pi / 180.0

def rad2deg(rad: NumSeq) -> torch.Tensor:
    """Convert radians to degrees."""
    return totensor(rad) * 180.0 / torch.pi
