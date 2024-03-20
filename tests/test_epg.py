import unittest

import torch

from mritorch import epg

_atol = 1e-6  # Absolute tolerance for allclose

class TestExcitation(unittest.TestCase):
    def test_Tx180(self):
        Tx180 = epg.excitation_operator(180)
        truth = torch.tensor([
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, -1]
        ], dtype=torch.cfloat)

        self.assertEqual(Tx180.shape, (3, 3))
        self.assertTrue(torch.allclose(Tx180, truth, atol=_atol))

    def test_Tx90(self):
        Tx90 = epg.excitation_operator(90)
        truth = torch.tensor([
            [0.5, 0.5, -1j],
            [0.5, 0.5, +1j],
            [-0.5j, +0.5j, 0]
        ], dtype=torch.cfloat)

        self.assertEqual(Tx90.shape, (3, 3))
        self.assertTrue(torch.allclose(Tx90, truth, atol=_atol))

    def test_Ty90(self):
        Ty90 = epg.excitation_operator(90, 90)
        truth = torch.tensor([
            [0.5, -0.5, 1],
            [-0.5, 0.5, 1],
            [-0.5, -0.5, 0]
        ], dtype=torch.cfloat)

        self.assertEqual(Ty90.shape, (3, 3))
        self.assertTrue(torch.allclose(Ty90, truth, atol=_atol))

    def test_Ty180(self):
        Ty90 = epg.excitation_operator(180, 90)
        truth = torch.tensor([
            [0, -1, 0],
            [-1, 0, 0],
            [0, 0, -1]
        ], dtype=torch.cfloat)

        self.assertEqual(Ty90.shape, (3, 3))
        self.assertTrue(torch.allclose(Ty90, truth, atol=_atol))

    def test_multi(self):
        flip_angle = torch.tensor([45, 90, 180, 45, 90, 180, 45, 90, 180])
        phase_angle = torch.tensor([0, 0, 0, 45, 45, 45, 90, 90, 90])
        Tx = epg.excitation_operator(flip_angle, phase_angle=phase_angle)
        truth = torch.stack([
            epg.excitation_operator(flip_angle[i], phase_angle[i])
            for i in range(flip_angle.shape[0])
        ])

        self.assertEqual(Tx.shape, (9, 3, 3))
        self.assertTrue(torch.allclose(Tx, truth, atol=_atol))

    def test_shapes(self):
        for M in range(3, 6):
            for N in range(2, 5):
                T = epg.excitation_operator(
                    torch.rand(M,N),
                    phase_angle=torch.rand(M,N)
                )
                self.assertEqual(T.shape, (M,N,3,3))

                T = epg.excitation_operator(
                    torch.rand(M,1),
                    phase_angle=torch.rand(N)
                )
                self.assertEqual(T.shape, (M,N,3,3))

class TestRelaxation(unittest.TestCase):
    def test_defaults(self):
        Erelax, Erecovery = epg.relaxation_operator(1)
        truth_relax = torch.tensor([1, 1, 1], dtype=torch.float)
        truth_recovery = torch.tensor([0, 0, 0], dtype=torch.float)

        self.assertEqual(Erelax.shape, (3,))
        self.assertTrue(torch.allclose(Erelax, truth_relax, atol=_atol))

        self.assertEqual(Erecovery.shape, (3,))
        self.assertTrue(torch.allclose(Erecovery, truth_recovery, atol=_atol))
    
    def test_T1(self):
        T1vals = torch.tensor([10, 30, 100, 300, 1000, 3000], dtype=torch.float)
        Erelax, Erecovery = epg.relaxation_operator(1, T1=T1vals)
        E1 = torch.exp(-1 / T1vals)
        E2 = torch.ones_like(E1)
        Zs = torch.zeros_like(E1)
        truth_relax = torch.stack([E2, E2, E1], dim=-1)
        truth_recovery = torch.stack([Zs, Zs, (1 - E1)], dim=-1)

        self.assertEqual(Erelax.shape, (6, 3,))
        self.assertTrue(torch.allclose(Erelax, truth_relax, atol=_atol))

        self.assertEqual(Erecovery.shape, (6, 3,))
        self.assertTrue(torch.allclose(Erecovery, truth_recovery, atol=_atol))

    def test_T2(self):
        T2vals = torch.tensor([1, 3, 10, 30, 100, 300, 1000, 3000], dtype=torch.float)
        Erelax, Erecovery = epg.relaxation_operator(1, T2=T2vals)
        E2 = torch.exp(-1 / T2vals)
        E1 = torch.ones_like(E2)
        Zs = torch.zeros_like(E2)
        truth_relax = torch.stack([E2, E2, E1], dim=-1)
        truth_recovery = torch.stack([Zs, Zs, (1 - E1)], dim=-1)

        self.assertEqual(Erelax.shape, (8, 3,))
        self.assertTrue(torch.allclose(Erelax, truth_relax, atol=_atol))

        self.assertEqual(Erecovery.shape, (8, 3,))
        self.assertTrue(torch.allclose(Erecovery, truth_recovery, atol=_atol))

    def test_M0(self):
        M0vals = torch.tensor([0.1, 0.3, 1, 3, 10, 30], dtype=torch.float)
        T1vals = torch.tensor([10, 30, 100, 300, 1000, 3000], dtype=torch.float)
        Erelax, Erecovery = epg.relaxation_operator(1, T1=T1vals, M0=M0vals)
        E1 = torch.exp(-1 / T1vals)
        E2 = torch.ones_like(E1)
        Zs = torch.zeros_like(E1)
        truth_relax = torch.stack([E2, E2, E1], dim=-1)
        truth_recovery = torch.stack([Zs, Zs, M0vals * (1 - E1)], dim=-1)

        self.assertEqual(Erelax.shape, (6, 3,))
        self.assertTrue(torch.allclose(Erelax, truth_relax, atol=_atol))

        self.assertEqual(Erecovery.shape, (6, 3,))
        self.assertTrue(torch.allclose(Erecovery, truth_recovery, atol=_atol))

    def test_dt(self):
        dt = torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.float)
        Erelax, Erecovery = epg.relaxation_operator(dt)
        truth_relax = torch.stack([torch.ones_like(dt), torch.ones_like(dt), torch.ones_like(dt)], dim=-1)
        truth_recovery = torch.stack([torch.zeros_like(dt), torch.zeros_like(dt), torch.zeros_like(dt)], dim=-1)

        self.assertEqual(Erelax.shape, (6, 3,))
        self.assertTrue(torch.allclose(Erelax, truth_relax, atol=_atol))

        self.assertEqual(Erecovery.shape, (6, 3,))
        self.assertTrue(torch.allclose(Erecovery, truth_recovery, atol=_atol))

    def test_multi(self):
        dt = torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.float)
        T1vals = torch.tensor([10, 30, 100, 300, 1000, 3000], dtype=torch.float)
        T2vals = torch.tensor([1, 3, 10, 30, 100, 300], dtype=torch.float)
        M0vals = torch.tensor([0.1, 0.3, 1, 3, 10, 30], dtype=torch.float)
        Erelax, Erecovery = epg.relaxation_operator(dt, T1=T1vals, T2=T2vals, M0=M0vals)
        E1 = torch.exp(-dt / T1vals)
        E2 = torch.exp(-dt / T2vals)
        Zs = torch.zeros_like(E1)
        truth_relax = torch.stack([E2, E2, E1], dim=-1)
        truth_recovery = torch.stack([Zs, Zs, M0vals * (1 - E1)], dim=-1)

        self.assertEqual(Erelax.shape, (6, 3,))
        self.assertTrue(torch.allclose(Erelax, truth_relax, atol=_atol))

        self.assertEqual(Erecovery.shape, (6, 3,))
        self.assertTrue(torch.allclose(Erecovery, truth_recovery, atol=_atol))

    def test_shapes(self):
        for M in range(3, 6):
            for N in range(1, 4):
                for P in range(2, 5):
                    for Q in range(4, 7):
                        Erelax, Erecovery = epg.relaxation_operator(
                            torch.rand(M, N, P, Q),
                            T1=torch.rand(M, N, P, Q),
                            T2=torch.rand(M, N, P, Q),
                            M0=torch.rand(M, N, P, Q)
                        )
                        self.assertEqual(Erelax.shape, (M, N, P, Q, 3))
                        self.assertEqual(Erecovery.shape, (M, N, P, Q, 3))

                        Erelax, Erecovery = epg.relaxation_operator(
                            torch.rand(M,1,P,1),
                            T1=torch.rand(N,1,Q),
                            T2=torch.rand(P,1),
                            M0=torch.rand(Q)
                        )
                        self.assertEqual(Erelax.shape, (M,N,P,Q,3))
                        self.assertEqual(Erecovery.shape, (M,N,P,Q,3,))

class TestDephase(unittest.TestCase):
    def test_shifts_matrix(self):
        s = torch.arange(15).view(3, 5)
        s[1,0] = 0

        shifted = epg.dephase_matrix(s, 0)
        truth = torch.tensor([
            [0, 1, 2, 3, 4],
            [0, 6, 7, 8, 9],
            [10, 11, 12, 13, 14]
        ])
        self.assertTrue(torch.allclose(shifted, truth, atol=_atol))

        shifted = epg.dephase_matrix(s, 1)
        truth = torch.tensor([
            [6, 0, 1, 2, 3],
            [6, 7, 8, 9, 0],
            [10, 11, 12, 13, 14]
        ])
        self.assertTrue(torch.allclose(shifted, truth, atol=_atol))

        shifted = epg.dephase_matrix(s, 2)
        truth = torch.tensor([
            [7, 6, 0, 1, 2],
            [7, 8, 9, 0, 0],
            [10, 11, 12, 13, 14]
        ])
        self.assertTrue(torch.allclose(shifted, truth, atol=_atol))

        shifted = epg.dephase_matrix(s, -1)
        truth = torch.tensor([
            [1, 2, 3, 4, 0],
            [1, 0, 6, 7, 8],
            [10, 11, 12, 13, 14]
        ])
        self.assertTrue(torch.allclose(shifted, truth, atol=_atol))

        shifted = epg.dephase_matrix(s, -3)
        truth = torch.tensor([
            [3, 4, 0, 0, 0],
            [3, 2, 1, 0, 6],
            [10, 11, 12, 13, 14]
        ])
        self.assertTrue(torch.allclose(shifted, truth, atol=_atol))
    
    def test_shape_matrix(self):
        for _ in range(10):
            sz = torch.randint(2, 5, (5,))
            sz = sz[1:(sz[0]+1)]
            sz[-2] = 3
            rng = range(-sz[-1], sz[-1]+1)
            tmp = torch.complex(torch.randn(*sz), torch.randn(*sz)) # type: ignore

            sz[-1] = sz[-1]*2 + 1
            s = torch.zeros(*sz, dtype=torch.cfloat) # type: ignore
            s[...,:tmp.shape[-1]] = tmp
            s[...,1,0] = torch.conj(s[...,0,0])

            for i in rng:
                shifted = epg.dephase_matrix(epg.dephase_matrix(s, i), -i)
                self.assertEqual(shifted.shape, s.shape)
                self.assertTrue(torch.allclose(shifted, s, atol=_atol))

    def test_inversion(self):
        tmp = torch.arange(15).view(3, 5)
        tmp[1,0] = 0

        s = torch.zeros(3, 10, dtype=torch.cfloat)
        s[0,0:5] = tmp[0,:]
        s[1,0:5] = 1j*tmp[1,:]

        for i in range(-4, 5):
            self.assertTrue(
                torch.allclose(s,
                    epg.dephase_matrix(epg.dephase_matrix(s, i), -i),
                    atol=_atol)
                    )

    def test_shifts_vector(self):
        for _ in range(10):
            sz = torch.randint(2, 5, (5,))
            sz[-1] = 9
            F_states = torch.complex(torch.randn(*sz), torch.randn(*sz))

            sz[-1] = 5
            Z_states = torch.complex(torch.randn(*sz), torch.randn(*sz))

            for i in range(-4, 5):
                print(i)
                F_shifted = epg.dephase_vector(F_states, i)
                # Compute matrix version
                S = epg.vectors_to_matrix(F_states, Z_states)
                # Shift matrix
                S = epg.dephase_matrix(S, i)
                # Convert back to vectors
                F_from_matrix, _ = epg.matrix_to_vectors(S)
                # Compare
                self.assertTrue(torch.allclose(F_from_matrix, F_shifted, atol=_atol))

class TestEPG(unittest.TestCase):
    def test_tse(self):
        """Compare to Weigel 2015 p.281: Simulating Turbo Spin Echo Sequences"""
        state = torch.zeros(3, 7, dtype=torch.cfloat)
        state[2,0] = 1

        Ty90 = epg.excitation_operator(90, 90)
        Tx120 = epg.excitation_operator(120, 0)

        state = Ty90 @ state
        truth = torch.tensor([
            [1, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0]
        ], dtype=torch.cfloat)
        self.assertTrue(torch.allclose(state, truth, atol=_atol))

        state = epg.dephase_matrix(state, 1)
        truth = torch.tensor([
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0]
        ], dtype=torch.cfloat)
        self.assertTrue(torch.allclose(state, truth, atol=_atol))

        state = Tx120 @ state
        truth = torch.tensor([
            [0, 0.25, 0, 0, 0, 0, 0],
            [0, 0.75, 0, 0, 0, 0, 0],
            [0, -0.433012701892219j, 0, 0, 0, 0, 0]
        ], dtype=torch.cfloat)
        self.assertTrue(torch.allclose(state, truth, atol=_atol))

        state = epg.dephase_matrix(state, 1)
        truth = torch.tensor([
            [0.75, 0, 0.25, 0, 0, 0, 0],
            [0.75, 0, 0, 0, 0, 0, 0],
            [0, -0.433012701892219j, 0, 0, 0, 0, 0]
        ], dtype=torch.cfloat)
        self.assertTrue(torch.allclose(state, truth, atol=_atol))

        state = Tx120 @ epg.dephase_matrix(state)
        truth = torch.tensor([
            [0, -0.1875, 0, 0.0625, 0, 0, 0],
            [0, 0.9375, 0, 0.1875, 0, 0, 0],
            [0, -0.108253175473055j, 0, -0.108253175473055j, 0, 0, 0]
        ], dtype=torch.cfloat)
        self.assertTrue(torch.allclose(state, truth, atol=_atol))

        state = epg.dephase_matrix(state, 1)
        truth = torch.tensor([
            [0.9375, 0, -0.1875, 0, 0.0625, 0, 0],
            [0.9375, 0, 0.1875, 0, 0, 0, 0],
            [0, -0.108253175473055j, 0, -0.108253175473055j, 0, 0, 0]
        ], dtype=torch.cfloat)
        self.assertTrue(torch.allclose(state, truth, atol=_atol))

        state = epg.dephase_matrix(Tx120 @ epg.dephase_matrix(state))
        truth = torch.tensor([
            [0.84375, 0, 0.28125, 0, -0.140625, 0, 0.015625],
            [0.84375, 0, -0.046875, 0, 0.046875, 0, 0],
            [0, -0.270632938682637j, 0, 0.135316469341319j, 0, -0.0270632938682637j, 0]
        ], dtype=torch.cfloat)
        self.assertTrue(torch.allclose(state, truth, atol=_atol))
    
    def test_fast_gre(self):
        """Weigel 2015: Simulating a Rapid Gradient Echo Sequence"""
        state = torch.zeros(3, 3, dtype=torch.cfloat)
        state[2,0] = 1

        Tx30 = epg.excitation_operator(30, 0)

        Erelax = torch.zeros(3, dtype=torch.float)
        Erelax[0] = 0.9
        Erelax[1] = 0.9
        Erelax[2] = 0.99

        Erecovery = torch.zeros(3, dtype=torch.float)
        Erecovery[2] = 0.01

        state = Tx30 @ state
        truth = torch.tensor([
            [-0.5j, 0, 0],
            [0.5j, 0, 0],
            [0.866025403784439, 0, 0]
        ], dtype=torch.cfloat)
        self.assertTrue(torch.allclose(state, truth, atol=_atol))

        state = Erelax[:,None] * epg.dephase_matrix(state, 1)
        state[:,0:1] = Erecovery[:,None] + state[:,0:1]
        truth = torch.tensor([
            [0, -0.45j, 0],
            [0, 0, 0],
            [0.867365149746594, 0, 0]
        ], dtype=torch.cfloat)
        self.assertTrue(torch.allclose(state, truth, atol=_atol))

        state = Tx30 @ state
        truth = torch.tensor([
            [-0.433682574873297j, -0.419855715851499j, 0],
            [+0.433682574873297j, -0.0301442841485013j, 0],
            [0.751160254037844, -0.1125, 0]
        ], dtype=torch.cfloat)
        self.assertTrue(torch.allclose(state, truth, atol=_atol)) 

        state = Erelax[:,None] * epg.dephase_matrix(state, 1)
        state[:,0:1] = Erecovery[:,None] + state[:,0:1]
        state = Tx30 @ state
        truth = torch.tensor([
            [-0.353329181482384j, -0.308480715851499j, -0.352557644266349j],
            [0.353329181482384j, -0.0818336015344687j, -0.0253125000000000j],
            [0.666243805591516, -0.194032158692984, -0.0944675360665872]
        ], dtype=torch.cfloat)
        self.assertTrue(torch.allclose(state, truth, atol=_atol))

    def test_hyperecho(self):
        """Weigel 2015: Hyperecho"""
        nPulse = 15
        phaseShifts = [0, 90, -90, 180]

        flip_angles = [fibonacci(i) for i in range(1, nPulse)]
        flip_angles = [90,] + flip_angles + [180,] + [-x for x in reversed(flip_angles)]
        phase_angles = [i for i in range(1, nPulse)]
        phase_angles = [90,] + phase_angles + [0,] + [-x for x in reversed(phase_angles)]

        for phase in phaseShifts:
            state = torch.zeros(3, 2*nPulse + 2, dtype=torch.cfloat)
            state[2,0] = 1
            # Apply excitations
            for fa, phi in zip(flip_angles, phase_angles):
                state = epg.dephase_matrix(epg.excitation_operator(fa, phase + phi) @ state)
            
            truth = torch.zeros(3, 2*nPulse + 2, dtype=torch.cfloat)
            # I have some uncertainty about the phase shifts, but these match the results from the sample code provided by Weigel 2015 so I will accept them.
            if phase == 0:
                truth[0,0] = 1
            elif phase == 90:
                truth[0,0] = torch.tensor([1j,], dtype=torch.cfloat)
            elif phase == 180:
                truth[0,0] = -1
            elif phase == -90:
                truth[0,0] = torch.tensor([-1j,], dtype=torch.cfloat)
            truth[1,0] = torch.conj(state[0,0])

            self.assertTrue(torch.allclose(state, truth, atol=_atol))

class TestRepresentations(unittest.TestCase):
    def test_matrix_to_vectors(self):
        S = 1j*torch.arange(15).view(3, 5)

        with self.assertRaises(ValueError):
            F, Z = epg.matrix_to_vectors(S)
        S[1,0] = 0

        Ftruth = torch.tensor([0j, 1j, 2j, 3j, 4j, -9j, -8j, -7j, -6j], dtype=torch.cfloat)
        Ztruth = torch.tensor([10j, 11j, 12j, 13j, 14j], dtype=torch.cfloat)
        F, Z = epg.matrix_to_vectors(S)
        self.assertTrue(torch.allclose(F, Ftruth, atol=_atol))
        self.assertTrue(torch.allclose(Z, Ztruth, atol=_atol))
    
    def test_vectors_to_matrix(self):
        F = torch.tensor([0j, 1j, 2j, 3j, 4j, -9j, -8j, -7j, -6j], dtype=torch.cfloat)
        Z = torch.tensor([10j, 11j, 12j, 13j, 14j], dtype=torch.cfloat)

        S = epg.vectors_to_matrix(F, Z)
        truth = 1j*torch.arange(15).view(3, 5)
        truth[1,0] = 0
        self.assertTrue(torch.allclose(S, truth, atol=_atol))

    def test_vectors_to_matrix_multi(self):
        N, K = 2, 4
        F = torch.arange(2*N*(K*2-1), dtype=torch.float).view(N, K*2-1, 2)
        F = torch.complex(F[:,:,0], F[:,:,1])

        Z = torch.arange(2*N*K, dtype=torch.float).view(N, K, 2)
        Z = torch.complex(Z[:,:,0], Z[:,:,1])

        S = epg.vectors_to_matrix(F, Z)
        truth = torch.tensor([[[ 0.+1.j,  2.+3.j,  4.+5.j,  6.+7.j],
         [ 0.-1.j, 12.-13.j, 10.-11.j,  8.-9.j],
         [ 0.+1.j,  2.+3.j,  4.+5.j,  6.+7.j]],
        [[14.+15.j, 16.+17.j, 18.+19.j, 20.+21.j],
         [14.-15.j, 26.-27.j, 24.-25.j, 22.-23.j],
         [ 8.+9.j, 10.+11.j, 12.+13.j, 14.+15.j]]])
        
        self.assertTrue(torch.allclose(S, truth, atol=_atol))

    def test_matrix_to_vectors_multi(self):
        S = torch.tensor([[[ 0.+1.j,  2.+3.j,  4.+5.j,  6.+7.j],
            [ 0.-1.j, 12.-13.j, 10.-11.j,  8.-9.j],
            [ 0.+1.j,  2.+3.j,  4.+5.j,  6.+7.j]],
        [[14.+15.j, 16.+17.j, 18.+19.j, 20.+21.j],
            [14.-15.j, 26.-27.j, 24.-25.j, 22.-23.j],
            [ 8.+9.j, 10.+11.j, 12.+13.j, 14.+15.j]]])

        N, K = 2, 4
        Ftruth = torch.arange(2*N*(K*2-1), dtype=torch.float).view(N, K*2-1, 2)
        Ftruth = torch.complex(Ftruth[:,:,0], Ftruth[:,:,1])

        Ztruth = torch.arange(2*N*K, dtype=torch.float).view(N, K, 2)
        Ztruth = torch.complex(Ztruth[:,:,0], Ztruth[:,:,1])

        F, Z = epg.matrix_to_vectors(S)

        self.assertTrue(torch.allclose(Z, Ztruth, atol=_atol))
        self.assertTrue(torch.allclose(F, Ftruth, atol=_atol))

from functools import lru_cache
 
# Function for nth Fibonacci number
@lru_cache(None)
def fibonacci(num: int) -> int:
    # check if num between 1, 0
    # it will return num
    if num < 2:
        return num
 
    # return the fibonacci of num - 1 & num - 2
    return fibonacci(num - 1) + fibonacci(num - 2)