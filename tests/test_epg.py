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

class TestRecovery(unittest.TestCase):
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
        truth_relax = torch.tensor([1, 1, 1], dtype=torch.float)
        truth_recovery = torch.tensor([1, 1, 1], dtype=torch.float)

        self.assertEqual(Erelax.shape, (3,))
        self.assertTrue(torch.allclose(Erelax, truth_relax, atol=_atol))

        self.assertEqual(Erecovery.shape, (3,))
        self.assertTrue(torch.allclose(Erecovery, truth_recovery, atol=_atol))