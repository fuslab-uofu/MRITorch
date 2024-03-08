import unittest

import torch

from mritorch import epg

_atol = 1e-6  # Absolute tolerance for allclose

class TestExcitation(unittest.TestCase):
    def test_epg_Tx180(self):
        Tx180 = epg.excitation_operator(180)
        truth = torch.tensor([
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, -1]
        ], dtype=torch.cfloat)

        self.assertEqual(Tx180.shape, (3, 3))
        self.assertTrue(Tx180.dtype == torch.cfloat)
        self.assertTrue(torch.allclose(Tx180, truth, atol=_atol))

    def test_epg_Tx90(self):
        Tx90 = epg.excitation_operator(90)
        truth = torch.tensor([
            [0.5, 0.5, -1j],
            [0.5, 0.5, +1j],
            [-0.5j, +0.5j, 0]
        ], dtype=torch.cfloat)

        self.assertEqual(Tx90.shape, (3, 3))
        self.assertTrue(Tx90.dtype == torch.cfloat)
        self.assertTrue(torch.allclose(Tx90, truth, atol=_atol))

    def test_epg_Ty90(self):
        Ty90 = epg.excitation_operator(90, 90)
        truth = torch.tensor([
            [0.5, -0.5, 1],
            [-0.5, 0.5, 1],
            [-0.5, -0.5, 0]
        ], dtype=torch.cfloat)

        self.assertEqual(Ty90.shape, (3, 3))
        self.assertTrue(Ty90.dtype == torch.cfloat)
        self.assertTrue(torch.allclose(Ty90, truth, atol=_atol))

    def test_epg_Ty180(self):
        Ty90 = epg.excitation_operator(180, 90)
        truth = torch.tensor([
            [0, -1, 0],
            [-1, 0, 0],
            [0, 0, -1]
        ], dtype=torch.cfloat)

        self.assertEqual(Ty90.shape, (3, 3))
        self.assertTrue(Ty90.dtype == torch.cfloat)
        self.assertTrue(torch.allclose(Ty90, truth, atol=_atol))

    def test_epg_multi(self):
        flip_angle = torch.tensor([45, 90, 180, 45, 90, 180, 45, 90, 180])
        phase_angle = torch.tensor([0, 0, 0, 45, 45, 45, 90, 90, 90])
        Tx = epg.excitation_operator(flip_angle, phase_angle=phase_angle)
        truth = torch.stack([
            epg.excitation_operator(flip_angle[i], phase_angle[i])
            for i in range(flip_angle.shape[0])
        ])

        self.assertEqual(Tx.shape, (9, 3, 3))
        self.assertTrue(Tx.dtype == torch.cfloat)
        self.assertTrue(torch.allclose(Tx, truth, atol=_atol))
