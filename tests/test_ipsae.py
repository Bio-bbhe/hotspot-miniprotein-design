import unittest

import numpy as np

from src.utils.ipsae import calculate_ipsae


class IpsaeTests(unittest.TestCase):
    def test_directional_d0res_minimum(self):
        pae = np.full((4, 4), 20.0)
        pae[0, 2:] = [1.0, 2.0]
        pae[2, :2] = [1.0, 1.0]
        scores = calculate_ipsae(pae, ["A", "A", "B", "B"])
        self.assertAlmostEqual(scores["group_a_to_b"], 0.35)
        self.assertAlmostEqual(scores["group_b_to_a"], 0.5)
        self.assertAlmostEqual(scores["ipsae_min"], 0.35)

    def test_rejects_mismatched_chain_map(self):
        with self.assertRaises(ValueError):
            calculate_ipsae(np.zeros((2, 2)), ["A"])
