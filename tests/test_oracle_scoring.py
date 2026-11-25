import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from src.models.esmfold2_oracle import ESMFold2Oracle


class OracleScoringTests(unittest.TestCase):
    def test_multiseed_complex_selection_uses_ipsae_only(self):
        oracle = ESMFold2Oracle.__new__(ESMFold2Oracle)
        oracle.config = {"ipsae_pae_cutoff": 10.0}
        result = SimpleNamespace(
            plddt=np.array([0.95]),
            pae=np.zeros((2, 2)),
            iptm=0.99,
        )
        with patch(
            "src.models.esmfold2_oracle.calculate_ipsae",
            return_value={"ipsae_min": 0.61},
        ):
            self.assertEqual(
                oracle._sample_score(result, "complex", 1, 1, True),
                0.61,
            )
            self.assertGreater(
                oracle._sample_score(result, "complex", 1, 1, False),
                0.61,
            )


if __name__ == "__main__":
    unittest.main()

