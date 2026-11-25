import random
import unittest

from src.core.candidate_pool import limit_embedding_pool


class CandidatePoolTests(unittest.TestCase):
    def test_caps_pool_and_retains_best_scores(self):
        random.seed(7)
        candidates = [
            {"id": str(index), "mpnn_score": float(index)}
            for index in range(500)
        ]
        selected = limit_embedding_pool(candidates, 200)
        selected_ids = {int(item["id"]) for item in selected}

        self.assertEqual(len(selected), 200)
        self.assertEqual(len(selected_ids), 200)
        self.assertTrue(set(range(160)).issubset(selected_ids))

    def test_rejects_nonpositive_limit(self):
        with self.assertRaises(ValueError):
            limit_embedding_pool([{"id": "x"}], 0)


if __name__ == "__main__":
    unittest.main()
