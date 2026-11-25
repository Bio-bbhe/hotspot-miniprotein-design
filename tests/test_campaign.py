import tempfile
import unittest
from pathlib import Path

import pandas as pd

from campaign import aggregate_results


class CampaignAggregationTests(unittest.TestCase):
    def test_deduplicates_sequences_and_ranks_full_results(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [
                {
                    "id": "a",
                    "binder_sequence": "AAAA",
                    "mutation_type": "final_mpnn_redesign",
                    "composite_affinity": 0.82,
                    "ipsae_min": 0.70,
                    "iptm": 0.85,
                    "binder_plddt": 0.90,
                    "prediction_variant": "full",
                    "prediction_seed_count": 5,
                    "hotspot_satisfied": True,
                },
                {
                    "id": "ignored",
                    "binder_sequence": "CCCC",
                    "mutation_type": "point_single",
                    "composite_affinity": 0.99,
                    "ipsae_min": 0.99,
                    "iptm": 0.99,
                    "binder_plddt": 0.99,
                    "prediction_variant": "fast",
                },
            ]
            for name, frame_rows in (
                ("campaign_001", rows),
                (
                    "campaign_002",
                    [
                        {
                            **rows[0],
                            "id": "b",
                            "composite_affinity": 0.88,
                        },
                        {
                            **rows[0],
                            "id": "c",
                            "binder_sequence": "DDDD",
                            "composite_affinity": 0.84,
                        },
                    ],
                ),
            ):
                path = root / name
                path.mkdir()
                pd.DataFrame(frame_rows).to_csv(
                    path / "final_post_processed_candidates.csv",
                    index=False,
                )

            master = aggregate_results(root)

            self.assertEqual(list(master["id"]), ["b", "c"])
            self.assertEqual(list(master["rank"]), [1, 2])
            self.assertEqual(
                master.iloc[0]["campaign_source"], "campaign_002"
            )
            self.assertTrue((root / "master_candidates.csv").exists())


if __name__ == "__main__":
    unittest.main()
