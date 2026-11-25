import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.database import BindSculptDatabase


class DatabaseTests(unittest.TestCase):
    def test_atomic_union_schema_paths_and_deduplication(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            pdb_path = run_dir / "structures" / "a.pdb"
            pdb_path.parent.mkdir()
            pdb_path.touch()
            database = BindSculptDatabase(run_dir)
            first = {
                "id": "a",
                "binder_sequence": "ACDE",
                "esmfold2_status": "success",
                "iptm": 0.8,
                "complex_pdb_path": str(pdb_path),
                "fixed_interface_positions": {"B": [1, 4]},
                "concat_embedding": np.array([1.0, 2.0]),
            }
            second = {
                "id": "b",
                "binder_sequence": "ACDF",
                "esmfold2_status": "success",
                "iptm": 0.9,
                "new_metric": 7,
                "concat_embedding": np.array([3.0, 4.0]),
            }
            database.save_evaluated_batch([first])
            database.save_evaluated_batch([second])
            database.save_evaluated_batch([{
                **first,
                "iptm": 0.85,
                "concat_embedding": np.array([5.0, 6.0]),
            }])

            frame = pd.read_csv(database.csv_path)
            self.assertEqual(len(frame), 2)
            self.assertIn("new_metric", frame.columns)
            records = {row["id"]: row for row in database.get_all_training_data()}
            self.assertEqual(
                records["a"]["fixed_interface_positions"], {"B": [1, 4]}
            )
            self.assertTrue(Path(records["a"]["complex_pdb_path"]).is_absolute())
            np.testing.assert_array_equal(
                records["a"]["concat_embedding"], np.array([5.0, 6.0])
            )

            database.save_evaluated_batch([{
                **first,
                "id": "c",
                "concat_embedding": np.array([7.0, 8.0]),
            }])
            frame = pd.read_csv(database.csv_path)
            self.assertEqual(set(frame["id"]), {"b", "c"})
            self.assertTrue(frame["binder_sequence"].is_unique)


if __name__ == "__main__":
    unittest.main()

