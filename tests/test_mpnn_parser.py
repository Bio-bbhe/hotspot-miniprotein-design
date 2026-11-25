import tempfile
import unittest
from pathlib import Path

from src.models.mpnn_mutator import MPNNMutator


class MPNNParserTests(unittest.TestCase):
    def test_skips_native_invalid_and_duplicate_records(self):
        fasta = (
            ">native, score=0.5\nAAAA\n"
            ">T=0.1, sample=1, score=-0.2\nAAAV\n"
            ">T=0.1, sample=2, score=-0.1\nAAXV\n"
            ">T=0.1, sample=3, score=-0.2\nAAAV\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "output.fa"
            path.write_text(fasta, encoding="utf-8")
            records = MPNNMutator._parse_fasta(
                path, 10, {"A": [], "B": [1]}
            )
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["binder_sequence"], "AAAV")
        self.assertEqual(records[0]["mpnn_score"], -0.2)
        self.assertEqual(records[0]["mutation_type"], "mpnn_scaffold_redesign")


if __name__ == "__main__":
    unittest.main()

