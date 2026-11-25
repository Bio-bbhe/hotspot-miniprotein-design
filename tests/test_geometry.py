import tempfile
import unittest
from pathlib import Path

from src.utils.geometry import HotspotGeometry


def atom_line(serial, chain, residue, x):
    return (
        f"ATOM  {serial:5d}  CA  ALA {chain}{residue:4d}    "
        f"{x:8.3f}{0.0:8.3f}{0.0:8.3f}"
        "  1.00 80.00           C\n"
    )


class GeometryTests(unittest.TestCase):
    def test_hotspots_use_sequence_positions_with_pdb_numbering(self):
        pdb = (
            atom_line(1, "A", 101, 0.0)
            + atom_line(2, "A", 102, 30.0)
            + atom_line(3, "B", 1, 4.0)
            + "END\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "complex.pdb"
            path.write_text(pdb, encoding="utf-8")
            geometry = HotspotGeometry()
            self.assertFalse(geometry.hotspot_drifted(path, "1", min_contacts=1))
            self.assertTrue(geometry.hotspot_drifted(path, "2", min_contacts=1))
            metrics = geometry.hotspot_contact_metrics(path, "1")
            wrong = geometry.hotspot_contact_metrics(path, "2")

        self.assertEqual(metrics["hotspot_residue_contacts"], 1)
        self.assertEqual(metrics["hotspot_contact_fraction"], 1.0)
        self.assertEqual(wrong["hotspot_residue_contacts"], 0)
        self.assertEqual(
            wrong["hotspot_contact_fraction"], 0.0
        )


if __name__ == "__main__":
    unittest.main()

