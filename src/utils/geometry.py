"""Scaffold placement and hotspot-contact checks for mini-protein binders."""

import copy
import logging
from pathlib import Path

import numpy as np
from Bio.PDB import MMCIFParser, Model, PDBIO, PDBParser, Structure
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

logger = logging.getLogger(__name__)


class HotspotGeometry:
    def __init__(self):
        self.pdb_parser = PDBParser(QUIET=True)
        self.writer = PDBIO()

    def _load(self, name, path):
        parser = (
            MMCIFParser(QUIET=True)
            if str(path).lower().endswith(".cif")
            else self.pdb_parser
        )
        return parser.get_structure(name, str(path))

    @staticmethod
    def _coords(chain, atom_names=None):
        names = atom_names or {"CA", "C", "N", "O", "CB"}
        return np.asarray([
            atom.coord
            for residue in chain
            if residue.id[0] == " "
            for atom in residue
            if atom.name in names
        ])

    @staticmethod
    def _parse_positions(value):
        positions = []
        for block in str(value).split(","):
            block = block.strip()
            if not block:
                continue
            if "-" in block:
                start, end = map(int, block.split("-", 1))
                if end < start:
                    raise ValueError(f"Invalid hotspot range: {block}")
                positions.extend(range(start, end + 1))
            else:
                positions.append(int(block))
        return sorted(set(positions))

    @staticmethod
    def _pdb_residue_numbers(structure, chain_id, positions):
        residues = [
            residue.id[1]
            for residue in structure[0][chain_id]
            if residue.id[0] == " "
        ]
        if not positions or not residues:
            raise ValueError("Target chain or hotspot is empty")
        if all(1 <= position <= len(residues) for position in positions):
            return [residues[position - 1] for position in positions]
        if set(positions).issubset(residues):
            return positions
        raise ValueError("Hotspot positions are outside the target chain")

    @staticmethod
    def _residue_com(chain, residue_numbers):
        coords = [
            residue["CA"].coord
            for residue in chain
            if residue.id[0] == " "
            and residue.id[1] in residue_numbers
            and "CA" in residue
        ]
        if not coords:
            raise ValueError("Configured hotspot residues were not found")
        return np.mean(coords, axis=0)

    def place_scaffolds_diverse(
        self,
        target_pdb,
        scaffold_pdb,
        out_dir,
        prefix,
        hotspot_residues_str=None,
        target_chain_id="A",
        scaffold_chain_id="B",
        num_poses=5,
        max_attempts=1000,
        clash_cutoff=2.8,
        min_contacts=15,
        contact_dist=6.0,
        rmsd_threshold=4.0,
    ):
        output_dir = Path(out_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        target_structure = self._load("target", target_pdb)
        scaffold_structure = self._load("scaffold", scaffold_pdb)
        target_chain = target_structure[0][target_chain_id]
        scaffold_chain = copy.deepcopy(
            next(scaffold_structure[0].get_chains())
        )
        scaffold_chain.id = scaffold_chain_id

        target_atoms = self._coords(target_chain)
        scaffold_atoms = self._coords(scaffold_chain)
        scaffold_ca = self._coords(scaffold_chain, {"CA"})
        if not len(target_atoms) or not len(scaffold_atoms):
            raise ValueError("Target or scaffold contains no backbone atoms")

        target_tree = cKDTree(target_atoms)
        target_com = target_atoms.mean(axis=0)
        scaffold_com = scaffold_atoms.mean(axis=0)
        hotspot = bool(hotspot_residues_str)
        if hotspot:
            positions = self._pdb_residue_numbers(
                target_structure,
                target_chain_id,
                self._parse_positions(hotspot_residues_str),
            )
            placement_center = self._residue_com(target_chain, positions)

        saved_paths = []
        saved_ca = []
        for _ in range(max_attempts):
            if len(saved_paths) >= num_poses:
                break

            rotation = Rotation.random().as_matrix()
            if hotspot:
                surface = placement_center
            else:
                surface = target_atoms[np.random.randint(len(target_atoms))]
            outward = surface - target_com
            if np.linalg.norm(outward) < 1e-6:
                continue
            outward /= np.linalg.norm(outward)
            translation = (
                surface
                + outward * np.random.uniform(6.0, 10.0)
                + np.random.normal(0.0, 2.0, size=3)
                - scaffold_com @ rotation.T
            )

            transformed_atoms = scaffold_atoms @ rotation.T + translation
            distances, _ = target_tree.query(transformed_atoms, k=1)
            if np.any(distances < clash_cutoff):
                continue
            if np.sum(distances < contact_dist) < min_contacts:
                continue

            transformed_ca = scaffold_ca @ rotation.T + translation
            if any(
                np.sqrt(np.mean(np.sum((transformed_ca - prior) ** 2, axis=1)))
                < rmsd_threshold
                for prior in saved_ca
            ):
                continue
            saved_ca.append(transformed_ca)

            placed_chain = copy.deepcopy(scaffold_chain)
            for atom in placed_chain.get_atoms():
                atom.coord = atom.coord @ rotation.T + translation

            merged = Structure.Structure(f"pose_{len(saved_paths)}")
            model = Model.Model(0)
            model.add(copy.deepcopy(target_chain))
            model.add(placed_chain)
            merged.add(model)
            path = output_dir / f"{prefix}_pose_{len(saved_paths)}.pdb"
            self.writer.set_structure(merged)
            self.writer.save(str(path))
            saved_paths.append(str(path))

        if not saved_paths:
            logger.warning("No valid poses found for %s", scaffold_pdb)
        return saved_paths

    def hotspot_drifted(
        self,
        complex_path,
        hotspot_residues_str,
        target_chain="A",
        binder_chain="B",
        distance_threshold=5.0,
        min_contacts=5,
    ):
        path = Path(complex_path)
        if not path.exists():
            return True

        structure = self._load("complex", path)
        model = structure[0]
        if target_chain not in model or binder_chain not in model:
            return True
        positions = self._pdb_residue_numbers(
            structure,
            target_chain,
            self._parse_positions(hotspot_residues_str),
        )
        hotspot_atoms = np.asarray([
            atom.coord
            for residue in model[target_chain]
            if residue.id[0] == " " and residue.id[1] in positions
            for atom in residue
            if atom.element != "H"
        ])
        binder_atoms = np.asarray([
            atom.coord
            for residue in model[binder_chain]
            for atom in residue
            if atom.element != "H"
        ])
        if not len(hotspot_atoms) or not len(binder_atoms):
            return True

        contacts = cKDTree(binder_atoms).query_ball_point(
            hotspot_atoms, distance_threshold
        )
        unique_binder_atoms = {index for group in contacts for index in group}
        return len(unique_binder_atoms) < min_contacts

    @staticmethod
    def _heavy_coords(residue):
        return np.asarray([
            atom.coord for atom in residue if atom.element != "H"
        ])

    @staticmethod
    def _has_contact(tree, residue, distance):
        return any(tree.query_ball_point(
            HotspotGeometry._heavy_coords(residue), distance
        ))

    def hotspot_contact_metrics(
        self,
        complex_path,
        hotspot_residues_str,
        target_chain="A",
        binder_chain="B",
        distance_threshold=5.0,
    ):
        empty = {
            "hotspot_residue_contacts": 0,
            "hotspot_contact_fraction": 0.0,
        }
        path = Path(complex_path)
        if not path.exists():
            return empty

        structure = self._load("complex", path)
        model = structure[0]
        if target_chain not in model or binder_chain not in model:
            return empty

        target_residues = [
            residue for residue in model[target_chain]
            if residue.id[0] == " "
        ]
        binder_residues = [
            residue for residue in model[binder_chain]
            if residue.id[0] == " "
        ]
        positions = set(self._pdb_residue_numbers(
            structure,
            target_chain,
            self._parse_positions(hotspot_residues_str),
        ))
        hotspot_residues = [
            residue for residue in target_residues
            if residue.id[1] in positions
        ]
        if not hotspot_residues or not binder_residues:
            return empty

        binder_atoms = np.concatenate([
            self._heavy_coords(residue) for residue in binder_residues
        ])
        binder_tree = cKDTree(binder_atoms)
        target_interface = sum(
            self._has_contact(binder_tree, residue, distance_threshold)
            for residue in target_residues
        )
        hotspot_contacts = sum(
            self._has_contact(binder_tree, residue, distance_threshold)
            for residue in hotspot_residues
        )
        return {
            "hotspot_residue_contacts": hotspot_contacts,
            "hotspot_contact_fraction": (
                hotspot_contacts / target_interface
                if target_interface else 0.0
            ),
        }
