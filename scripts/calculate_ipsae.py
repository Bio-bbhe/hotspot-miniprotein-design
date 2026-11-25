#!/usr/bin/env python3
"""Calculate directional ipSAE from an ESMFold2 PAE matrix and structure."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.ipsae import calculate_ipsae  # noqa: E402


def protein_chain_ids(pdb_path):
    structure = PDBParser(QUIET=True).get_structure("complex", pdb_path)
    return [
        chain.id
        for chain in structure[0]
        for residue in chain
        if is_aa(residue, standard=True)
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pae", help="ESMFold2 protein-token PAE matrix (.npy)")
    parser.add_argument("structure", help="Matching ESMFold2 complex PDB")
    parser.add_argument("--target-chains", default="A")
    parser.add_argument("--binder-chains", default="B")
    parser.add_argument("--cutoff", type=float, default=10.0)
    parser.add_argument("--output", help="Optional JSON output path")
    args = parser.parse_args()

    scores = calculate_ipsae(
        np.load(args.pae),
        protein_chain_ids(args.structure),
        tuple(args.target_chains.split(",")),
        tuple(args.binder_chains.split(",")),
        args.cutoff,
    )
    scores = {key: round(value, 6) for key, value in scores.items()}
    text = json.dumps(scores, indent=2)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
