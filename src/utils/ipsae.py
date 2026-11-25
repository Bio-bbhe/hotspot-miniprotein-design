"""Directional Dunbrack ipSAE using protein-token PAE."""

import numpy as np


def _d0res(length):
    length = max(float(length), 26.0)
    return max(1.0, 1.24 * (length - 15.0) ** (1.0 / 3.0) - 1.8)


def _directional_ipsae(pae, aligned, partner, cutoff):
    block = pae[np.ix_(aligned, partner)]
    best = 0.0
    for row in block:
        selected = row[row < cutoff]
        if selected.size:
            d0 = _d0res(selected.size)
            best = max(best, float(np.mean(1.0 / (1.0 + (selected / d0) ** 2))))
    return best


def calculate_ipsae(pae, chain_ids, group_a=("A",), group_b=("B",), cutoff=10.0):
    """Return directional d0res ipSAE and their stringent minimum."""
    matrix = np.asarray(pae, dtype=float)
    chains = np.asarray(chain_ids, dtype=str)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("PAE must be a square matrix")
    if matrix.shape[0] != chains.size:
        raise ValueError("PAE size does not match the protein-token chain map")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("PAE contains non-finite values")
    if cutoff <= 0:
        raise ValueError("PAE cutoff must be positive")

    indices_a = np.flatnonzero(np.isin(chains, tuple(group_a)))
    indices_b = np.flatnonzero(np.isin(chains, tuple(group_b)))
    if not indices_a.size or not indices_b.size:
        raise ValueError("Both chain groups must contain protein residues")
    if np.intersect1d(indices_a, indices_b).size:
        raise ValueError("Chain groups must not overlap")

    a_to_b = _directional_ipsae(matrix, indices_a, indices_b, cutoff)
    b_to_a = _directional_ipsae(matrix, indices_b, indices_a, cutoff)
    return {
        "group_a_to_b": a_to_b,
        "group_b_to_a": b_to_a,
        "ipsae_min": min(a_to_b, b_to_a),
        "ipsae_max": max(a_to_b, b_to_a),
    }
