"""Run one mini-protein binder-design trajectory."""

import argparse
import json
import logging
import random
from pathlib import Path

import numpy as np
from Bio import SeqIO
from omegaconf import OmegaConf

from src.core.optimizer import SingleTrajectoryOptimizer
from src.utils.helpers import (
    load_config,
    setup_logging,
    validate_standard_sequence,
)

logger = logging.getLogger(__name__)


def load_target_fasta(path):
    try:
        record = next(SeqIO.parse(str(path), "fasta"))
    except StopIteration as exc:
        raise ValueError(f"Target FASTA is empty: {path}") from exc
    return str(record.id), validate_standard_sequence(
        str(record.seq), "target sequence"
    )


def write_status(run_dir, status, result=None):
    payload = {"status": status}
    if result:
        payload["candidate_id"] = result["id"]
        payload["ipsae_min"] = result["ipsae_min"]
        payload["iptm"] = result["iptm"]
        payload["composite_affinity"] = result["composite_affinity"]
        payload["binder_plddt"] = result["binder_plddt"]
    (run_dir / "run_status.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser(
        description="ESMFold2 mini-protein binder design"
    )
    parser.add_argument("--config", default="config/binder.yaml")
    args, overrides = parser.parse_known_args()

    config = load_config(args.config, overrides)
    run_dir = Path(config.paths.run_output_dir)
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Run directory is not empty: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(config.log_level, run_dir / "logs")
    OmegaConf.save(config, run_dir / "run_config.yaml")

    seed = int(config.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)

    target_id, target_sequence = load_target_fasta(
        config.paths.target_fasta
    )
    hotspot = config.get("hotspot", {})
    hotspot_string = (
        str(hotspot.get("binding_region", "")).strip()
        if hotspot.get("enabled", False)
        else ""
    )
    if hotspot.get("enabled", False) and not hotspot_string:
        raise ValueError("hotspot.enabled requires hotspot.binding_region")

    result = SingleTrajectoryOptimizer(config).optimize_binder(
        target_id,
        target_sequence,
        config.paths.target_cif_path,
        hotspot_string or None,
    )
    if result is None:
        write_status(run_dir, "valid_no_hit")
        logger.warning("No strict final ProteinMPNN candidate passed all filters.")
        raise SystemExit(2)

    write_status(run_dir, "success", result)
    logger.info("Strict final binder saved at %s", result["complex_pdb_path"])


if __name__ == "__main__":
    main()
