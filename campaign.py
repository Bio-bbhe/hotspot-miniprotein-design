"""Run production design trajectories with resident models."""

import argparse
import json
import logging
import time
from pathlib import Path

import pandas as pd
from Bio import SeqIO
from omegaconf import OmegaConf

from src.core.optimizer import SingleTrajectoryOptimizer
from src.models.esm_engine import ESMEngine
from src.models.model_holder import ensure_model
from src.utils.helpers import validate_standard_sequence
from src.utils.helpers import load_config, setup_logging

logger = logging.getLogger("campaign")

MASTER_COLUMNS = [
    "rank",
    "id",
    "binder_sequence",
    "campaign_source",
    "mutation_type",
    "original_parent",
    "target_id",
    "hotspot_satisfied",
    "ipsae_min",
    "iptm",
    "binder_plddt",
    "composite_affinity",
    "delta_rmsd",
    "prediction_variant",
    "prediction_seed",
    "prediction_seed_count",
    "hotspot_residue_contacts",
    "hotspot_contact_fraction",
    "cdr_hotspot_contacts",
    "cdr_contact_fraction",
    "complex_pdb_path",
    "esmfold2_cif_path",
    "pae_path",
]


def load_target(path):
    try:
        record = next(SeqIO.parse(str(path), "fasta"))
    except StopIteration as exc:
        raise ValueError(f"Target FASTA is empty: {path}") from exc
    return str(record.id), validate_standard_sequence(
        str(record.seq), "target sequence"
    )


def validate_production_config(config):
    active = config.active_learning
    filters = config.filters.complex_filters
    if int(active.get("num_iterations", 0)) < 5:
        raise ValueError("Production campaigns require num_iterations >= 5")
    if int(active.get("embedding_pool_size", 200)) > 200:
        raise ValueError("Production embedding_pool_size cannot exceed 200")
    required = {
        "min_ipsae": 0.60,
        "min_iptm": 0.80,
        "min_binder_plddt": 0.80,
    }
    for name, minimum in required.items():
        if float(filters.get(name, 0)) < minimum:
            raise ValueError(f"Production {name} must be >= {minimum}")


def aggregate_results(base_output_dir):
    base = Path(base_output_dir)
    frames = []
    for path in sorted(base.glob(
        "campaign_*/final_post_processed_candidates.csv"
    )):
        frame = pd.read_csv(path)
        if frame.empty:
            continue
        frame["campaign_source"] = path.parent.name
        frames.append(frame)

    master = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not master.empty:
        required = {
            "mutation_type", "prediction_variant", "prediction_seed_count",
            "ipsae_min", "iptm", "binder_plddt", "hotspot_satisfied",
            "composite_affinity", "binder_sequence",
        }
        missing = required.difference(master.columns)
        if missing:
            logger.warning(
                "Ignoring candidate files missing fields: %s",
                ", ".join(sorted(missing)),
            )
            master = pd.DataFrame()
    if not master.empty:
        qualified = (
            master["mutation_type"].astype(str).str.startswith("final_")
            & master["prediction_variant"].astype(str).eq("full")
            & master["ipsae_min"].ge(0.60)
            & master["iptm"].ge(0.80)
            & master["binder_plddt"].ge(0.80)
            & master["hotspot_satisfied"].astype(str).str.lower().eq("true")
            & master["prediction_seed_count"].ge(1)
        )
        if "delta_rmsd" in master:
            qualified &= master["delta_rmsd"].le(2.0)
        master = master[qualified]
        master = master.sort_values(
            ["composite_affinity", "ipsae_min", "iptm"],
            ascending=False,
        )
        master = master.drop_duplicates(
            subset="binder_sequence", keep="first"
        ).reset_index(drop=True)
        master.insert(0, "rank", range(1, len(master) + 1))

    for column in MASTER_COLUMNS:
        if column not in master:
            master[column] = pd.Series(dtype="object")
    ordered = MASTER_COLUMNS + [
        column for column in master.columns
        if column not in MASTER_COLUMNS
    ]
    master = master[ordered]
    master.to_csv(base / "master_candidates.csv", index=False)
    return master


def write_status(run_dir, status, elapsed, result=None, error=None):
    payload = {
        "status": status,
        "elapsed_seconds": round(elapsed, 2),
    }
    if result:
        for key in (
            "id",
            "ipsae_min",
            "iptm",
            "binder_plddt",
            "composite_affinity",
            "prediction_variant",
            "prediction_seed",
        ):
            payload[key] = result.get(key)
    if error:
        payload["error"] = str(error)[:500]
    (run_dir / "run_status.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )


def run_campaign(
    config_path,
    trajectories=20,
    start=1,
    min_candidates=3,
    min_trajectories=1,
    resume=False,
    overrides=(),
):
    base_config = load_config(config_path, overrides)
    validate_production_config(base_config)
    target_id, target_sequence = load_target(
        base_config.paths.target_fasta
    )
    hotspot = base_config.get("hotspot", {})
    hotspot_string = (
        str(hotspot.get("binding_region", "")).strip()
        if hotspot.get("enabled", False)
        else None
    )
    base_output = Path(base_config.paths.base_output_dir)
    base_output.mkdir(parents=True, exist_ok=True)

    fold_cfg = base_config.models.esmfold2
    screening_variant = str(
        fold_cfg.get("screening_variant", "fast")
    )
    final_variant = str(fold_cfg.get("final_variant", "full"))
    started = time.perf_counter()
    ensure_model(base_config, variant=screening_variant)
    ensure_model(base_config, variant=final_variant)
    shared_engine = ESMEngine(base_config)
    shared_engine.load_model()
    shared_engine.cache_target_embedding(target_sequence)
    model_load_seconds = time.perf_counter() - started

    attempted = 0
    for index in range(start, start + trajectories):
        master = aggregate_results(base_output)
        completed = len(list(base_output.glob(
            "campaign_*/run_status.json"
        )))
        if completed >= min_trajectories and len(master) >= min_candidates:
            break

        run_name = f"campaign_{index:03d}"
        seed = int(base_config.get("seed", 42)) + index - start
        trajectory_overrides = list(overrides) + [
            f"run_name={run_name}",
            f"seed={seed}",
            f"models.esmfold2.seed={seed}",
        ]
        config = load_config(config_path, trajectory_overrides)
        validate_production_config(config)
        run_dir = Path(config.paths.run_output_dir)
        if run_dir.exists() and any(run_dir.iterdir()):
            if resume:
                logger.info("Skipping existing %s.", run_name)
                continue
            raise FileExistsError(
                f"Run directory is not empty: {run_dir}"
            )

        run_dir.mkdir(parents=True, exist_ok=True)
        setup_logging(config.log_level, run_dir / "logs")
        OmegaConf.save(config, run_dir / "run_config.yaml")
        attempted += 1
        trajectory_started = time.perf_counter()
        try:
            result = SingleTrajectoryOptimizer(config).optimize_binder(
                target_id,
                target_sequence,
                config.paths.target_cif_path,
                hotspot_string,
            )
            elapsed = time.perf_counter() - trajectory_started
            status = "success" if result else "valid_no_hit"
            write_status(run_dir, status, elapsed, result=result)
        except Exception as exc:
            elapsed = time.perf_counter() - trajectory_started
            write_status(run_dir, "error", elapsed, error=exc)
            logger.exception("%s failed.", run_name)
        aggregate_results(base_output)

    master = aggregate_results(base_output)
    campaign_seconds = {}
    status_counts = {}
    for status_path in sorted(base_output.glob(
        "campaign_*/run_status.json"
    )):
        payload = json.loads(status_path.read_text(encoding="utf-8"))
        name = status_path.parent.name
        campaign_seconds[name] = payload.get("elapsed_seconds")
        status = payload.get("status", "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1

    summary = {
        "model_load_seconds": round(model_load_seconds, 2),
        "campaigns_present": len(campaign_seconds),
        "campaigns_attempted_this_process": attempted,
        "unique_qualified_candidates": len(master),
        "target_candidate_count": min_candidates,
        "target_trajectory_count": min_trajectories,
        "target_reached": (
            len(master) >= min_candidates
            and len(campaign_seconds) >= min_trajectories
        ),
        "status_counts": status_counts,
        "campaign_seconds": campaign_seconds,
    }
    (base_output / "campaign_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return (
        len(master) >= min_candidates
        and len(campaign_seconds) >= min_trajectories
    )


def main():
    parser = argparse.ArgumentParser(description="Resident ESMFold2 mini-protein binder campaign")
    parser.add_argument("--config", required=True)
    parser.add_argument("--trajectories", type=int, default=20)
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--min-candidates", type=int, default=3)
    parser.add_argument("--min-trajectories", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    args, overrides = parser.parse_known_args()
    success = run_campaign(
        config_path=args.config,
        trajectories=args.trajectories,
        start=args.start,
        min_candidates=args.min_candidates,
        min_trajectories=args.min_trajectories,
        resume=args.resume,
        overrides=overrides,
    )
    raise SystemExit(0 if success else 2)


if __name__ == "__main__":
    main()
