"""Small runtime helpers shared by the binder entry point and models."""

import logging
from pathlib import Path

from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STANDARD_AAS = frozenset("ACDEFGHIKLMNPQRSTVWY")


def resolve_config_paths(config):
    """Resolve project-relative paths once after OmegaConf interpolation."""
    for key in (
        "target_fasta",
        "target_cif_path",
        "base_output_dir",
        "run_output_dir",
        "scaffold_dir",
    ):
        value = config.paths.get(key)
        if value and not Path(str(value)).is_absolute():
            config.paths[key] = str((PROJECT_ROOT / str(value)).resolve())

    for key in ("executable_path", "helper_script_dir"):
        value = config.models.proteinmpnn.get(key)
        if value and not Path(str(value)).is_absolute():
            config.models.proteinmpnn[key] = str(
                (PROJECT_ROOT / str(value)).resolve()
            )

    save_dir = config.models.surrogate_model.get("save_dir")
    if save_dir and not Path(str(save_dir)).is_absolute():
        config.models.surrogate_model.save_dir = str(
            (PROJECT_ROOT / str(save_dir)).resolve()
        )
    return config


def load_config(path, overrides=()):
    config = OmegaConf.load(path)
    if overrides:
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(overrides))
    OmegaConf.resolve(config)
    config.models.esmfold2.seed = int(config.get("seed", 42))
    return resolve_config_paths(config)


def setup_logging(level="INFO", log_dir="logs"):
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, str(level).upper(), logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(Path(log_dir) / "pipeline.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )


def validate_standard_sequence(sequence, label="sequence"):
    if not isinstance(sequence, str):
        raise TypeError(f"{label} must be a string")
    sequence = sequence.strip().upper().removesuffix("*")
    invalid = sorted(set(sequence) - STANDARD_AAS)
    if not sequence or invalid:
        detail = f"invalid residues: {''.join(invalid)}" if invalid else "empty"
        raise ValueError(f"Invalid {label}: {detail}")
    return sequence
