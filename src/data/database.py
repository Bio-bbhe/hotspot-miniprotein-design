"""Atomic CSV and NumPy persistence for evaluated binder candidates."""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class BindSculptDatabase:
    JSON_FIELDS = {"fixed_interface_positions"}
    PATH_FIELDS = {"complex_pdb_path", "esmfold2_cif_path", "pae_path"}
    NUMERIC_FIELDS = {
        "iptm",
        "ptm",
        "complex_plddt",
        "binder_plddt",
        "target_plddt",
        "ipsae_target_to_binder",
        "ipsae_binder_to_target",
        "ipsae_min",
        "composite_affinity",
        "iteration",
        "mpnn_score",
        "delta_rmsd",
        "prediction_seed",
        "prediction_seed_count",
        "hotspot_residue_contacts",
        "cdr_hotspot_contacts",
        "cdr_contact_fraction",
        "hotspot_contact_fraction",
    }

    def __init__(self, run_dir):
        self.run_dir = Path(run_dir).resolve()
        self.csv_path = self.run_dir / "binders.csv"
        self.emb_dir = self.run_dir / "embeddings"
        self.emb_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _safe_id(value):
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))

    def _embedding_path(self, binder_id):
        return self.emb_dir / f"{self._safe_id(binder_id)}.npy"

    def _portable_path(self, value):
        if not isinstance(value, str) or not value:
            return value
        path = Path(value)
        if not path.is_absolute():
            return value
        try:
            return str(path.resolve().relative_to(self.run_dir))
        except ValueError:
            return str(path)

    def save_evaluated_batch(self, binders_data):
        if not binders_data:
            return

        rows = []
        for binder in binders_data:
            row = dict(binder)
            embedding = row.pop("concat_embedding", None)
            if embedding is not None:
                output = self._embedding_path(row["id"])
                temporary = output.with_suffix(".npy.tmp")
                with temporary.open("wb") as handle:
                    np.save(handle, np.asarray(embedding))
                os.replace(temporary, output)

            for field in self.JSON_FIELDS:
                if field in row and row[field] is not None:
                    row[field] = json.dumps(row[field], separators=(",", ":"))
            for field in self.PATH_FIELDS:
                if field in row:
                    row[field] = self._portable_path(row[field])
            rows.append(row)

        new_frame = pd.DataFrame(rows)
        if self.csv_path.exists():
            current = pd.read_csv(self.csv_path)
            frame = pd.concat([current, new_frame], ignore_index=True, sort=False)
        else:
            frame = new_frame
        if "id" in frame.columns:
            frame = frame.drop_duplicates(subset="id", keep="last")
        if "binder_sequence" in frame.columns:
            frame = frame.drop_duplicates(
                subset="binder_sequence", keep="last"
            )

        temporary_csv = self.csv_path.with_suffix(".csv.tmp")
        frame.to_csv(temporary_csv, index=False)
        os.replace(temporary_csv, self.csv_path)
        logger.info("Saved %d records to %s", len(rows), self.csv_path)

    @staticmethod
    def _missing(value):
        return value is None or (
            not isinstance(value, (list, dict))
            and bool(pd.isna(value))
        )

    def get_all_training_data(self):
        if not self.csv_path.exists():
            return []

        frame = pd.read_csv(self.csv_path)
        if "esmfold2_status" in frame:
            frame = frame[frame["esmfold2_status"] == "success"]
        if "binder_sequence" in frame:
            frame = frame.drop_duplicates(
                subset="binder_sequence", keep="last"
            )

        records = []
        for raw in frame.to_dict(orient="records"):
            row = {
                key: (None if self._missing(value) else value)
                for key, value in raw.items()
            }
            for field in self.NUMERIC_FIELDS:
                if row.get(field) is not None:
                    try:
                        row[field] = float(row[field])
                    except (TypeError, ValueError):
                        row[field] = None
            for field in self.JSON_FIELDS:
                if isinstance(row.get(field), str):
                    try:
                        row[field] = json.loads(row[field])
                    except json.JSONDecodeError:
                        row[field] = None
            for field in self.PATH_FIELDS:
                value = row.get(field)
                if isinstance(value, str) and not Path(value).is_absolute():
                    row[field] = str((self.run_dir / value).resolve())

            embedding_path = self._embedding_path(row.get("id", ""))
            if not embedding_path.exists():
                continue
            embedding = np.load(embedding_path)
            if not np.all(np.isfinite(embedding)):
                continue
            row["concat_embedding"] = embedding
            records.append(row)

        logger.info(
            "Loaded %d unique training records from %s",
            len(records),
            self.csv_path,
        )
        return records

    def get_seen_sequences(self):
        if not self.csv_path.exists():
            return set()
        frame = pd.read_csv(self.csv_path, usecols=["binder_sequence"])
        return set(frame["binder_sequence"].dropna().astype(str))
