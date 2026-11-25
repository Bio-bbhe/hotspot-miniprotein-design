"""Single-trajectory mini-protein binder optimization."""

import glob
import logging
import os
import random
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
from Bio.PDB import PDBParser, Superimposer

from src.core.candidate_pool import limit_embedding_pool
from src.data.database import BindSculptDatabase
from src.models.esm_engine import ESMEngine
from src.models.esmfold2_oracle import ESMFold2Oracle
from src.models.mpnn_mutator import MPNNMutator
from src.models.surrogate_mobo import SurrogateMOBO
from src.utils.geometry import HotspotGeometry

logger = logging.getLogger(__name__)


class SingleTrajectoryOptimizer:
    def __init__(self, config):
        self.config = config
        self.run_dir = Path(config.paths.run_output_dir)
        self.db = BindSculptDatabase(self.run_dir)
        self.esm_engine = ESMEngine(config)
        self.oracle = ESMFold2Oracle(config)
        self.final_oracle = ESMFold2Oracle(
            config,
            variant=config.models.esmfold2.get("final_variant", "full"),
        )
        self.mpnn = MPNNMutator(config)
        self.surrogate = SurrogateMOBO(config)
        self.geometry = HotspotGeometry()
        self.scaffold_dir = Path(config.paths.scaffold_dir)
        self.seed = int(config.get("seed", 42))
        random.seed(self.seed)
        np.random.seed(self.seed)

    def _meets_filters(self, binder):
        filters = self.config.filters.complex_filters
        return (
            binder.get("esmfold2_status") == "success"
            and not binder.get("filter_reason")
            and binder.get("ipsae_min", 0)
            >= filters.get("min_ipsae", 0.6)
            and binder.get("iptm", 0) >= filters.get("min_iptm", 0.8)
            and binder.get("binder_plddt", 0)
            >= filters.get("min_binder_plddt", 0.8)
        )

    def _score(self, binders, hotspot):
        for binder in binders:
            binder["composite_affinity"] = 0.0
            binder["hotspot_satisfied"] = not bool(hotspot)
            binder.pop("filter_reason", None)
            if binder.get("esmfold2_status") != "success":
                continue
            if hotspot:
                metrics = self.geometry.hotspot_contact_metrics(
                    binder["complex_pdb_path"], hotspot
                )
                binder.update(metrics)
                hotspot_cfg = self.config.get("hotspot", {})
                if (
                    metrics["hotspot_residue_contacts"]
                    < int(hotspot_cfg.get("min_hotspot_residues", 3))
                    or metrics["hotspot_contact_fraction"]
                    < float(hotspot_cfg.get("min_hotspot_fraction", 0.5))
                ):
                    binder["filter_reason"] = "failed_hotspot_contacts"
                    continue
            binder["hotspot_satisfied"] = True
            binder["composite_affinity"] = (
                0.5 * binder.get("ipsae_min", 0.0)
                + 0.3 * binder.get("iptm", 0.0)
                + 0.2 * binder.get("binder_plddt", 0.0)
            )
        return binders

    @staticmethod
    def _point_mutants(parent, limit):
        sequence = parent["binder_sequence"]
        alphabet = "ACDEFGHIKLMNPQRSTVWY"
        mutants = []
        for position, original in enumerate(sequence):
            for residue in alphabet:
                if residue == original:
                    continue
                mutants.append({
                    "id": f"single_{original}{position + 1}{residue}_{uuid.uuid4().hex[:4]}",
                    "binder_sequence": sequence[:position] + residue + sequence[position + 1:],
                    "mutation_type": "point_single",
                    "parent_id": parent["id"],
                })
        random.shuffle(mutants)
        mutants = mutants[:limit]
        seen = {item["binder_sequence"] for item in mutants}
        attempts = 0
        while len(mutants) < limit and len(sequence) >= 2:
            attempts += 1
            if attempts > limit * 20:
                break
            first, second = random.sample(range(len(sequence)), 2)
            changed = list(sequence)
            for position in (first, second):
                changed[position] = random.choice([
                    residue for residue in alphabet
                    if residue != changed[position]
                ])
            changed = "".join(changed)
            if changed in seen:
                continue
            seen.add(changed)
            mutants.append({
                "id": f"double_{uuid.uuid4().hex[:8]}",
                "binder_sequence": changed,
                "mutation_type": "point_double",
                "parent_id": parent["id"],
            })
        return mutants

    @staticmethod
    def _delta_rmsd(bound_path, monomer_path):
        try:
            parser = PDBParser(QUIET=True)
            bound = parser.get_structure("bound", bound_path)
            monomer = parser.get_structure("monomer", monomer_path)
            bound_ca = [atom for atom in bound[0]["B"].get_atoms() if atom.name == "CA"]
            monomer_ca = [atom for atom in monomer[0]["A"].get_atoms() if atom.name == "CA"]
            if not bound_ca or len(bound_ca) != len(monomer_ca):
                return float("inf")
            superimposer = Superimposer()
            superimposer.set_atoms(bound_ca, monomer_ca)
            return float(superimposer.rms)
        except Exception as exc:
            logger.warning("Delta RMSD failed: %s", exc)
            return float("inf")

    def _write_final(self, candidates):
        output = self.run_dir / "final_post_processed_candidates.csv"
        if candidates:
            frame = pd.DataFrame(candidates).sort_values(
                ["composite_affinity", "delta_rmsd"], ascending=[False, True]
            )
        else:
            frame = pd.DataFrame(columns=[
                "id", "binder_sequence", "mutation_type", "original_parent",
                "ipsae_min", "iptm", "binder_plddt", "composite_affinity", "delta_rmsd",
                "complex_pdb_path",
            ])
        frame.to_csv(output, index=False)
        return frame

    def _final_redesign(self, target_id, target_sequence, hotspot):
        filters = self.config.filters.complex_filters
        active = self.config.active_learning
        parents = [
            binder for binder in self.db.get_all_training_data()
            if binder.get("esmfold2_status") == "success"
            and binder.get("composite_affinity", 0) > 0
            and not binder.get("filter_reason")
        ]
        parents.sort(key=lambda item: item["composite_affinity"], reverse=True)
        parents = parents[: int(active.get("final_parent_count", 2))]
        if not parents:
            self._write_final([])
            return None

        old_distance = self.mpnn.config.fixed_pos_distance_threshold
        old_temperature = self.mpnn.config.sampling_temp
        polished = []
        seen = set()
        try:
            self.mpnn.config.fixed_pos_distance_threshold = float(
                filters.get("final_fixed_distance", 4.0)
            )
            self.mpnn.config.sampling_temp = float(
                filters.get("final_sampling_temp", 0.1)
            )

            for parent in parents:
                path = parent.get("complex_pdb_path")
                if not path or not os.path.exists(path):
                    continue
                mutants = self.mpnn.generate_mutants(
                    path, int(active.get("final_mpnn_mutants", 100))
                )
                mutants.sort(
                    key=lambda item: item.get("mpnn_score", float("inf"))
                )
                kept = 0
                for mutant in mutants:
                    sequence = mutant["binder_sequence"]
                    if sequence == parent["binder_sequence"] or sequence in seen:
                        continue
                    seen.add(sequence)
                    mutant.update({
                        "target_id": target_id,
                        "target_sequence": target_sequence,
                        "mutation_type": "final_mpnn_redesign",
                        "original_parent": parent["id"],
                    })
                    polished.append(mutant)
                    kept += 1
                    if kept >= int(active.get("final_keep_per_parent", 5)):
                        break
        finally:
            self.mpnn.config.fixed_pos_distance_threshold = old_distance
            self.mpnn.config.sampling_temp = old_temperature

        if not polished:
            self._write_final([])
            return None

        screened = self._score(
            self.oracle.predict_batch(polished, mode="complex"), hotspot
        )
        screened = [
            item for item in screened
            if item.get("composite_affinity", 0) > 0
        ]
        screened.sort(
            key=lambda item: item["composite_affinity"], reverse=True
        )
        shortlist = screened[
            : int(active.get("final_rescore_candidates", 10))
        ]
        if not shortlist:
            self._write_final([])
            return None

        seed_count = int(self.config.models.esmfold2.get(
            "final_num_seeds", 5
        ))
        evaluated = self._score(
            self.final_oracle.predict_batch(
                shortlist,
                mode="complex",
                seeds=range(self.seed, self.seed + seed_count),
            ),
            hotspot,
        )
        self.db.save_evaluated_batch(evaluated)
        survivors = [item for item in evaluated if self._meets_filters(item)]
        if not survivors:
            self._write_final([])
            return None

        monomers = self.final_oracle.predict_batch([
            {
                "id": f"monomer_{binder['id']}",
                "binder_sequence": binder["binder_sequence"],
            }
            for binder in survivors
        ], mode="monomer")
        monomer_by_id = {item["id"]: item for item in monomers}
        final = []
        max_rmsd = float(filters.get("max_delta_rmsd", 2.0))
        for binder in survivors:
            monomer = monomer_by_id.get(f"monomer_{binder['id']}", {})
            if monomer.get("esmfold2_status") != "success":
                continue
            binder["delta_rmsd"] = self._delta_rmsd(
                binder["complex_pdb_path"], monomer["complex_pdb_path"]
            )
            if binder["delta_rmsd"] <= max_rmsd:
                final.append(binder)

        frame = self._write_final(final)
        return None if frame.empty else frame.iloc[0].to_dict()


    def optimize_binder(
        self,
        target_id,
        target_sequence,
        target_pdb,
        hotspot_residues_str=None,
    ):
        self._write_final([])
        self.esm_engine.load_model()
        self.esm_engine.cache_target_embedding(target_sequence)

        raw_scaffolds = sorted(
            glob.glob(str(self.scaffold_dir / "*.pdb"))
            + glob.glob(str(self.scaffold_dir / "*.cif"))
        )
        scaffolds = []
        seen_stems = set()
        for path in raw_scaffolds:
            if Path(path).stem in seen_stems:
                continue
            seen_stems.add(Path(path).stem)
            scaffolds.append(path)
        random.shuffle(scaffolds)
        if not scaffolds:
            raise FileNotFoundError(f"No scaffolds found in {self.scaffold_dir}")

        active = self.config.active_learning
        pose_dir = self.run_dir / "initial_poses"
        poses = []
        accepted = 0
        for scaffold in scaffolds:
            placed = self.geometry.place_scaffolds_diverse(
                target_pdb,
                scaffold,
                pose_dir,
                Path(scaffold).stem,
                hotspot_residues_str,
                num_poses=int(active.get("poses_per_scaffold", 5)),
            )
            if placed:
                poses.extend(placed)
                accepted += 1
            if accepted >= int(active.get("scaffolds_to_accept", 3)):
                break
        if not poses:
            raise RuntimeError("No scaffold could be placed at the target surface")

        old_distance = self.mpnn.config.fixed_pos_distance_threshold
        try:
            self.mpnn.config.fixed_pos_distance_threshold = -1.0
            batches = self.mpnn.generate_mutants_batch(
                poses, int(active.get("initial_mpnn_mutants", 100))
            )
        finally:
            self.mpnn.config.fixed_pos_distance_threshold = old_distance

        initial = []
        seen = set()
        for mutants in batches:
            for mutant in sorted(
                mutants, key=lambda item: item.get("mpnn_score", float("inf"))
            ):
                if mutant["binder_sequence"] in seen:
                    continue
                seen.add(mutant["binder_sequence"])
                mutant.update({
                    "target_id": target_id,
                    "target_sequence": target_sequence,
                    "iteration": 0,
                })
                initial.append(mutant)
                break
        if not initial:
            raise RuntimeError("ProteinMPNN produced no valid initial binders")

        evaluated = self._score(
            self.oracle.predict_batch(initial, mode="complex"),
            hotspot_residues_str,
        )
        evaluated = self.esm_engine.get_concat_embeddings(evaluated)
        self.db.save_evaluated_batch(evaluated)
        successful = [
            item for item in evaluated
            if item.get("composite_affinity", 0) > 0
        ]
        if not successful:
            return None
        current_best = max(
            successful, key=lambda item: item["composite_affinity"]
        )


        stalled = False
        patience = 0
        max_patience = int(active.get("max_patience", 3))
        original_distance = self.mpnn.config.fixed_pos_distance_threshold
        try:
            for iteration in range(1, int(active.get("num_iterations", 5)) + 1):
                historical = self.db.get_all_training_data()
                self.surrogate.train(historical)
                point_threshold = float(active.get("point_mutation_iptm", 0.75))
                if current_best.get("iptm", 0) >= point_threshold:
                    mutants = self._point_mutants(
                        current_best, int(active.get("point_mutants", 200))
                    )
                else:
                    self.mpnn.config.fixed_pos_distance_threshold = (
                        -1.0 if current_best.get("iptm", 0) < 0.7 or stalled
                        else original_distance
                    )
                    mutants = self.mpnn.generate_mutants(
                        current_best["complex_pdb_path"],
                        int(active.get("mpnn_mutants", 200)),
                    )
                    for mutant in mutants:
                        mutant["parent_id"] = current_best["id"]


                seen = self.db.get_seen_sequences()
                unique = []
                for mutant in mutants:
                    if mutant["binder_sequence"] in seen:
                        continue
                    seen.add(mutant["binder_sequence"])
                    mutant.update({
                        "target_id": target_id,
                        "target_sequence": target_sequence,
                        "iteration": iteration,
                    })
                    unique.append(mutant)
                if not unique:
                    logger.warning("Mutation pool was exhausted.")
                    break


                original_count = len(unique)
                unique = limit_embedding_pool(
                    unique, active.get("embedding_pool_size", 200)
                )
                logger.info(
                    "Embedding %d/%d unique mutants.",
                    len(unique), original_count,
                )
                unique = self.esm_engine.get_concat_embeddings(unique)
                unique = self.surrogate.predict(unique)
                if self.surrogate.is_trained:
                    unique.sort(
                        key=lambda item: item.get("predicted_affinity", -1.0),
                        reverse=True,
                    )
                else:
                    random.shuffle(unique)
                selected = unique[
                    : int(active.get("candidates_per_iteration", 15))
                ]

                iteration_results = self._score(
                    self.oracle.predict_batch(selected, mode="complex"),
                    hotspot_residues_str,
                )
                self.db.save_evaluated_batch(iteration_results)
                iteration_best = max(
                    (
                        item for item in iteration_results
                        if item.get("composite_affinity", 0) > 0
                    ),
                    key=lambda item: item["composite_affinity"],
                    default=None,
                )
                if (
                    iteration_best
                    and iteration_best["composite_affinity"]
                    > current_best["composite_affinity"]
                ):
                    current_best = iteration_best
                    stalled = False
                    patience = 0
                else:
                    stalled = True
                    patience += 1
                    if patience >= max_patience:
                        break
        finally:
            self.mpnn.config.fixed_pos_distance_threshold = original_distance

        self.surrogate.save_models(prefix=f"target_{target_id}_final")
        return self._final_redesign(
            target_id, target_sequence, hotspot_residues_str
        )

