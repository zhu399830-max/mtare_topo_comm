#!/usr/bin/env python3
"""Rescore the node descriptor contract and test the frozen 16 m graph gate."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import time

import numpy as np

import execute_gse_structural_node_evidence_funnel_v1 as evidence


SPATIAL_CANDIDATE_RADIUS_M = 16.0
PER_NODE_POSITION_ERROR_BOUND_M = 1.0


def _score(tp: int, fp: int, positives: int, pairs: int) -> dict:
    accepted = tp + fp
    return {
        "pairs": int(pairs), "positive_pairs": int(positives),
        "accepted_pairs": int(accepted), "true_positive": int(tp), "false_positive": int(fp),
        "precision": tp / accepted if accepted else 0.0,
        "recall": tp / positives if positives else 0.0,
        "accepted_pair_false_fraction": fp / accepted if accepted else 1.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--construction-root", required=True, type=Path)
    parser.add_argument("--teacher-root", required=True, type=Path)
    parser.add_argument("--traversal-manifest", required=True, type=Path)
    parser.add_argument("--source-summary", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    started = time.monotonic()
    if args.output_dir.exists():
        raise RuntimeError("graph-consistency attribution output exists; overwrite is forbidden")
    args.output_dir.mkdir(parents=True)
    source = json.loads(args.source_summary.read_text(encoding="utf-8"))
    five = source["modes"]["causal_five_frame"]
    threshold = float(five["fit_selection"]["threshold"])
    source_false = float(five["c07"]["false_accept_fraction"])
    if (
        source.get("status") != "PASS_GSE_STRUCTURAL_NODE_EVIDENCE_FUNNEL_V1"
        or abs(threshold - 0.0021670341191512813) > 1e-15
        or abs(source_false - 0.017590441420511117) > 1e-15
    ):
        raise RuntimeError("sealed V1R source metric drift")
    targets = evidence._traversal_targets(args.traversal_manifest.resolve())
    aggregate = {
        partition: defaultdict(int) for partition in ("fit", "c07")
    }
    families = {family: defaultdict(int) for family in (f"S{index:02d}" for index in range(1, 11))}
    false_examples = []
    observations = defaultdict(int)
    for partition, task_count in (("fit", 180), ("c07", 30)):
        paths = sorted((args.construction_root / partition).glob("*.json"))
        if len(paths) != task_count:
            raise RuntimeError(f"{partition} task population drift")
        for construction_path in paths:
            construction = json.loads(construction_path.read_text(encoding="utf-8"))
            anchors = {
                str(row["node_id"]): np.asarray(row["anchor_xyz_m"], dtype=np.float64)
                for row in construction["base_construction"]["composition_operations"]
            }
            rows = evidence._task_observations(
                construction_path,
                args.teacher_root / partition / f"{construction_path.stem}.zarr",
                targets,
            )
            observations[partition] += len(rows)
            descriptor = np.stack([row["descriptor"]["causal_five_frame"] for row in rows])
            left, right = np.triu_indices(len(rows), 1)
            distance = evidence.pair_distance(descriptor[left], descriptor[right])
            same = np.asarray([
                rows[a]["node"] == rows[b]["node"] for a, b in zip(left, right, strict=True)
            ], dtype=np.bool_)
            spatial = np.asarray([
                np.linalg.norm(anchors[rows[a]["node"]] - anchors[rows[b]["node"]])
                for a, b in zip(left, right, strict=True)
            ], dtype=np.float64)
            proposed = distance <= threshold
            gated = proposed & (spatial <= SPATIAL_CANDIDATE_RADIUS_M)
            # Worst case: independent bounded position errors can shrink a
            # false-pair separation or expand a true-pair separation by 2e.
            robust = proposed & np.where(
                same,
                spatial + 2.0 * PER_NODE_POSITION_ERROR_BOUND_M <= SPATIAL_CANDIDATE_RADIUS_M,
                np.maximum(spatial - 2.0 * PER_NODE_POSITION_ERROR_BOUND_M, 0.0) <= SPATIAL_CANDIDATE_RADIUS_M,
            )
            bank = aggregate[partition]
            bank["pairs"] += len(same); bank["positives"] += int(np.sum(same))
            bank["descriptor_tp"] += int(np.sum(proposed & same)); bank["descriptor_fp"] += int(np.sum(proposed & ~same))
            bank["gated_tp"] += int(np.sum(gated & same)); bank["gated_fp"] += int(np.sum(gated & ~same))
            bank["robust_tp"] += int(np.sum(robust & same)); bank["robust_fp"] += int(np.sum(robust & ~same))
            if partition == "c07":
                family = construction_path.stem[:3]
                fb = families[family]
                fb["pairs"] += len(same); fb["positives"] += int(np.sum(same))
                fb["gated_tp"] += int(np.sum(gated & same)); fb["gated_fp"] += int(np.sum(gated & ~same))
                for index in np.flatnonzero(proposed & ~same).tolist():
                    a, b = int(left[index]), int(right[index])
                    false_examples.append({
                        "task": construction_path.stem, "family": family,
                        "left_node": rows[a]["node"], "right_node": rows[b]["node"],
                        "left_traversal": rows[a]["traversal"], "right_traversal": rows[b]["traversal"],
                        "left_degree": rows[a]["degree"], "right_degree": rows[b]["degree"],
                        "descriptor_distance": float(distance[index]),
                        "anchor_distance_m": float(spatial[index]),
                        "passes_16m_gate": bool(gated[index]),
                        "passes_1m_each_worst_case_gate": bool(robust[index]),
                    })
    metrics = {}
    for partition, bank in aggregate.items():
        metrics[partition] = {
            "descriptor_only": _score(bank["descriptor_tp"], bank["descriptor_fp"], bank["positives"], bank["pairs"]),
            "spatial_16m": _score(bank["gated_tp"], bank["gated_fp"], bank["positives"], bank["pairs"]),
            "spatial_16m_worst_case_1m_each": _score(bank["robust_tp"], bank["robust_fp"], bank["positives"], bank["pairs"]),
        }
    per_family = {
        family: _score(values["gated_tp"], values["gated_fp"], values["positives"], values["pairs"])
        for family, values in sorted(families.items())
    }
    c07 = metrics["c07"]["spatial_16m"]
    robust = metrics["c07"]["spatial_16m_worst_case_1m_each"]
    gates = {
        "source_machine_pass_research_contract_fail_reproduced": source_false > .01,
        "effective_fit_observations_exact_36312": observations["fit"] == 36312,
        "effective_c07_observations_exact_5874": observations["c07"] == 5874,
        "six_raw_fit_traversal_realizations_without_window_recorded": 36318 - observations["fit"] == 6,
        "c07_precision_ge_0_99": c07["precision"] >= .99,
        "c07_false_fraction_le_0_01": c07["accepted_pair_false_fraction"] <= .01,
        "c07_recall_ge_0_25": c07["recall"] >= .25,
        "all_ten_families_have_true_positive": len(per_family) == 10 and all(value["true_positive"] > 0 for value in per_family.values()),
        "one_meter_each_worst_case_has_zero_false_accept": robust["false_positive"] == 0,
        "one_meter_each_worst_case_preserves_recall": robust["recall"] == c07["recall"],
    }
    passed = all(gates.values())
    false_examples.sort(key=lambda row: (row["anchor_distance_m"], row["task"], row["left_traversal"], row["right_traversal"]))
    (args.output_dir / "c07_descriptor_false_pairs.json").write_text(json.dumps(false_examples, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = {
        "schema_version": "gse_structural_node_graph_consistency_attribution_v1",
        "status": "PASS_GSE_STRUCTURAL_NODE_GRAPH_CONSISTENCY_ATTRIBUTION_V1" if passed else "FAIL_GSE_STRUCTURAL_NODE_GRAPH_CONSISTENCY_ATTRIBUTION_V1",
        "scientific_pass": passed,
        "source_contract_rescore": {
            "reported_machine_status": source["status"],
            "correct_research_status": "FORMAL_MACHINE_PASS_RESEARCH_CONTRACT_FAIL",
            "accepted_pair_false_fraction": source_false,
            "limit": .01,
        },
        "frozen_descriptor_threshold": threshold,
        "spatial_candidate_radius_m": SPATIAL_CANDIDATE_RADIUS_M,
        "position_error_bound_per_node_m": PER_NODE_POSITION_ERROR_BOUND_M,
        "population": {"fit_tasks": 180, "c07_tasks": 30, "raw_fit_observations": 36318, "effective_fit_observations": observations["fit"], "effective_c07_observations": observations["c07"], "c07_false_examples": len(false_examples)},
        "metrics": metrics,
        "c07_per_family": per_family,
        "minimum_c07_descriptor_false_pair_anchor_distance_m": false_examples[0]["anchor_distance_m"] if false_examples else None,
        "gates": gates,
        "optimizer_steps": 0, "model_inference_frames": 0,
        "c08_c09_c10_worlds_read": 0, "graph_replays": 0, "mtare_runs": 0,
        "duration_seconds": time.monotonic() - started,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
