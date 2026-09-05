#!/usr/bin/env python3
"""Read-only V2R recipe reclassification of the sealed Cano topology audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import PROJECT_ROOT
from mtare_topo.data.cano_topology_recipe_reclassification import (
    RECIPE_BY_STRATUM,
    recipe_batch_audit,
    recipe_candidate_audit,
    select_fixed_recipe_parents,
)
from mtare_topo.governance import load_json, write_json


SOURCE_RUN = PROJECT_ROOT / (
    "results/gate0_baseline/"
    "gate0_20260811_cano_100_topology_parent_candidate_audit_v2_bounded_resampling_seed0"
)


def execute(run_dir: Path) -> dict:
    candidate_paths = sorted((SOURCE_RUN / "metrics").glob("S*_C*.json"))
    if len(candidate_paths) != 120:
        raise RuntimeError("sealed V2 does not contain exactly 120 candidate metrics")
    candidates = {path.stem: load_json(path) for path in candidate_paths}
    if set(candidates) != {str(item["candidate_id"]) for item in candidates.values()}:
        raise RuntimeError("candidate metric filenames and candidate IDs disagree")

    audits = [recipe_candidate_audit(candidates[candidate_id]) for candidate_id in sorted(candidates)]
    selected = select_fixed_recipe_parents(audits, candidates)
    batch = recipe_batch_audit(audits, selected)
    if not batch["passed"]:
        raise RuntimeError(f"recipe batch audit failed: {batch}")

    source_relative = SOURCE_RUN.relative_to(PROJECT_ROOT)
    candidate_audit_manifest = {
        "schema_version": "cano_topology_recipe_candidate_audit_v2r",
        "source_run": str(source_relative),
        "candidate_count": len(audits),
        "candidates": audits,
    }
    parent_manifest = {
        "schema_version": "cano_topology_recipe_parent_manifest_v2r",
        "split_atom": "topology_parent",
        "selection": "fixed candidate IDs C01-C10 within each recipe stratum",
        "parents": [
            {
                **item,
                "source_graph": f"{source_relative}/artifacts/candidates/{item['candidate_id']}/graph.json",
                "source_splines": f"{source_relative}/artifacts/candidates/{item['candidate_id']}/splines.json",
            }
            for item in selected
        ],
    }
    split_manifest = {
        "schema_version": "cano_topology_recipe_split_manifest_v2r",
        "split_atom": "topology_parent",
        "selection_is_cycle_rank_independent": True,
        "splits": {
            split: [item["parent_id"] for item in selected if item["split"] == split]
            for split in ("train", "validation", "development_test")
        },
    }
    write_json(run_dir / "artifacts/candidate_recipe_audit_manifest.json", candidate_audit_manifest)
    write_json(run_dir / "artifacts/accepted_parent_manifest.json", parent_manifest)
    write_json(run_dir / "artifacts/split_manifest.json", split_manifest)
    write_json(
        run_dir / "metrics/recipe_summary.json",
        {
            "schema_version": "cano_topology_recipe_metrics_v2r",
            "recipe_mapping": RECIPE_BY_STRATUM,
            **batch,
        },
    )
    scope = {
        "sealed_candidates_read": 120,
        "selected_parent_references": 100,
        "new_topology_constructions": 0,
        "new_replays": 0,
        "new_graphs": 0,
        "new_splines": 0,
        "new_previews": 0,
        "meshes": 0,
        "lidar_observations": 0,
        "teacher_labels": 0,
        "formal_dataset_samples": 0,
        "training_samples": 0,
        "models": 0,
        "trajectories": 0,
        "mtare_changes": 0,
    }
    summary = {
        "schema_version": "cano_100_topology_parent_recipe_reclassification_summary_v2r",
        "overall_status": "PASS_CANO_100_TOPOLOGY_PARENT_RECIPE_RECLASSIFICATION_V2R",
        "source_run": str(source_relative),
        "scope": scope,
        "batch_audit": batch,
        "selected_development_test_parent_ids": [
            item["parent_id"] for item in selected if item["split"] == "development_test"
        ],
        "claim_boundary": "Topology recipe metadata and parent split only; no topology generation, mesh, LiDAR, labels, formal perception dataset, training, model, trajectory or M-TARE change.",
    }
    write_json(run_dir / "metrics/summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    execute(args.run_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
