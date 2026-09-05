#!/usr/bin/env python3
"""Materialize all 100 frozen Cano parents as immutable perception assets."""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

import execute_cano_100_parent_perception_mesh_contract_m0 as m0
from mtare_topo.data.cano_perception_mesh_contract import immutable_full_parent_asset_batch_audit
from mtare_topo.governance import load_json, write_json


EXPECTED_SCOPE = {
    "topology_parents_selected": 100,
    "topology_reconstructions": 100,
    "native_perception_meshes": 100,
    "primary_meshes": 100,
    "replay_meshes": 0,
    "train_complete_previews": 80,
    "validation_complete_previews": 0,
    "development_test_complete_previews": 0,
    "validation_or_development_test_parents_read": 20,
    "anchors": 0,
    "lidar_observations": 0,
    "teacher_labels": 0,
    "formal_dataset_samples": 0,
    "training_samples": 0,
    "models": 0,
    "trajectories": 0,
    "gazebo_runs": 0,
    "isaac_runs": 0,
    "mtare_changes": 0,
}


def execute(run_dir: Path) -> dict:
    started = time.monotonic()
    source_before = m0._source_precheck()
    parents = load_json(m0.SOURCE_V2R / "artifacts/accepted_parent_manifest.json")["parents"]
    strata = m0._stratum_registry()
    meshes_root = run_dir / "artifacts/meshes"
    maps_root = run_dir / "previews/train_complete_maps"
    meshes_root.mkdir(parents=True, exist_ok=False)
    maps_root.mkdir(parents=True, exist_ok=False)
    records = []

    for parent in parents:
        parent_id = parent["parent_id"]
        root = meshes_root / parent_id
        root.mkdir()
        primary, vertices, graph, splines = m0._materialize(
            parent,
            strata[parent["source_stratum_id"]],
            root / "primary",
            role="primary",
        )
        record = {
            "parent_id": parent_id,
            "recipe_stratum_id": parent["recipe_stratum_id"],
            "split": parent["split"],
            "source_parent": parent,
            "primary": primary,
            "immutable_asset": {
                "mesh_sha256": primary["mesh_sha256"],
                "remeshing_substitution_forbidden": True,
            },
        }
        write_json(run_dir / "metrics" / f"{parent_id}.json", record)
        records.append(record)
        if parent["split"] == "train":
            m0._render_complete_train_map(
                parent,
                vertices,
                graph,
                splines,
                maps_root / f"{parent_id}_complete_xy_xz.png",
            )
        if primary.get("mesh_audit", {}).get("passed") is not True:
            raise RuntimeError(f"{parent_id}: primary immutable asset quality failed")

    batch = immutable_full_parent_asset_batch_audit(records, parents)
    source_after = m0._source_precheck()
    status = (
        "PASS_CANO_100_PARENT_PERCEPTION_MESH_M1"
        if batch["passed"]
        else "FAIL_CANO_100_PARENT_PERCEPTION_MESH_M1"
    )
    summary = {
        "schema_version": "cano_100_parent_perception_mesh_summary_m1",
        "overall_status": status,
        "batch_audit": batch,
        "scope": EXPECTED_SCOPE,
        "parents": records,
        "source_before": source_before,
        "source_after": source_after,
        "duration_seconds": time.monotonic() - started,
        "asset_policy": "Downstream must read sealed M1 OBJ; remeshing substitution forbidden.",
        "split_policy": "Only 80 train parents are rendered for manual review; validation and development-test receive frozen automatic checks only.",
        "claim_boundary": "Immutable perception assets only; zero LiDAR, labels, formal data, training, simulator or M-TARE change.",
    }
    write_json(
        run_dir / "artifacts/mesh_manifest.json",
        {"asset_policy": summary["asset_policy"], "split_policy": summary["split_policy"], "parents": records},
    )
    write_json(
        run_dir / "previews/provenance.json",
        {
            "rendered_split": "train_only",
            "displayed_parent_ids": [item["parent_id"] for item in records if item["split"] == "train"],
            "train_rendered": 80,
            "validation_rendered": 0,
            "development_test_rendered": 0,
            "hypothesis": "Each sealed train perception asset is complete and aligned with its exact source graph/splines.",
            "units": "meters",
        },
    )
    write_json(run_dir / "metrics/summary.json", summary)
    if not batch["passed"]:
        raise RuntimeError(f"M1 batch audit failed: {batch}")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    try:
        summary = execute(args.run_dir.resolve())
    except Exception as exc:
        write_json(
            args.run_dir.resolve() / "metrics/executor_failure.json",
            {"exception_type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()},
        )
        raise
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
