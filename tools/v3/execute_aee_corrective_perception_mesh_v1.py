#!/usr/bin/env python3
"""Materialize the frozen 10+10 corrective perception meshes exactly once."""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

import numpy as np

import execute_cano_100_parent_perception_mesh_contract_m0 as m0
import execute_cano_100_parent_perception_mesh_m1r as m1r
from _bootstrap import PROJECT_ROOT
from generate_cano_audited_bundle import _source_precheck
from mtare_topo.governance import load_json, write_json


SOURCE_RUN = PROJECT_ROOT / "results/gate2_representation/gate2_20260821_aee_corrective_topology_candidate_audit_v1_seed20260821"
PROPOSAL = PROJECT_ROOT / "configs/v3/gate2/aee_corrective_topology_candidate_audit_v1.proposal.json"
EXPECTED_SCOPE = {
    "topology_parents_selected": 20,
    "topology_reconstructions": 20,
    "native_materializations": 20,
    "sanitized_primary_assets": 20,
    "train_complete_previews": 10,
    "validation_complete_previews": 0,
    "lidar_observations": 0,
    "teacher_labels": 0,
    "formal_dataset_samples": 0,
    "training_samples": 0,
    "models": 0,
    "c09_reads": 0,
    "c10_reads": 0,
    "mtare_changes": 0,
}


def _parents() -> list[dict]:
    records = load_json(SOURCE_RUN / "artifacts/selected_parent_manifest.json")["parents"]
    if len(records) != 20:
        raise RuntimeError("sealed selected-parent count drifted")
    adapted = []
    for item in records:
        row = dict(item)
        source = SOURCE_RUN / row["artifact_directory"]
        row.update(
            {
                "source_graph": str((source / "graph.json").relative_to(PROJECT_ROOT)),
                "source_splines": str((source / "splines.json").relative_to(PROJECT_ROOT)),
                "source_stratum_id": row["stratum_id"],
                "recipe_stratum_id": row["stratum_id"],
            }
        )
        adapted.append(row)
    return adapted


def _strata() -> dict[str, dict]:
    return {item["stratum_id"]: item for item in load_json(PROPOSAL)["frozen_candidate_scope"]["strata"]}


def execute(run_dir: Path) -> dict:
    started = time.monotonic()
    source_before = _source_precheck()
    parents = _parents()
    strata = _strata()
    m0.SOURCE_V2 = SOURCE_RUN
    meshes_root = run_dir / "artifacts/meshes"
    maps_root = run_dir / "previews/train_complete_maps"
    meshes_root.mkdir(parents=True, exist_ok=False)
    maps_root.mkdir(parents=True, exist_ok=False)
    records = []
    for parent in parents:
        root = meshes_root / parent["parent_id"] / "primary"
        primary, _, graph, splines = m0._materialize(
            parent, strata[parent["source_stratum_id"]], root, role="primary"
        )
        primary, vertices = m1r._sanitize_materialization(root, primary, graph, splines)
        record = {
            "parent_id": parent["parent_id"],
            "split": parent["split"],
            "source_parent": parent,
            "primary": primary,
            "immutable_asset": {
                "mesh_sha256": primary["mesh_sha256"],
                "remeshing_substitution_forbidden": True,
            },
        }
        write_json(run_dir / "metrics" / f"{parent['parent_id']}.json", record)
        records.append(record)
        if parent["split"] == "corrective_train":
            m0._render_complete_train_map(
                parent,
                vertices,
                graph,
                splines,
                maps_root / f"{parent['parent_id']}_complete_xy_xz.png",
                stage_label="AEE CORRECTIVE MESH V1",
            )

    ids = [item["parent_id"] for item in records]
    identities = [item["source_parent"]["canonical_parent_identity"] for item in records]
    train = [item for item in records if item["split"] == "corrective_train"]
    validation = [item for item in records if item["split"] == "corrective_validation"]
    checks = {
        "exact_parent_order": ids == [item["parent_id"] for item in parents],
        "unique_parent_ids": len(set(ids)) == 20,
        "unique_topology_identities": len(set(identities)) == 20,
        "exact_split": len(train) == len(validation) == 10,
        "all_identity_checks": all(all(item["primary"]["identity_checks"].values()) for item in records),
        "all_meshes_pass": all(item["primary"]["mesh_audit"]["passed"] is True for item in records),
        "all_sanitation_pass": all(item["primary"]["sanitation"]["passed"] is True for item in records),
        "all_mesh_hashes_unique": len({item["primary"]["mesh_sha256"] for item in records}) == 20,
    }
    source_after = _source_precheck()
    passed = all(checks.values()) and source_before == source_after
    status = "PASS_AEE_CORRECTIVE_PERCEPTION_MESH_V1" if passed else "FAIL_AEE_CORRECTIVE_PERCEPTION_MESH_V1"
    summary = {
        "schema_version": "aee_corrective_perception_mesh_v1",
        "overall_status": status,
        "checks": checks,
        "scope": EXPECTED_SCOPE,
        "parents": records,
        "source_before": source_before,
        "source_after": source_after,
        "duration_seconds": time.monotonic() - started,
        "claim_boundary": "Twenty immutable perception meshes only; zero LiDAR, teacher, formal samples, training, C09/C10 or planner change.",
    }
    write_json(run_dir / "artifacts/mesh_manifest.json", {"parents": records})
    write_json(
        run_dir / "previews/provenance.json",
        {
            "rendered_split": "corrective_train_only",
            "displayed_parent_ids": [item["parent_id"] for item in train],
            "train_rendered": 10,
            "validation_rendered": 0,
        },
    )
    write_json(run_dir / "metrics/summary.json", summary)
    if not passed:
        raise RuntimeError("corrective perception-mesh batch failed")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    try:
        result = execute(args.run_dir.resolve())
    except Exception as exc:
        write_json(
            args.run_dir.resolve() / "metrics/executor_failure.json",
            {"exception_type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()},
        )
        raise
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
