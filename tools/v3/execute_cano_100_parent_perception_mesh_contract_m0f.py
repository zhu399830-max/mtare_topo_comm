#!/usr/bin/env python3
"""Generate and qualify ten single-pass immutable Cano perception-mesh assets."""

from __future__ import annotations

import argparse, json, time, traceback
from pathlib import Path

import execute_cano_100_parent_perception_mesh_contract_m0 as m0
from mtare_topo.data.cano_perception_mesh_contract import immutable_asset_batch_audit, select_train_sentinels
from mtare_topo.governance import load_json, write_json

EXPECTED_SCOPE = {
    "topology_parents_selected": 10, "topology_reconstructions": 10,
    "native_perception_meshes": 10, "primary_meshes": 10, "replay_meshes": 0,
    "train_complete_previews": 10, "validation_or_development_test_parents_read": 0,
    "anchors": 0, "lidar_observations": 0, "teacher_labels": 0,
    "formal_dataset_samples": 0, "training_samples": 0, "models": 0,
    "trajectories": 0, "gazebo_runs": 0, "isaac_runs": 0, "mtare_changes": 0,
}


def execute(run_dir: Path) -> dict:
    started = time.monotonic(); source_before = m0._source_precheck()
    parents = load_json(m0.SOURCE_V2R / "artifacts/accepted_parent_manifest.json")["parents"]
    sentinels = select_train_sentinels(parents); strata = m0._stratum_registry()
    meshes_root = run_dir / "artifacts/meshes"; maps_root = run_dir / "previews/train_complete_maps"
    meshes_root.mkdir(parents=True, exist_ok=False); maps_root.mkdir(parents=True, exist_ok=False)
    records = []
    for parent in sentinels:
        root = meshes_root / parent["parent_id"]; root.mkdir()
        primary, vertices, graph, splines = m0._materialize(parent, strata[parent["source_stratum_id"]], root / "primary", role="primary")
        record = {"parent_id": parent["parent_id"], "recipe_stratum_id": parent["recipe_stratum_id"], "split": parent["split"], "source_parent": parent, "primary": primary, "immutable_asset": {"mesh_sha256": primary["mesh_sha256"], "remeshing_substitution_forbidden": True}}
        write_json(run_dir / "metrics" / f"{parent['parent_id']}.json", record); records.append(record)
        m0._render_complete_train_map(parent, vertices, graph, splines, maps_root / f"{parent['parent_id']}_complete_xy_xz.png")
        if primary.get("mesh_audit", {}).get("passed") is not True:
            raise RuntimeError(f"{parent['parent_id']}: primary immutable asset quality failed")
    ids = tuple(item["parent_id"] for item in records)
    batch = immutable_asset_batch_audit(records)
    source_after = m0._source_precheck()
    summary = {"schema_version":"cano_100_parent_perception_mesh_contract_summary_m0f","overall_status":"PASS_CANO_100_PARENT_PERCEPTION_MESH_CONTRACT_M0F" if batch["passed"] else "FAIL_CANO_100_PARENT_PERCEPTION_MESH_CONTRACT_M0F","batch_audit":batch,"scope":EXPECTED_SCOPE,"parents":records,"source_before":source_before,"source_after":source_after,"duration_seconds":time.monotonic()-started,"asset_policy":"Downstream must read sealed OBJ; remeshing substitution forbidden.","claim_boundary":"Train-only immutable native perception assets; zero replay, LiDAR, data, training, simulator or M-TARE change."}
    write_json(run_dir / "artifacts/mesh_manifest.json", {"asset_policy":summary["asset_policy"],"parents":records})
    write_json(run_dir / "previews/provenance.json", {"split":"train_only","selection":"accepted_rank_in_recipe=1 fixed before meshing","displayed_parent_ids":list(ids),"validation_or_development_test_rendered":0,"hypothesis":"Each sealed primary perception asset is complete and aligned with exact source graph/splines.","units":"meters"})
    write_json(run_dir / "metrics/summary.json", summary)
    if not batch["passed"]: raise RuntimeError(f"M0F batch audit failed: {batch}")
    return summary


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--run-dir",required=True,type=Path); args=parser.parse_args()
    try: summary=execute(args.run_dir.resolve())
    except Exception as exc:
        write_json(args.run_dir.resolve()/"metrics/executor_failure.json",{"exception_type":type(exc).__name__,"message":str(exc),"traceback":traceback.format_exc()}); raise
    print(json.dumps(summary,ensure_ascii=False,indent=2)); return 0


if __name__=="__main__": raise SystemExit(main())
