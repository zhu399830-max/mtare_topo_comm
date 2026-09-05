#!/usr/bin/env python3
"""Read-only audit of the sealed C09 V1R2 geometry evidence and patch semantics."""
from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, write_json


SOURCE_RUN = PROJECT_ROOT / "results/gate4_topology/gate4_20260819_cano_c09_route_conditioned_geometry_qualification_v1r2_seed0"
RESOLUTIONS = (0.1, 0.05, 0.025)
PASS_STATUS = "PASS_CANO_C09_GEOMETRY_EVIDENCE_CORRECTIVE_AUDIT_V1"


def physical_patch_contract(windows: list[dict], patches: list[dict]) -> dict:
    expected = {
        (window["world"], window["node_id"], int(tunnel_id), resolution)
        for window in windows
        for tunnel_id in window["incident_tunnel_ids"]
        for resolution in RESOLUTIONS
    }
    observed = {
        (row["world"], row["node_id"], int(row["layer_tunnel_id"]), float(row["resolution_m"]))
        for row in patches
    }
    counts = Counter(
        (row["world"], row["node_id"], int(row["layer_tunnel_id"]), float(row["resolution_m"]))
        for row in patches
    )
    duplicates = [list(key) for key, count in counts.items() if count != 1]
    integrity_failures = [
        row["artifact"] for row in patches
        if row["degenerate_triangle_count"] != 0
        or row["nonmanifold_edge_count"] != 0
        or row["seam"]["passed"] is not True
        or row["transition_semantics"]["passed"] is not True
        or not (SOURCE_RUN / row["artifact"]).is_file()
    ]
    return {
        "window_count": len(windows),
        "physical_layer_count": sum(len(row["incident_tunnel_ids"]) for row in windows),
        "physical_layers_per_window": dict(Counter(len(row["incident_tunnel_ids"]) for row in windows)),
        "resolution_count": len(RESOLUTIONS),
        "expected_patch_count": len(expected),
        "observed_patch_count": len(patches),
        "resolution_counts": dict(sorted(Counter(str(row["resolution_m"]) for row in patches).items())),
        "missing": [list(key) for key in sorted(expected - observed)],
        "unexpected": [list(key) for key in sorted(observed - expected)],
        "duplicates": duplicates,
        "integrity_failures": integrity_failures,
        "passed": expected == observed and not duplicates and not integrity_failures,
    }


def frame_contract(path: Path) -> dict:
    rows = 0
    failures = 0
    window_frames = 0
    resolution_disagreements = 0
    worlds: Counter[str] = Counter()
    with path.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            rows += 1
            worlds[row["world"]] += 1
            window_frames += int(row["inside_union_window"] == "True")
            flags = (row["passed_0.100m"], row["passed_0.050m"], row["passed_0.025m"])
            failures += int(row["passed"] != "True")
            resolution_disagreements += int(len(set(flags)) != 1)
    return {
        "rows": rows,
        "worlds": len(worlds),
        "world_frame_counts": dict(sorted(worlds.items())),
        "window_frames": window_frames,
        "outside_window_frames": rows - window_frames,
        "failed_rows": failures,
        "resolution_disagreements": resolution_disagreements,
        "passed": rows == 15833 and len(worlds) == 10 and window_frames == 1331
        and failures == 0 and resolution_disagreements == 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    started = time.monotonic()
    windows = load_json(SOURCE_RUN / "artifacts/window_manifest.json")
    patches = load_json(SOURCE_RUN / "artifacts/patch_provenance_manifest.json")
    arcs = load_json(SOURCE_RUN / "artifacts/arc_incidence_manifest.json")
    executor_summary = load_json(SOURCE_RUN / "metrics/summary.json")
    runner_summary = load_json(SOURCE_RUN / "metrics/runner_summary.json")
    source_state = load_json(SOURCE_RUN / "RUN_STATE.json")

    patch_proof = physical_patch_contract(windows, patches)
    frames = frame_contract(SOURCE_RUN / "artifacts/complete_frame_audit.csv")
    world_metrics = [load_json(path) for path in sorted((SOURCE_RUN / "metrics").glob("S*.json"))]
    metric_proof = {
        "world_metrics": len(world_metrics),
        "frames": sum(row["frames"] for row in world_metrics),
        "passed_frames": sum(row["passed_frames"] for row in world_metrics),
        "failed_frames": sum(row["failed_frames"] for row in world_metrics),
        "all_continuity_passed": all(row["continuity_passed"] for row in world_metrics),
        "all_resolution_identity_passed": all(row["resolution_pass_fail_identical"] for row in world_metrics),
        "all_exterior_poses_exact": all(row["outside_window_pose_exact"] for row in world_metrics),
    }
    metric_proof["passed"] = bool(
        metric_proof["world_metrics"] == 10 and metric_proof["frames"] == 15833
        and metric_proof["passed_frames"] == 15833 and metric_proof["failed_frames"] == 0
        and metric_proof["all_continuity_passed"] and metric_proof["all_resolution_identity_passed"]
        and metric_proof["all_exterior_poses_exact"]
    )
    provenance = {
        "source_run_state": source_state,
        "source_runner_status": runner_summary.get("overall_status"),
        "source_executor_status": executor_summary.get("overall_status"),
        "arc_endpoint_records": len(arcs),
        "incident_support_arc_count_from_readonly_contract": 213,
        "semantic_distinction": "Incident support arcs are directed graph-edge arcs; visualization patches are keyed by unique physical tunnel layer per window. They are not interchangeable counts.",
        "source_run_mutated": False,
    }
    passed = bool(
        patch_proof["passed"] and frames["passed"] and metric_proof["passed"]
        and len(arcs) == 2054 and source_state.get("state") == "FAILED"
        and runner_summary.get("overall_status") == "FAIL_CANO_C09_ROUTE_CONDITIONED_GEOMETRY_QUALIFICATION_V1R2"
        and executor_summary.get("overall_status") == "PASS_CANO_C09_ROUTE_CONDITIONED_GEOMETRY_QUALIFICATION_V1R2"
        and executor_summary.get("visual_patch_count") == 426
        and executor_summary.get("resolution_frame_audits") == 47499
        and executor_summary.get("outside_window_pose_exact") is True
        and executor_summary.get("all_frames_passed") is True
    )
    write_json(run / "artifacts/physical_patch_contract.json", patch_proof)
    write_json(run / "artifacts/frame_contract.json", frames)
    write_json(run / "artifacts/source_provenance.json", provenance)
    summary = {
        "schema_version": "cano_c09_geometry_evidence_corrective_audit_v1",
        "overall_status": PASS_STATUS if passed else "FAIL_CANO_C09_GEOMETRY_EVIDENCE_CORRECTIVE_AUDIT_V1",
        "source_run_mutated": False,
        "worlds": 10,
        "windows": len(windows),
        "physical_window_layers": patch_proof["physical_layer_count"],
        "resolutions": len(RESOLUTIONS),
        "visual_patch_count": len(patches),
        "arc_endpoint_records": len(arcs),
        "frames": frames["rows"],
        "resolution_frame_audits": executor_summary.get("resolution_frame_audits"),
        "failed_frames": frames["failed_rows"],
        "outside_window_pose_exact": metric_proof["all_exterior_poses_exact"],
        "patch_contract": patch_proof,
        "frame_contract": frames,
        "metric_contract": metric_proof,
        "inference_frames": 0,
        "graph_updates": 0,
        "training_samples_consumed": 0,
        "c10_worlds_read": 0,
        "mtare_worlds_read": 0,
        "duration_seconds": time.monotonic() - started,
        "claim_boundary": "Corrective read-only evidence audit only; V1R2 remains immutable FAILED. PASS restores C09 geometry replay eligibility but does not execute replay.",
    }
    write_json(run / "metrics/summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    if not passed:
        raise RuntimeError("C09 geometry corrective evidence audit failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
