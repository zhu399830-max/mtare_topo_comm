#!/usr/bin/env python3
"""Freeze the one-shot ERCSS Teacher/representation feasibility audit."""

from __future__ import annotations

import hashlib
from pathlib import Path

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, validate_data_card, validate_run_spec, write_json


SLUG = "gse_registered_skeleton_feasibility_v1"
RUN_ID = f"gate3_20260829_{SLUG}_seed0"
PYTHON = "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python"
DATASET = "results/gate2_representation/gate2_20260824_gse_deduplicated_dataset_export_v1_seed0"
SUPERVISION = "results/gate3_semantics/gate3_20260829_gse_composer_supervision_v1r_seed0"
MESH = "results/gate0_baseline/gate0_20260811_cano_100_parent_perception_mesh_m1r_sanitized_assets_seed0"
PROFILE_FAIL = "results/gate3_semantics/gate3_20260829_gse_route_geometry_profile_proof_v1_seed0"
SOURCE_CARD = PROJECT_ROOT / "configs/v3/gate3/data_cards/gse_route_geometry_profile_proof_v1.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    card_path = PROJECT_ROOT / f"configs/v3/gate3/data_cards/{SLUG}.json"
    spec_path = PROJECT_ROOT / f"configs/v3/gate3/{SLUG}.json"
    if card_path.exists() or spec_path.exists():
        raise RuntimeError("ERCSS feasibility freeze is immutable")
    source = load_json(SOURCE_CARD)
    card_status = "APPROVED_FOR_ONE_IMMUTABLE_GSE_REGISTERED_SKELETON_FEASIBILITY_V1"
    card = {
        "schema_version": "v3_data_card_v1",
        "card_id": SLUG,
        "status": card_status,
        "purpose": "After the formal single-scan RouteGeometryProfile failure, prove whether five ego-motion-registered causal LiDAR scans support a unique ego-connected visible physical tunnel skeleton Teacher and a fit-frozen finite representation capacity before any ERCSS model implementation or training.",
        "approval": {
            "status": "APPROVED", "approved_by": "user-standing-authorization",
            "approved_at": "2026-08-29T23:58:00+08:00", "authorized_gates": [3],
            "authorized_operations": ["audit"],
            "scope": "One immutable zero-training C01-C08 ERCSS audit over all 188126 relative-pose windows and all 37162 junction/terminal/turn/geometry-transition observations; no C09/C10/M-TARE/graph/planner.",
            "confirmation_reference": "User instructed autonomous optimal continuation without repeated approval; docs/GSE_GRAPH_METHOD_SPEC_V2 fixes this zero-training proof as the sole next step.",
        },
        "source": {
            "raw_sources": [
                "Sealed C01-C08 deduplicated 16x720 causal LiDAR shards and teacher-only poses.",
                "Sealed corrected causal event supervision with physical identity and traversal rows.",
                "Sealed C01-C08 native perception meshes, TNG graphs, splines and geometry parameters.",
                "Sealed RouteGeometryProfile scientific FAIL selecting the registered-skeleton candidate.",
            ],
            "dataset_run": DATASET, "teacher_run": SUPERVISION, "mesh_run": MESH,
            "failed_profile_run": PROFILE_FAIL,
            "license_or_allowed_use": "Local research use of sealed procedural Cano assets and project-generated objective supervision.",
        },
        "worlds": source["worlds"],
        "trajectories": source["trajectories"],
        "sampling": {
            "raw_frame_count": 252430, "effective_sample_count": 188126,
            "effective_structure_event_count": 37162,
            "declared_directed_traversals": 16078, "framed_directed_traversals": 16076,
            "event_observation_count": 37162,
            "event_observation_counts": {"junction": 26608, "terminal": 7525, "turn": 1998, "geometry_transition": 1031},
            "structure_event_counts": {"junction": 26608, "terminal": 7525, "turn": 1998, "geometry_transition": 1031},
            "fit_event_identity_counts": {"junction": 417, "terminal": 375, "turn": 297, "geometry_transition": 59},
            "selection_event_identity_counts": {"junction": 146, "terminal": 128, "turn": 95, "geometry_transition": 17},
            "independent_sampling_units": "80 world-disjoint C01-C08 topology parents and physical event identities; adjacent one-metre observations and opposite traversals are repeated evidence, not independent identities.",
            "spatial_interval_m": 1.0,
            "rule": "Validate all 188126 five-frame relative pose windows. Materialize visible skeletons only for all 37162 non-corridor event observations. Physical edges are independently sampled at 1m and never collapsed by tunnel ID.",
        },
        "teacher": {
            "source": "TNG supplies physical node/edge connectivity; spline supplies canonical edge axes and slope/curvature; sealed native-mesh per-observation geometry supplies width/height; native mesh raycast supplies past/current LOS.",
            "visibility": "A sampled axis point is eligible only within 50m and the frozen -15..15deg LiDAR vertical field from at least one of the five causal poses, and visible only if native-mesh first hit is no nearer than point distance minus the existing 0.25m LOS margin.",
            "connectivity": "Take only the visible discrete component nearest the current objective axis. Never bridge an invisible sample, join a nonincident edge, merge by tunnel ID or invent an unobserved global edge.",
            "geometry": "Map existing native-mesh width/height/slope/curvature rows to canonical physical-edge segments; d1 slope is sign-reversed into d0 canonical order. Invalid geometry remains masked.",
            "valid_mask": "Each segment dimension is valid only when the sealed native-mesh geometry_valid_mask for the nearest same-physical-edge observation is true and finite; invalid dimensions remain NaN/masked and cannot support a metric-change event.",
            "student_forbidden_inputs": "Absolute/world pose is used only to compute relative transforms and Teacher LOS. World, TNG, edge, traversal, identity, mesh, future frames and test data are never student fields.",
            "planner_consistency_plan": "No graph or planner. PASS permits only a separately governed ERCSS model readiness.",
        },
        "split": {
            "world_disjoint": True, "trajectory_disjoint": True,
            "fit": "C01-C06: 60 worlds, 142184 sequences. The smallest powers of two covering the fit maximum visible node and segment populations freeze representation capacity.",
            "selection": "C07-C08: 20 worlds, 45942 sequences. Selection may only test identity coverage and capacity overflow; it cannot resize capacity or change visibility/event rules.",
            "historical_pollution_audit": "C09/C10 and M-TARE remain unread; no checkpoint, normalization, augmentation or threshold is selected.",
        },
        "leakage_audit": {
            "test_excluded_from_supervised_training": True, "test_excluded_from_ssl": True,
            "test_excluded_from_normalization": True, "test_excluded_from_teacher_calibration": True,
            "test_excluded_from_threshold_calibration": True, "test_excluded_from_augmentation_tuning": True,
            "test_excluded_from_checkpoint_selection": True, "mtare_benchmark_excluded": True,
            "future_sensor_frames_excluded": True, "absolute_pose_not_retained_in_student_representation": True,
            "optimizer_step_count": 0, "model_inference_count": 0,
        },
        "metrics_and_pre_registered_gates": {
            "population": "Exact 80 worlds, 252430 frames, 188126 causal windows and exact 26608/7525/1998/1031 junction/terminal/turn/transition rows with the fixed partition identity counts.",
            "registration": "All five-frame pose windows are finite, current-last, and yield zero current relative translation/yaw; global SE(2) invariance and future exclusion pass unit tests.",
            "identity_support": "For each event in both fit and selection, at least 80% of physical identities have one supported observation: visible three-arm junction, visible terminal endpoint, >=15deg visible turn, or >=1m visible width/height change respectively.",
            "capacity": "Fit-derived node/segment powers-of-two are <=512/1024 and no C07-C08 observation exceeds them.",
            "geometry": "At least one physical segment receives valid sealed native-mesh geometry; invalid values stay masked and do not create metric support.",
            "system": "Zero optimizer/inference/C09/C10/M-TARE/graph/planner; source unchanged, complete per-event evidence and SHA-256 seal.",
        },
        "failure_policy": "Any source/population/identity drift, future or test read, nonunique physical edge, nonincident/tunnel-ID merge, invisible bridge, event coverage below 0.80, fit ceiling or selection overflow, resource overrun or system error seals FAIL. Do not change five-frame history, 1m sampling, 50m/-15..15deg visibility, 0.25m LOS margin, event thresholds, coverage or capacity rule after execution.",
        "estimated_cost": {"compute": "CPU-only native-mesh LOS over 37162 event observations plus relative-pose validation over 188126 windows", "wall_time_hours": 2.5, "host_ram_gb": 8, "disk_gb": 1.0, "gpu": 0},
        "retention": "Retain 37162-row observation audit, world summary, identity coverage, fit/selection capacity, PNG/PDF/SVG, logs, environment, source hashes, RUN_STATE and SHA-256 seal.",
    }
    report = validate_data_card(card)
    if not report.passed:
        raise RuntimeError("invalid ERCSS card: " + "; ".join(report.errors))
    write_json(card_path, card)
    inputs = [
        str(SOURCE_CARD.relative_to(PROJECT_ROOT)),
        f"{DATASET}/RUN_STATE.json", f"{DATASET}/metrics/summary.json", f"{DATASET}/artifacts/shard_manifest.json", f"{DATASET}/artifacts/evidence_sha256.txt",
        f"{SUPERVISION}/RUN_STATE.json", f"{SUPERVISION}/metrics/summary.json", f"{SUPERVISION}/artifacts/evidence_sha256.txt",
        f"{MESH}/RUN_STATE.json", f"{MESH}/artifacts/mesh_manifest.json", f"{MESH}/artifacts/evidence_sha256.txt",
        f"{PROFILE_FAIL}/RUN_STATE.json", f"{PROFILE_FAIL}/metrics/summary.json", f"{PROFILE_FAIL}/artifacts/evidence_sha256.txt",
        "docs/GSE_GRAPH_METHOD_SPEC_V2.md",
    ]
    tools = {
        "card": str(card_path.relative_to(PROJECT_ROOT)),
        "registration": "src/mtare_topo/representation/gse_registered_structural_skeleton.py",
        "teacher": "src/mtare_topo/teacher/gse_registered_structural_skeleton_teacher.py",
        "evaluator": "tools/v3/evaluate_gse_registered_skeleton_feasibility_v1.py",
        "runner": "tools/v3/run_gse_registered_skeleton_feasibility_v1.py",
        "freezer": "tools/v3/freeze_gse_registered_skeleton_feasibility_spec_v1.py",
        "tests_registration": "tests/v3/unit/test_gse_registered_structural_skeleton.py",
        "tests_teacher": "tests/v3/unit/test_gse_registered_structural_skeleton_teacher.py",
        "governance": "src/mtare_topo/governance.py", "preflight": "tools/v3/preflight.py", "create_run": "tools/v3/create_run.py",
    }
    spec = {
        "schema_version": "v3_run_spec_v1", "gate": 3, "execution_phase": 3,
        "date": "20260829", "slug": SLUG, "seed": 0, "operation": "audit",
        "question": "Do five ego-motion-registered causal LiDAR scans support a unique capacity-bounded visible local tunnel skeleton Teacher with sufficient junction/terminal/turn/transition identity coverage to justify ERCSS model readiness?",
        "method": "Independent 1m physical-edge sampling; five-pose relative registration; frozen LiDAR FOV plus native-mesh LOS; ego-connected visible component; sealed per-segment metric geometry; fit-frozen power-of-two capacity.",
        "baseline": "Formal single-current-scan RouteGeometryProfile proof, which had valid objective geometry but insufficient longitudinal observability.",
        "fallback": "Any scientific failure stops ERCSS and reopens method contribution analysis; no visibility/capacity/coverage adjustment and no graph/planner compensation.",
        "data_card": str(card_path.relative_to(PROJECT_ROOT)), "config_path": str(card_path.relative_to(PROJECT_ROOT)), "user_authorization": card["approval"],
        "acceptance_criteria": list(card["metrics_and_pre_registered_gates"].values()),
        "expected_counts": {"worlds": 80, "frames": 252430, "rows": 188126, "event_rows": 37162, "optimizer_steps": 0, "model_inference_frames": 0, "c09_worlds_read": 0, "c10_worlds_read": 0, "mtare_worlds_read": 0, "graph_replays": 0},
        "expected_evidence": ["37162-row event support audit, 80-world summary, eight partition/event identity coverage records, fit-frozen capacity and selection overflow proof, PNG/PDF/SVG, logs/environment/source hashes/RUN_STATE/seal."],
        "estimated_cost": card["estimated_cost"],
        "frozen_inputs": {path: sha256(PROJECT_ROOT / path) for path in inputs},
        "frozen_tools": {name: {"path": path, "sha256": sha256(PROJECT_ROOT / path)} for name, path in tools.items()},
        "working_directory": str(PROJECT_ROOT),
        "command": ["/usr/bin/timeout", "10800s", PYTHON, "tools/v3/run_gse_registered_skeleton_feasibility_v1.py", "--spec", str(spec_path), "--run-dir", str(PROJECT_ROOT / "results/gate3_semantics" / RUN_ID)],
    }
    report = validate_run_spec(spec)
    if not report.passed:
        raise RuntimeError("invalid ERCSS spec: " + "; ".join(report.errors))
    write_json(spec_path, spec)
    print(card_path.relative_to(PROJECT_ROOT)); print(spec_path.relative_to(PROJECT_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
