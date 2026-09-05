#!/usr/bin/env python3
"""Freeze the one-shot RouteGeometryProfile visibility/uniqueness proof."""

from __future__ import annotations

import hashlib
from pathlib import Path

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, validate_data_card, validate_run_spec, write_json


SLUG = "gse_route_geometry_profile_proof_v1"
RUN_ID = f"gate3_20260829_{SLUG}_seed0"
PYTHON = "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python"
DATASET = "results/gate2_representation/gate2_20260824_gse_deduplicated_dataset_export_v1_seed0"
SUPERVISION = "results/gate3_semantics/gate3_20260829_gse_composer_supervision_v1r_seed0"
COMPOSER_FAIL = "results/gate3_semantics/gate3_20260829_gse_dual_composer_three_seed_training_v1_seed0"
CHANGE_PROOF = "results/gate3_semantics/gate3_20260826_gse_causal_change_point_proof_v1_seed0"
SOURCE_CARD = PROJECT_ROOT / "configs/v3/gate3/data_cards/gse_dual_composer_three_seed_training_v1.json"


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
        raise RuntimeError("RouteGeometryProfile proof freeze is immutable")
    source = load_json(SOURCE_CARD)
    card_status = "APPROVED_FOR_ONE_IMMUTABLE_GSE_ROUTE_GEOMETRY_PROFILE_PROOF_V1"
    card = {
        "schema_version": "v3_data_card_v1",
        "card_id": SLUG,
        "status": card_status,
        "purpose": "After the frozen dual-Composer scientific failure, prove whether causal 16x720 LiDAR directly supports a unique localized five-bin forward route profile of width, height, slope and curvature before implementing or training a replacement representation.",
        "approval": {
            "status": "APPROVED",
            "approved_by": "user-standing-authorization",
            "approved_at": "2026-08-29T23:40:00+08:00",
            "authorized_gates": [3],
            "authorized_operations": ["audit"],
            "scope": "One immutable zero-training C01-C08 proof over all 188126 observations and every 3029 turn/geometry-transition row; no C09/C10/M-TARE/graph/planner.",
            "confirmation_reference": "User instructed autonomous optimal continuation; docs/GSE_GRAPH_METHOD_SPEC_V1 pre-registers RouteGeometryProfile proof as the sole method-level fallback after the dual-Composer failure.",
        },
        "source": {
            "raw_sources": [
                "Sealed C01-C08 deduplicated 16x720 range-image shards.",
                "Sealed corrected causal Composer supervision with physical turn/transition identities.",
                "Sealed bidirectional causal native-mesh change-point proof.",
                "Sealed dual-Composer scientific FAIL selecting this fallback.",
            ],
            "dataset_run": DATASET,
            "teacher_run": SUPERVISION,
            "failed_model_run": COMPOSER_FAIL,
            "license_or_allowed_use": "Local research use of sealed procedural Cano assets and project-generated objective supervision.",
        },
        "worlds": source["worlds"],
        "trajectories": source["trajectories"],
        "sampling": {
            "raw_frame_count": 252430,
            "effective_sample_count": 188126,
            "effective_structure_event_count": 3029,
            "structure_event_counts": {"turn": 1998, "geometry_transition": 1031},
            "physical_identity_counts": {"turn": 392, "geometry_transition": 76},
            "independent_sampling_units": "80 world-disjoint C01-C08 topology parents and 468 typed physical event identities; adjacent observations are not independent identities.",
            "spatial_interval_m": 1.0,
            "profile_offsets_m": [0.0, 5.0, 10.0, 15.0, 20.0],
            "rule": "Audit every positive turn/transition row. Objective profile values are gathered only within the same traversal. Current LiDAR is projected into five non-overlapping 5m robot-forward slabs; a slab is supported only with first returns on both lateral and both vertical sides.",
        },
        "teacher": {
            "source": "Existing objective width/height from native-mesh cross sections and slope/curvature from the sealed traversal spline, indexed at current and +5/+10/+15/+20m within the same directed traversal.",
            "valid_mask": "A dimension is valid only when the sealed geometry target is valid at that exact traversal row; missing future rows remain masked and are never borrowed from another traversal.",
            "student_forbidden_inputs": "Future geometry, spline, mesh, TNG, world/edge/traversal/identity and test data are Teacher/audit only. Range-profile support consumes one current robot-frame range image and validity mask.",
            "planner_consistency_plan": "No graph or planner. A proof PASS permits only a separately governed typed profile representation readiness.",
        },
        "split": {
            "world_disjoint": True,
            "trajectory_disjoint": True,
            "fit": "C01-C06: 60 worlds; no fitting occurs.",
            "selection": "C07-C08: 20 worlds; the same frozen support and consistency rules are applied once without choosing thresholds.",
            "historical_pollution_audit": "C09/C10 and M-TARE worlds remain unread; the proof does not select a checkpoint or normalization statistic.",
        },
        "leakage_audit": {
            "test_excluded_from_supervised_training": True,
            "test_excluded_from_ssl": True,
            "test_excluded_from_normalization": True,
            "test_excluded_from_teacher_calibration": True,
            "test_excluded_from_threshold_calibration": True,
            "test_excluded_from_augmentation_tuning": True,
            "test_excluded_from_checkpoint_selection": True,
            "mtare_benchmark_excluded": True,
            "future_sensor_frames_excluded": True,
            "objective_future_geometry_teacher_only": True,
            "optimizer_step_count": 0,
            "model_inference_count": 0,
        },
        "metrics_and_pre_registered_gates": {
            "population": "Exact 80 worlds, 252430 frames, 188126 observations, 1998 turn rows/392 identities and 1031 transition rows/76 identities.",
            "profile_support": "For each event type separately in C01-C06 and C07-C08, at least 80% of physical identities have at least one observation with three or more jointly valid LiDAR-supported profile bins.",
            "teacher_completeness": "Every partition/event cell has nonzero rows with all five objective bins complete; every lookup remains within one traversal.",
            "reverse_uniqueness": "At least one opposite-direction canonical pair exists and p95 disagreement stays within the already frozen geometry limits: width/height 2m, slope 2deg after sign reversal, curvature 0.02/m.",
            "system": "Zero optimizer/inference/C09/C10/M-TARE/graph/planner; source unchanged and complete evidence seal.",
        },
        "failure_policy": "Any population/source drift, cross-traversal lookup, missing identity, fit or selection coverage below 0.80, reverse inconsistency, forbidden read, resource overrun or system error seals FAIL. Do not change offsets, support bins, coverage floor or geometry limits after execution.",
        "estimated_cost": {"compute": "CPU projection of 3029 current range images plus read-only Teacher/reverse audit over 80 shards", "wall_time_hours": 0.25, "host_ram_gb": 4, "disk_gb": 0.2, "gpu": 0},
        "retention": "Retain per-observation support audit, event summary, reverse metrics, PNG/PDF/SVG, logs, environment, source hashes, RUN_STATE and SHA-256 seal.",
    }
    report = validate_data_card(card)
    if not report.passed:
        raise RuntimeError("invalid RouteGeometryProfile card: " + "; ".join(report.errors))
    write_json(card_path, card)
    inputs = [
        str(SOURCE_CARD.relative_to(PROJECT_ROOT)),
        f"{DATASET}/RUN_STATE.json", f"{DATASET}/metrics/summary.json", f"{DATASET}/artifacts/shard_manifest.json", f"{DATASET}/artifacts/evidence_sha256.txt",
        f"{SUPERVISION}/RUN_STATE.json", f"{SUPERVISION}/metrics/summary.json", f"{SUPERVISION}/artifacts/evidence_sha256.txt",
        f"{COMPOSER_FAIL}/RUN_STATE.json", f"{COMPOSER_FAIL}/metrics/summary.json", f"{COMPOSER_FAIL}/artifacts/evidence_sha256.txt",
        f"{CHANGE_PROOF}/RUN_STATE.json", f"{CHANGE_PROOF}/metrics/summary.json", f"{CHANGE_PROOF}/artifacts/evidence_sha256.txt",
    ]
    tools = {
        "card": str(card_path.relative_to(PROJECT_ROOT)),
        "profile": "src/mtare_topo/representation/gse_route_geometry_profile.py",
        "evaluator": "tools/v3/evaluate_gse_route_geometry_profile_proof_v1.py",
        "runner": "tools/v3/run_gse_route_geometry_profile_proof_v1.py",
        "freezer": "tools/v3/freeze_gse_route_geometry_profile_proof_spec_v1.py",
        "tests": "tests/v3/unit/test_gse_route_geometry_profile.py",
        "governance": "src/mtare_topo/governance.py",
        "preflight": "tools/v3/preflight.py",
        "create_run": "tools/v3/create_run.py",
    }
    spec = {
        "schema_version": "v3_run_spec_v1", "gate": 3, "execution_phase": 3, "date": "20260829", "slug": SLUG, "seed": 0, "operation": "audit",
        "question": "Can current causal 16x720 LiDAR support a unique localized forward-route width/height/slope/curvature profile for every turn and geometry-transition family before profile-model training?",
        "method": "Five fixed 5m route slabs at offsets 0..20m; objective same-traversal mesh/spline profile; current-scan four-side surface support; opposite-direction canonical Teacher consistency; no fitting or inference.",
        "baseline": "Failed per-frame four-scalar GeometrySemanticState used by the sealed dual-Composer run.",
        "fallback": "Any scientific failure stops RouteGeometryProfile implementation and triggers method-contribution reassessment; no graph/planner compensation or threshold retry.",
        "data_card": str(card_path.relative_to(PROJECT_ROOT)), "config_path": str(card_path.relative_to(PROJECT_ROOT)), "user_authorization": card["approval"],
        "acceptance_criteria": list(card["metrics_and_pre_registered_gates"].values()),
        "expected_counts": {"worlds": 80, "frames": 252430, "rows": 188126, "turn_rows": 1998, "transition_rows": 1031, "turn_identities": 392, "transition_identities": 76, "optimizer_steps": 0, "model_inference_frames": 0, "c09_worlds_read": 0, "c10_worlds_read": 0, "mtare_worlds_read": 0, "graph_replays": 0},
        "expected_evidence": ["3029-row support audit, four partition/event coverage records, reverse-direction consistency, deterministic proxy diagnostic, PNG/PDF/SVG, logs/environment/source hashes/RUN_STATE/seal."],
        "estimated_cost": card["estimated_cost"],
        "frozen_inputs": {path: sha256(PROJECT_ROOT / path) for path in inputs},
        "frozen_tools": {name: {"path": path, "sha256": sha256(PROJECT_ROOT / path)} for name, path in tools.items()},
        "working_directory": str(PROJECT_ROOT),
        "command": ["/usr/bin/timeout", "1800s", PYTHON, "tools/v3/run_gse_route_geometry_profile_proof_v1.py", "--spec", str(spec_path), "--run-dir", str(PROJECT_ROOT / "results/gate3_semantics" / RUN_ID)],
    }
    report = validate_run_spec(spec)
    if not report.passed:
        raise RuntimeError("invalid RouteGeometryProfile spec: " + "; ".join(report.errors))
    write_json(spec_path, spec)
    print(card_path.relative_to(PROJECT_ROOT)); print(spec_path.relative_to(PROJECT_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
