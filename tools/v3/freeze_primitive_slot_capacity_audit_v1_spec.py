#!/usr/bin/env python3
"""Freeze the V1R P1 capacity audit with qualified finite-cap poses."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from _bootstrap import PROJECT_ROOT


CARD = PROJECT_ROOT / "configs/v3/gate3/data_cards/primitive_slot_capacity_audit_v1r.json"
SPEC = PROJECT_ROOT / "configs/v3/gate3/primitive_slot_capacity_audit_v1r.json"
RUN_ID = "gate3_20260830_primitive_slot_capacity_audit_v1r_seed0"
MESH_REL = "results/gate0_baseline/gate0_20260811_cano_100_parent_perception_mesh_m1r_sanitized_assets_seed0/artifacts/meshes"
INVENTORY_CARD = PROJECT_ROOT / "configs/v3/gate3/data_cards/geometry_variant_inventory_v1.json"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def main():
    inventory = json.loads(INVENTORY_CARD.read_text(encoding="utf-8")); worlds = inventory["worlds"]["train"]
    if len(worlds) != 80: raise RuntimeError("source world list drift")
    approval = {"status": "APPROVED", "approved_by": "user-standing-authorization", "approved_at": "2026-08-30T00:00:00+08:00", "authorized_gates": [3], "authorized_operations": ["audit"], "confirmation_reference": "User delegated continuous optimal execution; finite-cap pose qualification formally passed and V1 produced no capacity conclusion.", "scope": "One immutable V1R zero-training capacity audit over 80 C01-C08 parents, three paired geometries and three fixed role windows using only formally qualified poses; C01-C06 selects, C07/C08 only transfer-check, no C09/C10/export/model/graph/planner."}
    card = {
        "schema_version": "v3_data_card_v1", "card_id": "primitive_slot_capacity_audit_v1r", "status": "APPROVED_FOR_ONE_IMMUTABLE_QUALIFIED_POSE_SLOT_CAPACITY_AUDIT", "approval": approval,
        "purpose": "Determine whether the frozen 8-slot visible primitive Teacher is lossless and select the smallest predeclared fixed capacity with fit-only headroom before P1 export.",
        "worlds": {"train": worlds, "validation": inventory["worlds"]["validation"], "strict_test": inventory["worlds"]["strict_test"]},
        "source": {"raw_sources": ["Sealed C01-C08 graph/spline/geometry assets", "sealed 16,078-directed-traversal manifest", "three deterministic area-preserving geometry realizations", "identity-preserving CSG backend", "formally sealed 134-frame finite-cap pose qualification"], "license_or_allowed_use": "Local research audit of project-generated Cano assets.", "worlds": ["80 independent C01-C08 topology parents"]},
        "trajectories": [{"id": "aggregated_80_parent_directed_traversal_sampling_universe", "world": worlds[0], "split": "train", "independent": True, "duration_s": 244373.9254091009, "distance_m": 244373.9254091009, "spatial_coverage_m": 122186.96270455045}],
        "sampling": {"independent_sampling_units": "80 topology parents; the three geometry realizations and three role windows are paired repeated measures.", "raw_frame_count": 3600, "effective_sample_count": 80, "spatial_interval_m": 1.0, "temporal_window_frames": 5, "effective_structure_event_count": 720, "structure_event_counts": {"fit_windows": 540, "c07_windows": 90, "c08_windows": 90, "geometry_realizations": 240}, "rule": "First reproduce the sealed union-qualified pose contract over every parent; only its 134 source endpoint corrections are allowed. Then for each parent and each ellipse/rounded-rectangle/C1-mixed realization, deterministically choose the eligible five-frame suffix nearest an edge-incidence junction, nearest a terminal and farthest from either. Cast all 16x720 rays in all five frames; no weak-source filtering."},
        "split": {"fit": "C01-C06 only determine fit maximum and selected capacity.", "selection": "Choose the smallest of 8/16/32 not below ceil(1.25*fit maximum); C07 and C08 only pass/fail this unchanged capacity.", "strict_test": "C09/C10 and M-TARE are not read.", "world_disjoint": True, "trajectory_disjoint": True, "historical_pollution_audit": "C08 cannot change capacity; no predictions, checkpoint, model or test asset is read."},
        "teacher": {"source": "Every qualified CSG union-surface hit retains its exact construction primitive source set; a primitive counts as visible if any of the five causal scans contains a qualified hit from it.", "valid_mask": "No hit-count threshold or top-K truncation; multi-source hits contribute every source identity.", "student_forbidden_inputs": "No student forward is executed.", "planner_consistency_plan": "No graph or planner run."},
        "leakage_audit": {"optimizer_step_count": 0, "model_inference_count": 0, "future_sensor_frames_excluded": True, "absolute_pose_not_retained_in_student_representation": True, "mtare_benchmark_excluded": True, "test_excluded_from_supervised_training": True, "test_excluded_from_ssl": True, "test_excluded_from_normalization": True, "test_excluded_from_augmentation_tuning": True, "test_excluded_from_teacher_calibration": True, "test_excluded_from_threshold_calibration": True, "test_excluded_from_checkpoint_selection": True},
        "metrics_and_pre_registered_gates": {"population": "Exactly 80 parents, 240 paired realizations, 720 windows with 540/90/90 fit/C07/C08 and 240 per role.", "regression": "Exactly 40 unit tests and four analytic CSG cases pass.", "pose_qualification": "Exactly 402 paired correction records are reproduced across 240 tasks and every selected origin is inside its realization within 1e-9m.", "selection": "Old capacity 8 must reproduce as insufficient; selected capacity is smallest 8/16/32 >=ceil(1.25*fit maximum).", "transfer": "Every C07 and C08 window fits the unchanged selected capacity; otherwise fixed-slot decoder fails.", "provenance": "Every codebook is uint16-safe and all source memberships remain explicit."},
        "estimated_cost": {"compute": "4-process CPU Open3D full-scan audit", "wall_time_hours": 0.5, "host_ram_gb": 16, "gpu": 0, "disk_gb": 0.2},
        "retention": "Keep all 240 task/720 window records, distribution plot PNG/PDF/SVG, tests, environment, logs, RUN_STATE and seal.",
        "failure_policy": "Any hash/count/worker/pose/regression/provenance failure is FAIL. If no <=32 capacity satisfies fit headroom or C07/C08 transfer, stop fixed-slot decoder and design a variable-length set; never drop weak hits or truncate targets.",
    }
    write(CARD, card)
    tools = {
        "construction_supervisor": "src/mtare_topo/teacher/primitive_construction_supervisor.py", "shape_field": "src/mtare_topo/teacher/swept_superellipse_field.py", "variant_contract": "src/mtare_topo/teacher/geometry_variant_contract.py", "csg_backend": "src/mtare_topo/teacher/csg_mesh_provenance.py", "dataset_contract": "src/mtare_topo/data/primitive_relation_dataset.py", "pose_contract": "src/mtare_topo/data/gse_sensor_export.py", "runner": "tools/v3/run_primitive_slot_capacity_audit_v1.py", "analytic_checker": "tools/v3/check_csg_mesh_provenance_contract.py", "construction_tests": "tests/v3/unit/test_primitive_construction_supervisor.py", "provenance_tests": "tests/v3/unit/test_primitive_provenance_field.py", "shape_tests": "tests/v3/unit/test_swept_superellipse_field.py", "variant_tests": "tests/v3/unit/test_geometry_variant_contract.py", "dataset_tests": "tests/v3/unit/test_primitive_relation_dataset.py", "pose_tests": "tests/v3/unit/test_gse_sensor_export.py", "preflight": "tools/v3/preflight.py", "create_run": "tools/v3/create_run.py",
    }
    inputs = ["results/gate2_representation/gate2_20260827_gse_corrected_causal_teacher_manifest_v1r_seed0/artifacts/traversal_manifest.jsonl", "results/gate2_representation/gate2_20260827_gse_corrected_causal_teacher_manifest_v1r_seed0/artifacts/evidence_sha256.txt", "results/gate3_semantics/gate3_20260830_geometry_variant_inventory_v1r2_seed0/artifacts/evidence_sha256.txt", "results/gate3_semantics/gate3_20260830_csg_mesh_provenance_acceleration_v1_seed0/artifacts/evidence_sha256.txt", "results/gate3_semantics/gate3_20260830_primitive_slot_capacity_audit_v1_seed0/artifacts/evidence_sha256.txt", "results/gate3_semantics/gate3_20260830_primitive_finite_cap_pose_qualification_v1_seed0/artifacts/evidence_sha256.txt"]
    for world in worlds:
        inputs.extend(f"{MESH_REL}/{world}/primary/{name}" for name in ("graph.json", "splines.json", "geometry_parameters.json"))
    spec = {
        "schema_version": "v3_run_spec_v1", "gate": 3, "execution_phase": 3, "operation": "audit", "date": "20260830", "slug": "primitive_slot_capacity_audit_v1r", "seed": 0,
        "question": "What fit-only fixed primitive-set capacity preserves all actually visible five-frame construction targets with 25 percent headroom and transfers unchanged to C07/C08?",
        "method": "Reproduce the formally qualified minimal endpoint poses, then render three fixed role windows for every C01-C08 parent and every paired geometry with the qualified CSG backend; count all source primitives with at least one hit; select only from C01-C06 using smallest power-of-two 8/16/32 above 1.25x maximum, then audit C07/C08 unchanged.",
        "baseline": "Frozen maximum 8 slots, already contradicted by a C01 junction with 12 supported visible primitives.", "fallback": "If 32 is insufficient or transfer overflows, stop fixed slots and design a variable-length set; no source filtering or target truncation.",
        "data_card": str(CARD.relative_to(PROJECT_ROOT)), "config_path": str(CARD.relative_to(PROJECT_ROOT)), "user_authorization": approval,
        "acceptance_criteria": list(card["metrics_and_pre_registered_gates"].values()), "expected_counts": {"worlds": 80, "realizations": 240, "windows": 720, "fit_windows": 540, "c07_windows": 90, "c08_windows": 90, "window_frame_references": 3600, "maximum_rays": 41472000, "paired_pose_corrections": 402, "unit_tests": 40, "analytic_cases": 4, "optimizer_steps": 0, "model_inference_frames": 0, "c09_worlds_read": 0, "c10_worlds_read": 0, "graph_replays": 0},
        "expected_evidence": ["All task/window cardinalities, fit-only selection, unchanged C07/C08 transfer, provenance maxima, PNG/PDF/SVG, environment, raw logs, RUN_STATE and SHA-256 seal."], "estimated_cost": card["estimated_cost"],
        "frozen_tools": {name: {"path": path, "sha256": sha(PROJECT_ROOT / path)} for name, path in tools.items()}, "frozen_inputs": {path: sha(PROJECT_ROOT / path) for path in inputs}, "working_directory": str(PROJECT_ROOT),
        "command": ["/usr/bin/timeout", "7200s", "/tmp/mtare_gate4_meshing_sidecar_v1/bin/python", "tools/v3/run_primitive_slot_capacity_audit_v1.py", "--spec", str(SPEC), "--run-dir", str(PROJECT_ROOT / f"results/gate3_semantics/{RUN_ID}"), "--workers", "4"],
    }
    write(SPEC, spec); print(SPEC)


if __name__ == "__main__": main()
