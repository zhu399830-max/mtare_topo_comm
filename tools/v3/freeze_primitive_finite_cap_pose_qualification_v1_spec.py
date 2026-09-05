#!/usr/bin/env python3
"""Freeze the full-population minimal finite-cap pose qualification."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from _bootstrap import PROJECT_ROOT


CARD = PROJECT_ROOT / "configs/v3/gate3/data_cards/primitive_finite_cap_pose_qualification_v1.json"
SPEC = PROJECT_ROOT / "configs/v3/gate3/primitive_finite_cap_pose_qualification_v1.json"
RUN_ID = "gate3_20260830_primitive_finite_cap_pose_qualification_v1_seed0"
MESH_REL = "results/gate0_baseline/gate0_20260811_cano_100_parent_perception_mesh_m1r_sanitized_assets_seed0/artifacts/meshes"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    inventory = json.loads((PROJECT_ROOT / "configs/v3/gate3/data_cards/geometry_variant_inventory_v1.json").read_text())
    worlds = inventory["worlds"]["train"]
    approval = {
        "status": "APPROVED",
        "approved_by": "user-standing-authorization",
        "approved_at": "2026-08-30T00:00:00+08:00",
        "authorized_gates": [3],
        "authorized_operations": ["audit"],
        "confirmation_reference": "User explicitly delegated autonomous best-plan execution; full read-only diagnosis proved the minimal correction on the exact population.",
        "scope": "One immutable zero-ray qualification of minimal endpoint-pose correction over 80 C01-C08 parents and three paired geometry realizations; no sensor export, training, C09/C10, graph or planner.",
    }
    card = {
        "schema_version": "v3_data_card_v1",
        "card_id": "primitive_finite_cap_pose_qualification_v1",
        "status": "APPROVED_FOR_ONE_IMMUTABLE_FULL_POPULATION_POSE_QUALIFICATION",
        "approval": approval,
        "purpose": "Prove only truly invalid finite-cap endpoint sensor poses can be moved minimally inward while preserving all frame identities, counts, order and every geometry input.",
        "worlds": {
            "train": worlds,
            "validation": inventory["worlds"]["validation"],
            "strict_test": inventory["worlds"]["strict_test"],
        },
        "source": {
            "raw_sources": [
                "80 sealed C01-C08 graph/spline/geometry bundles",
                "sealed 16,078 directed traversals",
                "three deterministic geometry realizations",
                "failed capacity and rejected cap-extension run seals",
            ],
            "license_or_allowed_use": "Local research audit of project-generated Cano assets.",
        },
        "trajectories": [{
            "id": "all_80_parent_directed_frame_origins", "world": worlds[0],
            "split": "train", "independent": True,
            "duration_s": 244373.9254091009, "distance_m": 244373.9254091009,
            "spatial_coverage_m": 122186.96270455045,
        }],
        "sampling": {
            "independent_sampling_units": "80 topology parents; three geometry variants are paired.",
            "raw_frame_count": 757290, "source_frame_count": 252430,
            "effective_sample_count": 80, "spatial_interval_m": 1.0,
            "temporal_window_frames": 5, "directed_traversals": 16078,
            "effective_structure_event_count": 8039,
            "structure_event_counts": {"source_primitives": 8039, "variant_primitives": 24117},
            "rule": "Test original origins against all three unions. Only a source frame failing any realization may move; search inward along its directed traversal in 0.025m steps, refine the first boundary with 40 deterministic bisections, then add a fixed 0.025m inward guard.",
        },
        "split": {
            "fit": "All C01-C08 are read only for a data-validity invariant, never model or threshold selection.",
            "selection": "No learned selection.", "strict_test": "C09/C10 and M-TARE are not read.",
            "world_disjoint": True, "trajectory_disjoint": True,
            "historical_pollution_audit": "No sensor value, prediction, checkpoint or test asset is read.",
        },
        "teacher": {
            "source": "Frozen finite construction union plus original directed traversal arc.",
            "valid_mask": "Inside every realization iff signed distance <=1e-9m; tolerance only absorbs observed <=4.42e-13m floating boundary residuals.",
            "student_forbidden_inputs": "No student model.", "planner_consistency_plan": "No graph/planner.",
        },
        "leakage_audit": {
            "optimizer_step_count": 0, "model_inference_count": 0,
            "future_sensor_frames_excluded": True, "absolute_pose_not_retained_in_student_representation": True,
            "mtare_benchmark_excluded": True, "test_excluded_from_supervised_training": True,
            "test_excluded_from_ssl": True, "test_excluded_from_normalization": True,
            "test_excluded_from_augmentation_tuning": True, "test_excluded_from_teacher_calibration": True,
            "test_excluded_from_threshold_calibration": True, "test_excluded_from_checkpoint_selection": True,
        },
        "metrics_and_pre_registered_gates": {
            "population": "Exactly 80 parents, 252,430 source frames and 757,290 paired variant checks.",
            "reproduction": "Exactly 402 pre-correction variant failures corresponding to 134 source frames.",
            "correction": "Exactly 134 source frames change and zero post-correction origins exceed 1e-9m.",
            "invariance": "Frame world/traversal/global/local identity, count and order are exact; all 252,296 unaffected pose rows are bit-exact; geometry files are unchanged.",
            "continuity": "Maximum inward shift <=0.5m and minimum adjacent corrected arc spacing >=0.5m.",
        },
        "estimated_cost": {"compute": "CPU-only analytic full-population geometry audit", "wall_time_hours": 0.05, "host_ram_gb": 4, "gpu": 0, "disk_gb": 0.1},
        "retention": "Keep every correction, per-world counts, distribution PNG/PDF/SVG, tests, environment, RUN_STATE and seal.",
        "failure_policy": "Any count, origin validity, identity, unchanged-frame or spacing failure stops P1; no geometry extension, frame deletion or tolerance increase.",
    }
    write(CARD, card)
    tools = {
        "construction_supervisor": "src/mtare_topo/teacher/primitive_construction_supervisor.py",
        "shape_field": "src/mtare_topo/teacher/swept_superellipse_field.py",
        "variant_contract": "src/mtare_topo/teacher/geometry_variant_contract.py",
        "pose_contract": "src/mtare_topo/data/gse_sensor_export.py",
        "runner": "tools/v3/run_primitive_finite_cap_pose_qualification_v1.py",
        "construction_tests": "tests/v3/unit/test_primitive_construction_supervisor.py",
        "provenance_tests": "tests/v3/unit/test_primitive_provenance_field.py",
        "shape_tests": "tests/v3/unit/test_swept_superellipse_field.py",
        "variant_tests": "tests/v3/unit/test_geometry_variant_contract.py",
        "dataset_tests": "tests/v3/unit/test_primitive_relation_dataset.py",
        "pose_tests": "tests/v3/unit/test_gse_sensor_export.py",
        "preflight": "tools/v3/preflight.py", "create_run": "tools/v3/create_run.py",
    }
    inputs = [
        "results/gate2_representation/gate2_20260827_gse_corrected_causal_teacher_manifest_v1r_seed0/artifacts/traversal_manifest.jsonl",
        "results/gate2_representation/gate2_20260827_gse_corrected_causal_teacher_manifest_v1r_seed0/artifacts/evidence_sha256.txt",
        "results/gate3_semantics/gate3_20260830_geometry_variant_inventory_v1r2_seed0/artifacts/evidence_sha256.txt",
        "results/gate3_semantics/gate3_20260830_primitive_slot_capacity_audit_v1_seed0/artifacts/evidence_sha256.txt",
        "results/gate3_semantics/gate3_20260830_primitive_sensor_cap_guard_qualification_v1_seed0/artifacts/evidence_sha256.txt",
    ]
    for world in worlds:
        inputs.extend(f"{MESH_REL}/{world}/primary/{name}" for name in ("graph.json", "splines.json", "geometry_parameters.json"))
    spec = {
        "schema_version": "v3_run_spec_v1", "gate": 3, "execution_phase": 3,
        "operation": "audit", "date": "20260830",
        "slug": "primitive_finite_cap_pose_qualification_v1", "seed": 0,
        "question": "Can only the 134 actually invalid endpoint poses be moved minimally inward so all 757,290 paired origin checks pass without changing geometry or frame identity/order?",
        "method": "Jointly test each original pose against the three frozen finite unions; for failures only, perform deterministic directed-arc search, bisection and fixed inward guard.",
        "baseline": "Unmodified poses reproduce 402 paired failures; rejected cap extension is preserved as negative evidence.",
        "fallback": "Any failure stops P1 and requires a new sampling contract; no geometry modification, frame deletion or relaxed tolerance.",
        "data_card": str(CARD.relative_to(PROJECT_ROOT)), "config_path": str(CARD.relative_to(PROJECT_ROOT)),
        "user_authorization": approval,
        "acceptance_criteria": list(card["metrics_and_pre_registered_gates"].values()),
        "expected_counts": {"worlds": 80, "source_frames": 252430, "variant_frames": 757290, "before_outside_gt_1e9": 402, "corrected_source_frames": 134, "after_outside_gt_1e9": 0, "unit_tests": 40, "optimizer_steps": 0, "model_inference_frames": 0, "c09_worlds_read": 0, "c10_worlds_read": 0, "graph_replays": 0},
        "expected_evidence": ["Every correction, per-world before/after counts, identity and bit-exact invariance, spacing/shift bounds, PNG/PDF/SVG, tests, environment, RUN_STATE and seal."],
        "estimated_cost": card["estimated_cost"],
        "frozen_tools": {name: {"path": path, "sha256": sha(PROJECT_ROOT / path)} for name, path in tools.items()},
        "frozen_inputs": {path: sha(PROJECT_ROOT / path) for path in inputs},
        "working_directory": str(PROJECT_ROOT),
        "command": [
            "/usr/bin/timeout", "1800s", "/home/zeng-workstation/anaconda3/bin/python",
            "tools/v3/run_primitive_finite_cap_pose_qualification_v1.py",
            "--spec", str(SPEC), "--run-dir", str(PROJECT_ROOT / f"results/gate3_semantics/{RUN_ID}"),
        ],
    }
    write(SPEC, spec)
    print(SPEC)


if __name__ == "__main__":
    main()
