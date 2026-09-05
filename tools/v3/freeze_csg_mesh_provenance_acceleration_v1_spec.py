#!/usr/bin/env python3
"""Freeze the one-run C01 accelerated provenance proof."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from _bootstrap import PROJECT_ROOT


CARD = PROJECT_ROOT / "configs/v3/gate3/data_cards/csg_mesh_provenance_acceleration_v1.json"
SPEC = PROJECT_ROOT / "configs/v3/gate3/csg_mesh_provenance_acceleration_v1.json"
RUN_ID = "gate3_20260830_csg_mesh_provenance_acceleration_v1_seed0"
WORLD = "S01_flat_tree_small_C01"
PRIMARY = "results/gate0_baseline/gate0_20260811_cano_100_parent_perception_mesh_m1r_sanitized_assets_seed0/artifacts/meshes/S01_flat_tree_small_C01/primary"


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
    approval = {
        "status": "APPROVED", "approved_by": "user-standing-authorization",
        "approved_at": "2026-08-30T00:00:00+08:00", "authorized_gates": [3],
        "authorized_operations": ["audit"],
        "confirmation_reference": "User requested continuous optimal execution without repeated approvals.",
        "scope": "One immutable C01-only 6-pose/6,144-ray backend proof; no export, training, C09/C10, graph or planner.",
    }
    card = {
        "schema_version": "v3_data_card_v1", "card_id": "csg_mesh_provenance_acceleration_v1",
        "status": "APPROVED_FOR_ONE_IMMUTABLE_CSG_PROVENANCE_ACCELERATION_PROOF", "approval": approval,
        "purpose": "Prove that closed per-primitive meshes plus analytic CSG qualification preserve range/provenance while making the frozen P1 teacher computationally feasible.",
        "worlds": {"train": [WORLD], "validation": ["S01_flat_tree_small_C09"], "strict_test": ["S01_flat_tree_small_C10"]},
        "source": {"raw_sources": [f"{PRIMARY}/graph.json", f"{PRIMARY}/splines.json", f"{PRIMARY}/geometry_parameters.json", "sealed C01 training manifest poses"], "license_or_allowed_use": "Local research audit of project-generated Cano assets."},
        "trajectories": [{"id": "C01_role_stratified_static_pose_proof", "world": WORLD, "split": "train", "independent": True, "duration_s": 6.0, "distance_m": 3.0, "spatial_coverage_m": 3.0, "pose_count": 6, "roles": {"interior": 2, "junction": 2, "terminal": 2}}],
        "sampling": {"independent_sampling_units": "The C01 topology is the one independent development unit; its six poses and rays are nested diagnostics, never counted as independent worlds.", "raw_frame_count": 6, "raw_ray_count": 6144, "effective_sample_count": 1, "rays_per_pose": 1024, "ray_selection": "Uniform deterministic indices over the frozen 16x720 scan", "spatial_interval_m": 1.0, "temporal_window_frames": 1, "effective_structure_event_count": 55, "structure_event_counts": {"swept_primitive_edges": 55, "role_stratified_poses": 6}, "rule": "Choose the first two sealed eligible C01 frames in each of interior, junction and terminal roles, then select 1,024 uniformly spaced ray indices from each frozen 16x720 scan."},
        "split": {"fit": "No fitting", "selection": "No threshold or model selection; fixed engineering contracts", "strict_test": "C09/C10 and M-TARE are not read", "world_disjoint": True, "trajectory_disjoint": True, "historical_pollution_audit": "Only C01 development geometry and sealed pose metadata are read."},
        "teacher": {"source": "Deterministic C1-mixed area-preserving swept-superellipse construction; dense analytic ray-exit is the range/provenance reference.", "valid_mask": "A CSG return is qualified only when the candidate lies on and exits the analytic union; ambiguous co-active primitive identities remain multi-source.", "student_forbidden_inputs": "No student model or checkpoint is invoked.", "planner_consistency_plan": "No graph or planner run."},
        "leakage_audit": {"optimizer_step_count": 0, "model_inference_count": 0, "future_sensor_frames_excluded": True, "absolute_pose_not_retained_in_student_representation": True, "mtare_benchmark_excluded": True, "test_excluded_from_supervised_training": True, "test_excluded_from_ssl": True, "test_excluded_from_normalization": True, "test_excluded_from_augmentation_tuning": True, "test_excluded_from_teacher_calibration": True, "test_excluded_from_threshold_calibration": True, "test_excluded_from_checkpoint_selection": True},
        "metrics_and_pre_registered_gates": {"contract": "30 unit tests and four analytic union/provenance cases pass.", "geometry": "valid agreement and qualified coverage >=0.98; p95/p99 range difference <=0.05 m; >0.05 m difference fraction <=0.005.", "identity": "Primitive identity agreement >=0.995 and ambiguity retained.", "determinism": "Full and sparse operand queries produce identical CSG hit digests; two analytic references produce identical digests.", "throughput": "Sparse CSG is >=3x analytic reference; conservative 2x-complexity 32-worker P1 projection <=72 h."},
        "estimated_cost": {"compute": "CPU-only Open3D CSG and analytic comparison", "wall_time_hours": 0.03, "host_ram_gb": 4, "gpu": 0, "disk_gb": 0.1},
        "retention": "Keep exact full/sparse metrics and digests, analytic cases, logs, environment, backend schema, PNG/PDF/SVG, RUN_STATE and seal.",
        "failure_policy": "Any hash/count/accuracy/identity/determinism/throughput failure stops P1 export; no ray, pose, resolution, tolerance or acceptance threshold may be changed after execution.",
    }
    write(CARD, card)
    tools = {
        "construction_supervisor": "src/mtare_topo/teacher/primitive_construction_supervisor.py",
        "shape_field": "src/mtare_topo/teacher/swept_superellipse_field.py",
        "variant_contract": "src/mtare_topo/teacher/geometry_variant_contract.py",
        "csg_backend": "src/mtare_topo/teacher/csg_mesh_provenance.py",
        "benchmark": "tools/v3/benchmark_csg_mesh_provenance_backend.py",
        "analytic_checker": "tools/v3/check_csg_mesh_provenance_contract.py",
        "runner": "tools/v3/run_csg_mesh_provenance_acceleration_v1.py",
        "construction_tests": "tests/v3/unit/test_primitive_construction_supervisor.py",
        "provenance_tests": "tests/v3/unit/test_primitive_provenance_field.py",
        "shape_tests": "tests/v3/unit/test_swept_superellipse_field.py",
        "variant_tests": "tests/v3/unit/test_geometry_variant_contract.py",
        "preflight": "tools/v3/preflight.py", "create_run": "tools/v3/create_run.py",
    }
    inputs = [
        f"{PRIMARY}/graph.json", f"{PRIMARY}/splines.json", f"{PRIMARY}/geometry_parameters.json",
        "results/gate1_data/gate1_20260812_cano_phase2_supervised_range_dataset_v2r_seed0/artifacts/manifest.jsonl",
        "results/gate3_semantics/gate3_20260830_geometry_variant_inventory_v1r2_seed0/artifacts/evidence_sha256.txt",
    ]
    spec = {
        "schema_version": "v3_run_spec_v1", "gate": 3, "execution_phase": 3, "operation": "audit",
        "date": "20260830", "slug": "csg_mesh_provenance_acceleration_v1", "seed": 0,
        "question": "Can an identity-preserving closed-mesh CSG backend reproduce the C1-mixed analytic teacher accurately and reduce the frozen 757,290-frame P1 provenance cost to a feasible parallel run?",
        "method": "Build every C01 primitive as a separate 5 cm axial/64-segment closed mesh, list ordered Open3D intersections, qualify union exits with the frozen 2.5 cm analytic field and three interior path probes, retain all co-active identities, and accelerate analytic queries with spacing-padded operand AABBs.",
        "baseline": "The exact same candidates using all-55-operand analytic queries, plus the original dense analytic ray-exit reference.",
        "fallback": "Any contract failure stops P1; there is no alternate resolution, reduced dataset, identity collapse or acceptance change.",
        "data_card": str(CARD.relative_to(PROJECT_ROOT)), "config_path": str(CARD.relative_to(PROJECT_ROOT)),
        "command": ["/usr/bin/timeout", "1800s", "/tmp/mtare_gate4_meshing_sidecar_v1/bin/python", "tools/v3/run_csg_mesh_provenance_acceleration_v1.py", "--spec", str(SPEC), "--run-dir", str(PROJECT_ROOT / f"results/gate3_semantics/{RUN_ID}")],
        "acceptance_criteria": list(card["metrics_and_pre_registered_gates"].values()),
        "expected_counts": {"worlds": 1, "poses": 6, "rays": 6144, "primitives": 55, "analytic_cases": 4, "unit_tests": 30, "optimizer_steps": 0, "model_inference_frames": 0, "c09_worlds_read": 0, "c10_worlds_read": 0, "graph_replays": 0},
        "expected_evidence": ["Full/sparse hit digests and metrics, analytic cases, unit log, resource projection, schema, environment, PNG/PDF/SVG, RUN_STATE and SHA-256 seal."],
        "estimated_cost": card["estimated_cost"], "user_authorization": approval,
        "frozen_tools": {name: {"path": path, "sha256": sha(PROJECT_ROOT / path)} for name, path in tools.items()},
        "frozen_inputs": {path: sha(PROJECT_ROOT / path) for path in inputs},
        "working_directory": str(PROJECT_ROOT),
    }
    write(SPEC, spec)
    print(SPEC)


if __name__ == "__main__":
    main()
