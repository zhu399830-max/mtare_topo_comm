#!/usr/bin/env python3
"""Freeze the complete P1a LiDAR and primitive-provenance export."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from _bootstrap import PROJECT_ROOT


CARD = PROJECT_ROOT / "configs/v3/gate3/data_cards/primitive_relation_p1a_sensor_provenance_export_v1.json"
SPEC = PROJECT_ROOT / "configs/v3/gate3/primitive_relation_p1a_sensor_provenance_export_v1.json"
RUN_ID = "gate3_20260830_primitive_relation_p1a_sensor_provenance_export_v1_seed0"
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
        "status": "APPROVED", "approved_by": "user-standing-authorization",
        "approved_at": "2026-08-30T00:00:00+08:00", "authorized_gates": [3],
        "authorized_operations": ["data_export"],
        "confirmation_reference": "User instructed continuous autonomous execution; all P0 construction, shape, inventory, CSG, pose and 32-slot capacity prerequisites formally passed.",
        "scope": "One immutable P1a export over exact C01-C08 80 parents and three paired geometry realizations. Produce sensor/provenance/construction shards only; no C09/C10, model, optimizer, graph, planner or M-TARE.",
    }
    card = {
        "schema_version": "v3_data_card_v1", "card_id": "primitive_relation_p1a_sensor_provenance_export_v1",
        "status": "APPROVED_FOR_ONE_IMMUTABLE_P1A_SENSOR_PROVENANCE_EXPORT", "approval": approval,
        "purpose": "Materialize the expensive reusable bottom layer for geometry-primitive learning: paired LiDAR, exact per-ray construction source sets, qualified poses and realized construction programs. P1b derives 32-slot relation supervision only after this export passes.",
        "worlds": {"train": worlds, "validation": inventory["worlds"]["validation"], "strict_test": inventory["worlds"]["strict_test"]},
        "source": {
            "raw_sources": ["80 sealed C01-C08 graph/spline/geometry bundles", "sealed 16,078 directed traversals", "three deterministic area-preserving geometry realizations", "formal fast CSG provenance backend", "formal 134-frame finite-cap correction", "formal 32-slot capacity selection"],
            "license_or_allowed_use": "Local research export of project-generated Cano assets; redistribution remains subject to upstream licensing.",
            "worlds": ["80 independent topology parents, C01-C08 only"],
        },
        "trajectories": [{
            "id": "all_80_parent_all_directed_traversals_three_paired_geometries", "world": worlds[0],
            "split": "train", "independent": True, "duration_s": 733121.7762273027,
            "distance_m": 733121.7762273027, "spatial_coverage_m": 122186.96270455045,
        }],
        "sampling": {
            "independent_sampling_units": "80 topology parents; three geometry realizations are paired repeated measures and never cross parent split.",
            "raw_frame_count": 757290, "effective_sample_count": 80,
            "spatial_interval_m": 1.0, "temporal_window_frames": 5,
            "effective_structure_event_count": 24117,
            "structure_event_counts": {"source_primitives": 8039, "variant_primitives": 24117, "directed_traversals": 16078, "planned_five_frame_sequences": 564378},
            "rule": "For every frozen frame and paired realization, render all 16x720 rays to 50m from the formally corrected pose. Encode no-hit as code0 and every qualified union-exit source set as a world-realization-local uint16 code; never select, truncate or collapse ambiguous sources.",
        },
        "split": {
            "fit": "C01-C06: 60 parents, 180 geometry shards.",
            "selection": "C07: 10 parents, 30 geometry shards; generated but not used to alter P1a schema.",
            "development_transfer": "C08: 10 parents, 30 geometry shards; generated once under the already frozen schema.",
            "strict_test": "C09/C10 and all M-TARE benchmark worlds are forbidden.",
            "world_disjoint": True, "trajectory_disjoint": True,
            "historical_pollution_audit": "No historical sensor prediction, model error, checkpoint, graph result or planner outcome affects geometry, pose, rendering or code assignment.",
        },
        "teacher": {
            "source": "Procedural edge-level construction identity retained through independent closed swept-superellipse operands and analytically qualified CSG union exits.",
            "labels": "range_m, valid_mask, lossless primitive_membership_code plus codebook, corrected route pose and realized construction program.",
            "valid_mask": "A return is valid only within frozen near/50m range and after CSG path/surface qualification; code0 iff invalid.",
            "student_forbidden_inputs": "Pose, world, traversal, edge, primitive identity, TNG and future frames are Teacher/evaluation metadata only.",
            "planner_consistency_plan": "No graph or planner in P1a.",
        },
        "leakage_audit": {
            "optimizer_step_count": 0, "model_inference_count": 0, "future_sensor_frames_excluded": True,
            "absolute_pose_not_retained_in_student_representation": True, "mtare_benchmark_excluded": True,
            "test_excluded_from_supervised_training": True, "test_excluded_from_ssl": True,
            "test_excluded_from_normalization": True, "test_excluded_from_augmentation_tuning": True,
            "test_excluded_from_teacher_calibration": True, "test_excluded_from_threshold_calibration": True,
            "test_excluded_from_checkpoint_selection": True,
        },
        "metrics_and_pre_registered_gates": {
            "population": "Exactly 80 parents, 240 shards, 757,290 frames, 8,723,980,800 rays, 24,117 realized primitives and 402 paired correction applications.",
            "regression": "Exactly 67 unit tests, including the Gate-3 data-export authorization contract, and four independent analytic CSG cases pass before export.",
            "origin_validity": "Every corrected sensor origin has finite-union distance <=1e-9m.",
            "provenance": "Every valid ray has nonzero uint16 code, every invalid ray code0, complete multi-source membership is retained and at least one ambiguous ray exists globally.",
            "determinism": "The first frame of every shard re-renders exactly with no codebook growth; all shard trees are hashed.",
            "resources": "28 workers, total compressed shard bytes <=20 GiB, no GPU, no C09/C10/model/graph/planner.",
        },
        "estimated_cost": {"compute": "28-process CPU Open3D CSG rendering and streaming Zarr compression", "wall_time_hours": 24.0, "host_ram_gb": 48, "gpu": 0, "disk_gb": 20},
        "retention": "Keep all 240 Zarr shards, codebooks, realized constructions, task manifest, source manifests, tests/logs/environment, RUN_STATE and SHA-256 seal. These assets are reusable for P1b, baselines, ablations and paper figures.",
        "failure_policy": "Any input drift, origin failure, code/valid mismatch, deterministic replay drift, count/resource failure or worker exception stops and seals FAIL. Do not delete frames, collapse memberships, alter geometry, raise tolerance or silently resume into the same run.",
    }
    write(CARD, card)
    tools = {
        "construction_supervisor": "src/mtare_topo/teacher/primitive_construction_supervisor.py",
        "shape_field": "src/mtare_topo/teacher/swept_superellipse_field.py",
        "variant_contract": "src/mtare_topo/teacher/geometry_variant_contract.py",
        "csg_backend": "src/mtare_topo/teacher/csg_mesh_provenance.py",
        "membership_contract": "src/mtare_topo/data/primitive_relation_dataset.py",
        "sensor_frame_contract": "src/mtare_topo/data/primitive_relation_sensor_export.py",
        "governance": "src/mtare_topo/governance.py",
        "pose_contract": "src/mtare_topo/data/gse_sensor_export.py",
        "runner": "tools/v3/run_primitive_relation_p1a_sensor_provenance_export_v1.py",
        "pilot": "tools/v3/check_primitive_relation_p1a_streaming_contract.py",
        "construction_tests": "tests/v3/unit/test_primitive_construction_supervisor.py",
        "provenance_tests": "tests/v3/unit/test_primitive_provenance_field.py",
        "shape_tests": "tests/v3/unit/test_swept_superellipse_field.py",
        "variant_tests": "tests/v3/unit/test_geometry_variant_contract.py",
        "membership_tests": "tests/v3/unit/test_primitive_relation_dataset.py",
        "pose_tests": "tests/v3/unit/test_gse_sensor_export.py",
        "sensor_tests": "tests/v3/unit/test_primitive_relation_sensor_export.py",
        "governance_tests": "tests/v3/unit/test_governance.py",
        "analytic_checker": "tools/v3/check_csg_mesh_provenance_contract.py",
        "preflight": "tools/v3/preflight.py", "create_run": "tools/v3/create_run.py",
    }
    inputs = [
        "results/gate2_representation/gate2_20260827_gse_corrected_causal_teacher_manifest_v1r_seed0/artifacts/traversal_manifest.jsonl",
        "results/gate2_representation/gate2_20260827_gse_corrected_causal_teacher_manifest_v1r_seed0/artifacts/evidence_sha256.txt",
        "results/gate3_semantics/gate3_20260830_geometry_variant_inventory_v1r2_seed0/artifacts/evidence_sha256.txt",
        "results/gate3_semantics/gate3_20260830_csg_mesh_provenance_acceleration_v1_seed0/artifacts/evidence_sha256.txt",
        "results/gate3_semantics/gate3_20260830_primitive_finite_cap_pose_qualification_v1_seed0/artifacts/frame_pose_corrections.json",
        "results/gate3_semantics/gate3_20260830_primitive_finite_cap_pose_qualification_v1_seed0/artifacts/evidence_sha256.txt",
        "results/gate3_semantics/gate3_20260830_primitive_slot_capacity_audit_v1r_seed0/artifacts/evidence_sha256.txt",
        "results/gate3_semantics/gate3_20260830_csg_mesh_provenance_acceleration_v1_seed0/config/environment.json",
    ]
    for world in worlds:
        inputs.extend(f"{MESH_REL}/{world}/primary/{name}" for name in ("graph.json", "splines.json", "geometry_parameters.json"))
    spec = {
        "schema_version": "v3_run_spec_v1", "gate": 3, "execution_phase": 3,
        "operation": "data_export", "date": "20260830",
        "slug": "primitive_relation_p1a_sensor_provenance_export_v1", "seed": 0,
        "question": "Can the complete three-geometry P1 sensor/provenance population be rendered losslessly, deterministically and within frozen resources so relation supervision can be derived without re-rendering?",
        "method": "Use the formally qualified endpoint poses, independent closed superellipse operands and accelerated CSG union-exit backend. Stream 16-frame compressed shards and encode every exact source membership set in a local uint16 codebook; preserve all realized constructions.",
        "baseline": "Historical native Poisson LiDAR lacks primitive identity; the C01 accelerated backend and one-frame three-realization streaming pilot are frozen prerequisites.",
        "fallback": "Any failure stops P1a. Do not fall back to nearest-primitive labels, native Poisson identity inference, frame deletion, source truncation or in-place resume.",
        "data_card": str(CARD.relative_to(PROJECT_ROOT)), "config_path": str(CARD.relative_to(PROJECT_ROOT)),
        "user_authorization": approval,
        "acceptance_criteria": list(card["metrics_and_pre_registered_gates"].values()),
        "expected_counts": {"worlds": 80, "tasks": 240, "fit_tasks": 180, "c07_tasks": 30, "c08_tasks": 30, "frames": 757290, "planned_sequences": 564378, "rays": 8723980800, "source_primitives": 8039, "variant_primitives": 24117, "paired_pose_corrections": 402, "unit_tests": 67, "analytic_cases": 4, "workers": 28, "optimizer_steps": 0, "model_inference_frames": 0, "c09_worlds_read": 0, "c10_worlds_read": 0, "graph_replays": 0},
        "expected_evidence": ["240 Zarr sensor/provenance shards, local lossless codebooks, realized construction documents, task/tree hashes, correction/traversal manifests, unit/analytic logs, environment, RUN_STATE and seal."],
        "estimated_cost": card["estimated_cost"],
        "frozen_tools": {name: {"path": path, "sha256": sha(PROJECT_ROOT / path)} for name, path in tools.items()},
        "frozen_inputs": {path: sha(PROJECT_ROOT / path) for path in inputs},
        "working_directory": str(PROJECT_ROOT),
        "command": [
            "/usr/bin/systemd-inhibit", "--what=sleep:shutdown", "--why=GSE P1a full sensor provenance export",
            "--mode=block", "/usr/bin/timeout", "--signal=INT", "--kill-after=600s", "259200s",
            "/tmp/mtare_gate4_meshing_sidecar_v1/bin/python",
            "tools/v3/run_primitive_relation_p1a_sensor_provenance_export_v1.py",
            "--spec", str(SPEC), "--run-dir", str(PROJECT_ROOT / f"results/gate3_semantics/{RUN_ID}"),
            "--workers", "28",
        ],
    }
    write(SPEC, spec); print(SPEC)


if __name__ == "__main__": main()
