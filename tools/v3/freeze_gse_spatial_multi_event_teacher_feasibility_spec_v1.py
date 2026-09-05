#!/usr/bin/env python3
"""Freeze the Data Card and run spec for one C01-C08 LOS feasibility proof."""

from __future__ import annotations

import hashlib
from pathlib import Path

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import write_json


RUN_ID = "gate3_20260828_gse_spatial_multi_event_teacher_feasibility_v1_seed0"
CARD = PROJECT_ROOT / "configs/v3/gate3/data_cards/gse_spatial_multi_event_teacher_feasibility_v1.json"
SPEC = PROJECT_ROOT / "configs/v3/gate3/gse_spatial_multi_event_teacher_feasibility_v1.json"
SIDECAR = "/tmp/mtare_gate4_meshing_sidecar_v1/bin/python"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    if CARD.exists() or SPEC.exists():
        raise RuntimeError("spatial multi-event feasibility card/spec already exists")
    approval = {
        "status": "APPROVED",
        "approved_by": "user-standing-authorization",
        "approved_at": "2026-08-28T00:00:00+08:00",
        "authorized_gates": [3],
        "authorized_operations": ["audit"],
        "scope": "One immutable zero-training C01-C08 native-mesh LOS feasibility proof. No full Teacher export, C09/C10, M-TARE, inference or threshold selection.",
        "confirmation_reference": "User instructed automatic best-choice execution without routine approval prompts.",
    }
    card = {
        "schema_version": "v3_data_card_v1",
        "card_id": "gse_spatial_multi_event_teacher_feasibility_v1",
        "title": "GSE spatial multi-event Teacher native-mesh LOS feasibility",
        "status": "APPROVED_FOR_ONE_IMMUTABLE_GSE_SPATIAL_MULTI_EVENT_TEACHER_FEASIBILITY_V1",
        "purpose": "Determine whether one causal LiDAR observation can have a unique bounded set of independently visible terminal and junction targets, replacing the failed mutually exclusive scene event label.",
        "approval": approval,
        "worlds": {
            "exact_ids": "The 80 sealed development parents S01-S10 with suffix C01-C08 from the train shard manifest.",
            "world_count": 80,
            "fit_worlds": 60,
            "selection_worlds": 20,
            "strict_test_worlds": 0,
            "mtare_worlds": 0,
        },
        "trajectories": {
            "source": "Frozen directed traversal poses already stored in the deduplicated GSE dataset; current pose is the fifth frame of every causal sequence.",
            "directed_traversals": 16078,
            "future_frames_used": 0,
            "trajectory_modification": "none",
        },
        "sampling": {
            "independent_sampling_unit": "one sealed procedural parent topology; observations within a world are correlated and are never counted as independent worlds",
            "raw_observations": 188126,
            "effective_observations": 188126,
            "causal_history_frames": 5,
            "arc_spacing_m": 1.0,
            "spatial_event_range_m": 50.0,
            "los_margin_m": 0.25,
            "fit_split": "C01-C06 only (60 worlds)",
            "selection_split": "C07-C08 only (20 worlds)",
            "split_role": "descriptive feasibility comparison only; no parameter fitting, capacity selection or threshold calibration",
        },
        "teacher": {
            "source": "TNG node degree and xyz define objective terminal (degree 1) and junction (degree at least 3) identities; frozen FTA defines sensor height; native Cano perception mesh raycasting independently qualifies line of sight.",
            "student_input": "none in this audit; future student may receive only five causal LiDAR frames",
            "target_interface": "event type plus forward-left-up relative xyz; objective identity remains Teacher-only",
            "target_height": "node axis z + fta_distance_m + 1.0 m",
            "visibility": "unoccluded when first native-mesh hit is absent or no closer than target distance minus 0.25 m",
            "full_training_target_export": False,
        },
        "leakage_audit": {
            "C09_read": False,
            "C10_read": False,
            "mtare_read": False,
            "test_world_used": False,
            "absolute_pose_student_input": False,
            "TNG_identity_student_input": False,
            "future_frame_input": False,
            "model_training_or_inference": False,
            "threshold_or_capacity_selected_from_results": False,
        },
        "methods": {
            "main": "For all 188,126 C01-C08 causal observations, raycast every degree-1 or degree>=3 TNG node within a predeclared 50 m sphere against the frozen native perception mesh, express each visible target in the current robot frame, and audit set cardinality, unique mapping, terminal-junction co-visibility and opposite-heading coverage.",
            "baseline": "The sealed mutually exclusive event classifier cannot represent terminal and junction simultaneously and failed both rare selection endpoints.",
            "fallback": "If feasibility fails, stop before training and revise the Teacher range/representation or collect additional causal views; do not tune the old classifier or planner.",
        },
        "acceptance": {
            "population": "Exactly 80 C01-C08 worlds and 188,126 five-frame causal observations; zero C09/C10/M-TARE reads.",
            "visibility": "Every objective degree-1/degree>=3 event identity is visible at least once; both fit and selection contain multi-event and terminal-junction co-visible observations.",
            "directional_diversity": "At least 90% of objective event identities are visible from a pair of headings separated by 120-240 degrees.",
            "bounded_unique_set": "Maximum visible set cardinality is at most 16; identities are unique and objective targets are separated by at least the fixed 0.25 m LOS margin.",
            "rare_failures": "All four rare rows see their objective terminal; rows 44299 and 110361 each simultaneously see terminal and junction.",
            "isolation": "Zero optimization, inference, model update, threshold selection, student-input read or full Teacher export.",
        },
        "estimated_cost": {
            "compute": "Serial CPU/Open3D raycasting over 80 native meshes plus sealed shard integrity hashing; two CPU threads, zero GPU.",
            "wall_time_hours": 0.5,
            "host_ram_gb": 4,
            "gpu_memory_gb": 0,
            "disk_gb": 0.2,
            "gpu": "none",
        },
    }
    write_json(CARD, card)

    dataset = "results/gate2_representation/gate2_20260824_gse_deduplicated_dataset_export_v1_seed0"
    meshes = "results/gate0_baseline/gate0_20260811_cano_100_parent_perception_mesh_m1r_sanitized_assets_seed0"
    low = "results/gate3_semantics/gate3_20260828_gse_low_support_endpoint_observability_audit_v1_seed0"
    inputs = [
        f"{dataset}/RUN_STATE.json",
        f"{dataset}/metrics/summary.json",
        f"{dataset}/artifacts/evidence_sha256.txt",
        f"{dataset}/artifacts/shard_manifest.json",
        f"{dataset}/artifacts/world_dataset_summary.jsonl",
        f"{meshes}/RUN_STATE.json",
        f"{meshes}/metrics/summary.json",
        f"{meshes}/artifacts/evidence_sha256.txt",
        f"{meshes}/artifacts/mesh_manifest.json",
        f"{low}/RUN_STATE.json",
        f"{low}/artifacts/evidence_sha256.txt",
        f"{low}/artifacts/audit/low_selection_rows.json",
        "configs/v3/gate4/environments/gate4_meshing_sidecar_v1.json",
    ]
    tools = {
        "data_card": str(CARD.relative_to(PROJECT_ROOT)),
        "teacher_interface": "src/mtare_topo/teacher/gse_spatial_multi_event_teacher.py",
        "executor": "tools/v3/execute_gse_spatial_multi_event_teacher_feasibility_v1.py",
        "runner": "tools/v3/run_gse_spatial_multi_event_teacher_feasibility_v1.py",
        "freezer": "tools/v3/freeze_gse_spatial_multi_event_teacher_feasibility_spec_v1.py",
        "tests": "tests/v3/unit/test_gse_spatial_multi_event_teacher.py",
        "governance": "src/mtare_topo/governance.py",
        "preflight": "tools/v3/preflight.py",
        "create_run": "tools/v3/create_run.py",
    }
    spec = {
        "schema_version": "v3_run_spec_v1",
        "gate": 3,
        "execution_phase": 3,
        "date": "20260828",
        "slug": "gse_spatial_multi_event_teacher_feasibility_v1",
        "seed": 0,
        "operation": "audit",
        "question": "Can native-mesh LOS define unique bounded-cardinality relative event sets over every C01-C08 causal observation without leakage?",
        "method": card["methods"]["main"],
        "baseline": card["methods"]["baseline"],
        "fallback": card["methods"]["fallback"],
        "data_card": str(CARD.relative_to(PROJECT_ROOT)),
        "config_path": str(CARD.relative_to(PROJECT_ROOT)),
        "user_authorization": approval,
        "acceptance_criteria": list(card["acceptance"].values()),
        "expected_counts": {
            "development_worlds": 80,
            "fit_worlds": 60,
            "selection_worlds": 20,
            "observations": 188126,
            "directed_traversals": 16078,
            "rare_rows": 4,
            "optimizer_steps": 0,
            "model_inference_frames": 0,
            "full_teacher_export_rows": 0,
            "c09_worlds_read": 0,
            "c10_worlds_read": 0,
            "mtare_worlds_read": 0,
        },
        "expected_evidence": [
            "80-row world feasibility summary, complete objective event identity coverage, exact four-row native-mesh LOS evidence, cardinality/coverage figure in PNG/PDF/SVG with source JSON, raw log, metrics, RUN_STATE and SHA-256 seal."
        ],
        "estimated_cost": card["estimated_cost"],
        "frozen_inputs": {path: _sha256(PROJECT_ROOT / path) for path in sorted(inputs)},
        "frozen_tools": {
            name: {"path": path, "sha256": _sha256(PROJECT_ROOT / path)}
            for name, path in tools.items()
        },
        "working_directory": str(PROJECT_ROOT),
        "command": [
            "/usr/bin/timeout",
            "3600s",
            SIDECAR,
            "tools/v3/run_gse_spatial_multi_event_teacher_feasibility_v1.py",
            "--spec",
            str(SPEC),
            "--run-dir",
            str(PROJECT_ROOT / f"results/gate3_semantics/{RUN_ID}"),
        ],
    }
    write_json(SPEC, spec)
    print(CARD.relative_to(PROJECT_ROOT))
    print(SPEC.relative_to(PROJECT_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
