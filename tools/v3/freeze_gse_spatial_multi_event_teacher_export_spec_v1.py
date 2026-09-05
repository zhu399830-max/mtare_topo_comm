#!/usr/bin/env python3
"""Freeze the Gate-2 Data Card and spec for the spatial event set export."""

from __future__ import annotations

import hashlib
from pathlib import Path

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, validate_data_card, write_json


RUN_ID = "gate2_20260828_gse_spatial_multi_event_teacher_export_v1_seed0"
CARD = PROJECT_ROOT / "configs/v3/gate2/data_cards/gse_spatial_multi_event_teacher_export_v1.json"
SPEC = PROJECT_ROOT / "configs/v3/gate2/gse_spatial_multi_event_teacher_export_v1.json"
BASE_CARD = PROJECT_ROOT / "configs/v3/gate2/data_cards/gse_deduplicated_dataset_export_v1.json"
SIDECAR = "/tmp/mtare_gate4_meshing_sidecar_v1/bin/python"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _suffix(parent_id: str) -> int:
    return int(parent_id.rsplit("_C", 1)[1])


def main() -> int:
    if CARD.exists() or SPEC.exists():
        raise RuntimeError("spatial multi-event Teacher export card/spec already exists")
    base = load_json(BASE_CARD)
    development = sorted(parent_id for parent_id in base["worlds"]["train"] if 1 <= _suffix(parent_id) <= 8)
    fit = [parent_id for parent_id in development if _suffix(parent_id) <= 6]
    selection = [parent_id for parent_id in development if _suffix(parent_id) >= 7]
    c09 = sorted(base["worlds"]["validation"])
    approval = {
        "status": "APPROVED",
        "approved_by": "user-standing-authorization",
        "approved_at": "2026-08-28T00:00:00+08:00",
        "scope": "One immutable Gate-2 Teacher generation over exactly 60 C01-C06 fit and 20 C07-C08 selection worlds. Export 188126 fixed-16-slot relative event sets; no LiDAR duplication, training, inference, C09/C10 or M-TARE read.",
        "authorized_operations": ["teacher_generation"],
        "authorized_gates": [2],
        "confirmation_reference": "User instructed automatic best-choice execution without routine approval prompts and authorized the fixed GSE-Graph paper scope.",
    }
    trajectories = []
    for row in base["trajectories"]:
        parent_id = str(row["world"])
        if parent_id not in development:
            continue
        updated = dict(row)
        updated["split"] = "train" if parent_id in fit else "validation"
        trajectories.append(updated)
    card = dict(base)
    card.update(
        {
            "card_id": "gse_spatial_multi_event_teacher_export_v1",
            "status": "APPROVED_FOR_ONE_IMMUTABLE_GSE_SPATIAL_MULTI_EVENT_TEACHER_EXPORT_V1",
            "purpose": "Losslessly export the passed native-mesh-LOS terminal/junction event sets as fixed 16-slot robot-relative targets aligned to every C01-C08 causal observation before any set-prediction model is designed.",
            "approval": approval,
            "source": {
                "raw_sources": [
                    "Sealed PASS 80-world C01-C08 deduplicated GSE dataset poses and global sequence identities.",
                    "Sealed native Cano perception meshes, TNG graphs and FTA geometry parameters.",
                    "Sealed PASS spatial multi-event feasibility V1R with exact per-world candidate/visible/blocked counts and cardinality histograms.",
                ],
                "license_or_allowed_use": base["source"]["license_or_allowed_use"],
            },
            "worlds": {
                "train": fit,
                "validation": selection,
                "ssl": [],
                "normalization": [],
                "teacher_calibration": [],
                "threshold_calibration": [],
                "augmentation_tuning": [],
                "checkpoint_selection": [],
                "strict_test": c09 + list(base["worlds"]["strict_test"]),
            },
            "trajectories": trajectories,
            "sampling": {
                "raw_frame_count": 252430,
                "effective_sample_count": 188126,
                "effective_structure_event_count": 133055,
                "spatial_interval_m": 1.0,
                "spatial_interval_interpretation": "Each sample is the fifth pose of one existing five-frame causal sequence at one-metre directed-traversal arc spacing. Parent worlds, not correlated frames, are independent units.",
                "structure_event_counts": {"terminal": 30789, "junction": 102266},
                "rule": "Keep all 188126 C01-C08 observations including 81069 zero-event rows. Independently native-mesh-raycast every degree-1/degree>=3 TNG node within 50 m, sort visible events by distance then identity, write at most 16 slots and fail instead of truncating.",
                "exact_counts": {
                    "fit_worlds": 60,
                    "selection_worlds": 20,
                    "physical_edges": 8039,
                    "directed_traversals": 16078,
                    "unique_frames": 252430,
                    "causal_sequences": 188126,
                    "objective_event_identities": 1076,
                    "visible_event_tokens": 133055,
                    "zero_event_rows": 81069,
                    "multi_event_rows": 23964,
                    "maximum_observed_cardinality": 5,
                    "fixed_export_capacity": 16,
                },
            },
            "teacher": {
                "source": "TNG degree/xyz defines terminal and junction identity/axis; current fifth-frame sealed pose defines the robot frame; native perception mesh first-hit rays define independent LOS with fixed 0.25 m margin.",
                "valid_mask": "event_mask marks real slots; type and identity padding are -1 and continuous padding is zero. Zero-event observations remain valid samples with all event slots masked false. Any set above 16 is a hard failure.",
                "planner_consistency_plan": "Identity indices and TNG remain Teacher-only. The future student joins global_sequence_index to the existing five causal LiDAR frames and receives no world, pose, identity or future frame. Geometry-transition remains an edge profile, not a structural event token in this export.",
            },
            "split": {
                "world_disjoint": True,
                "trajectory_disjoint": True,
                "historical_pollution_audit": "C01-C06 are fit and C07-C08 are selection. C09, C10 and M-TARE are explicitly moved to the forbidden strict_test role for this corrective export and are never opened.",
            },
            "leakage_audit": dict(base["leakage_audit"]),
            "estimated_cost": {
                "disk_gb": 0.2,
                "wall_time_hours": 0.5,
                "compute": "Serial CPU/Open3D LOS replay plus Zarr export in frozen Python 3.12.3/NumPy 1.26.4/Open3D 0.19.0/Zarr 2.18.7; two CPU threads, <=4 GiB RSS, zero GPU, model inference or optimizer steps.",
            },
            "evidence": {
                "machine_metrics": "80 hashed Teacher shards, 188126 unique global identities, exact 133055 token/type counts, per-world replay equality to the feasibility run, identity map, logs, RUN_STATE and SHA-256 seal.",
                "complete_visual_review": "Retain PNG/PDF/SVG cardinality and robot-relative event density figure with machine-readable histogram source for the paper and export audit.",
                "failure_policy": "Stop and seal on source/hash/count/split/identity/order/shape/mask/truncation/replay/resource drift or any C09/C10/M-TARE/model access. Never drop zero-event rows or resize capacity after observing results.",
            },
        }
    )
    validation = validate_data_card(card)
    if not validation.passed:
        raise RuntimeError(f"generated spatial event Data Card invalid: {validation.errors}")
    write_json(CARD, card)

    dataset = "results/gate2_representation/gate2_20260824_gse_deduplicated_dataset_export_v1_seed0"
    meshes = "results/gate0_baseline/gate0_20260811_cano_100_parent_perception_mesh_m1r_sanitized_assets_seed0"
    feasibility = "results/gate3_semantics/gate3_20260828_gse_spatial_multi_event_teacher_feasibility_v1r_seed0"
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
        f"{feasibility}/RUN_STATE.json",
        f"{feasibility}/metrics/summary.json",
        f"{feasibility}/artifacts/evidence_sha256.txt",
        f"{feasibility}/artifacts/audit/summary.json",
        f"{feasibility}/artifacts/audit/world_feasibility_summary.jsonl",
        f"{feasibility}/artifacts/audit/event_identity_coverage.jsonl",
        "configs/v3/gate4/environments/gate4_meshing_sidecar_v1.json",
    ]
    tools = {
        "data_card": str(CARD.relative_to(PROJECT_ROOT)),
        "teacher_interface": "src/mtare_topo/teacher/gse_spatial_multi_event_teacher.py",
        "feasibility_dependency": "tools/v3/execute_gse_spatial_multi_event_teacher_feasibility_v1.py",
        "executor": "tools/v3/execute_gse_spatial_multi_event_teacher_export_v1.py",
        "runner": "tools/v3/run_gse_spatial_multi_event_teacher_export_v1.py",
        "freezer": "tools/v3/freeze_gse_spatial_multi_event_teacher_export_spec_v1.py",
        "tests": "tests/v3/unit/test_gse_spatial_multi_event_teacher_export.py",
        "governance": "src/mtare_topo/governance.py",
        "preflight": "tools/v3/preflight.py",
        "create_run": "tools/v3/create_run.py",
    }
    spec = {
        "schema_version": "v3_run_spec_v1",
        "gate": 2,
        "execution_phase": 2,
        "date": "20260828",
        "slug": "gse_spatial_multi_event_teacher_export_v1",
        "seed": 0,
        "operation": "teacher_generation",
        "question": "Can the passed C01-C08 spatial event sets be exported losslessly into a fixed 16-slot causal Teacher without truncation or leakage?",
        "method": "Replay the frozen native-mesh LOS definition on all C01-C08 causal poses, sort visible terminal/junction targets by distance and identity, and write exact 16-slot masked robot-relative Zarr targets joined only by global_sequence_index.",
        "baseline": "The old one-class-per-scene Teacher cannot represent terminal and junction simultaneously; feasibility V1R is the exact count oracle for export replay.",
        "fallback": "On any mismatch, stop before model design and inspect export alignment only; do not alter LOS, capacity, rows, split or targets.",
        "data_card": str(CARD.relative_to(PROJECT_ROOT)),
        "config_path": str(CARD.relative_to(PROJECT_ROOT)),
        "user_authorization": approval,
        "acceptance_criteria": [
            "Exactly 80 worlds, 188126 unique rows, 1076 identities and 133055 visible tokens with terminal/junction counts 30789/102266.",
            "All 80 per-world candidate, visible, blocked and cardinality statistics exactly replay feasibility V1R.",
            "Sixteen slots, zero truncation, 81069 zero-event rows retained, unique identity per set and exact source global identity alignment.",
            "Zero training, inference, normalization, threshold selection, C09, C10 or M-TARE reads.",
        ],
        "expected_counts": {
            "fit_worlds": 60,
            "selection_worlds": 20,
            "observations": 188126,
            "event_identities": 1076,
            "visible_event_tokens": 133055,
            "terminal_tokens": 30789,
            "junction_tokens": 102266,
            "zero_event_rows": 81069,
            "multi_event_rows": 23964,
            "maximum_slots": 16,
            "optimizer_steps": 0,
            "model_inference_frames": 0,
            "c09_worlds_read": 0,
            "c10_worlds_read": 0,
            "mtare_worlds_read": 0,
        },
        "expected_evidence": [
            "80 hashed Zarr Teacher shards, identity map, exact per-world replay manifest, summary, robot-relative event density/cardinality PNG/PDF/SVG with source, logs, RUN_STATE and SHA-256 seal."
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
            "tools/v3/run_gse_spatial_multi_event_teacher_export_v1.py",
            "--spec",
            str(SPEC),
            "--run-dir",
            str(PROJECT_ROOT / f"results/gate2_representation/{RUN_ID}"),
        ],
    }
    write_json(SPEC, spec)
    print(CARD.relative_to(PROJECT_ROOT))
    print(SPEC.relative_to(PROJECT_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
