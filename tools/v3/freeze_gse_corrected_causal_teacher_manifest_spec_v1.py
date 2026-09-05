#!/usr/bin/env python3
"""Freeze the corrected causal Teacher Data Card and one-shot run spec."""

from __future__ import annotations

import hashlib
from pathlib import Path

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, write_json


CARD_PATH = PROJECT_ROOT / "configs/v3/gate2/data_cards/gse_corrected_causal_teacher_manifest_v1.json"
SPEC_PATH = PROJECT_ROOT / "configs/v3/gate2/gse_corrected_causal_teacher_manifest_v1.json"
SOURCE_CARD = PROJECT_ROOT / "configs/v3/gate3/data_cards/gse_directional_structural_event_training_v1.json"
RUN_ID = "gate2_20260826_gse_corrected_causal_teacher_manifest_v1_seed0"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _record(path: str) -> dict[str, str]:
    return {"path": path, "sha256": _sha256(PROJECT_ROOT / path)}


def main() -> int:
    source_card = load_json(SOURCE_CARD)
    worlds = {
        "train": source_card["worlds"]["train"],
        "validation": source_card["worlds"]["validation"],
        "ssl": [],
        "normalization": [],
        "teacher_calibration": [],
        "threshold_calibration": [],
        "augmentation_tuning": [],
        "checkpoint_selection": [],
        "strict_test": source_card["worlds"]["strict_test"],
        "reserved_c09": source_card["worlds"]["reserved_c09"],
        "mtare_benchmark": [],
    }
    card = {
        "schema_version": "v3_data_card_v1",
        "card_id": "gse_corrected_causal_teacher_manifest_v1",
        "status": "APPROVED_FOR_ONE_IMMUTABLE_CORRECTED_TEACHER_GENERATION",
        "purpose": "Generate a corrected C01-C08 Teacher manifest by replacing every historical geometry-transition label with the sealed persistent bidirectional causal change-point labels while preserving all other observation fields and protected node-event priority.",
        "approval": {
            "status": "APPROVED",
            "approved_by": "user",
            "approved_at": "2026-08-26T23:58:00+08:00",
            "authorized_operations": ["teacher_generation"],
            "authorized_gates": [2],
            "scope": "One immutable corrected Teacher manifest over exactly 80 C01-C08 worlds, 16078 directed traversals and 188126 observations. Apply exactly 1031 causal labels and suppress exactly 59 proof labels by junction/terminal priority. Zero C09/C10/M-TARE/model inference/training.",
            "confirmation_reference": "User explicitly selected A/a and authorized continuing the GSE-Graph paper pipeline without repeated routine approvals."
        },
        "source": {
            "raw_sources": [
                "Sealed C01-C08 rows from PASS_GSE_TEACHER_MANIFEST_V1; all non-event observation fields remain byte-value identical.",
                "Sealed PASS_GSE_CAUSAL_CHANGE_POINT_PROOF_V1 with 1090 proof labels, 76 persistent bidirectional causal identities and exact source hashes.",
                "The prior GSE directional training Data Card supplies only the already frozen 60/20/10 world lists and 80 trajectory inventory rows."
            ],
            "license_or_allowed_use": "Local research use of the sealed procedural Cano assets and derived objective Teacher manifests with full provenance.",
            "old_teacher_run": "results/gate2_representation/gate2_20260824_gse_teacher_manifest_v1_seed0",
            "causal_proof_run": "results/gate3_semantics/gate3_20260826_gse_causal_change_point_proof_v1_seed0"
        },
        "worlds": worlds,
        "trajectories": source_card["trajectories"],
        "sampling": {
            "raw_frame_count": 252430,
            "effective_sample_count": 188126,
            "effective_structure_event_count": 37162,
            "spatial_interval_m": 1.0,
            "rule": "Retain every existing five-frame C01-C08 observation at one-metre directed-traversal spacing. Reset all 16867 historical transition observations to corridor, then apply exactly 1031 sealed causal labels after protecting 39 junction and 20 terminal rows; no frame, edge, traversal or world is removed.",
            "structure_event_counts": {
                "corridor": 150964,
                "junction": 26608,
                "terminal": 7525,
                "turn": 1998,
                "geometry_transition": 1031
            },
            "identity_counts": {
                "junction": 563,
                "terminal": 503,
                "turn": 392,
                "geometry_transition": 76
            },
            "association_pair_counts": {
                "cross_traversal_positive": 37029,
                "same_traversal_revisit_positive": 63,
                "same_event_geometry_hard_negative": 36972
            },
            "independent_unit": "Topology parent for split isolation and structural identity for event diversity; the 1031 adjacent labels represent 76 independent change-point identities and are not counted as 1031 independent structures."
        },
        "teacher": {
            "source": "Junction/terminal identities remain TNG-node objective labels; turn identities remain canonical physical-edge clusters; geometry transitions come only from the sealed persistent bidirectional past-only causal proof.",
            "valid_mask": "All width/height validity, slope, curvature, frame references, pose/traversal metadata and observation IDs remain unchanged. Corridor has no identity; every structural row has exactly one identity.",
            "planner_consistency_plan": "A structure node may be proposed only by a stable learned event. Degree-two change-points use back-projected node identity; terminal/junction keep priority. Edge creation remains forbidden until physical traversal evidence is complete.",
            "priority": "junction/terminal > corrected geometry_transition > turn > corridor",
            "student_input": "No mesh, TNG, spline, identity, future frame, C09/C10 or M-TARE information enters the student input."
        },
        "split": {
            "world_disjoint": True,
            "trajectory_disjoint": True,
            "fit": "C01-C06: 60 worlds; corrected manifest only, no training in this operation.",
            "selection": "C07-C08: 20 worlds; corrected manifest only, no threshold or checkpoint selection in this operation.",
            "strict_test": "C10 and M-TARE remain unread.",
            "historical_pollution_audit": "C09 failures motivated the Teacher audit, but no C09 data, labels, model outputs or graph results enter this generation. C10 has never been read. The corrected labels and all scales were frozen by the C01-C08 proof before this operation."
        },
        "leakage_audit": {
            "test_excluded_from_supervised_training": True,
            "test_excluded_from_ssl": True,
            "test_excluded_from_normalization": True,
            "test_excluded_from_teacher_calibration": True,
            "test_excluded_from_threshold_calibration": True,
            "test_excluded_from_augmentation_tuning": True,
            "test_excluded_from_checkpoint_selection": True,
            "future_frames_excluded": True,
            "c09_records_consumed": 0,
            "c10_records_consumed": 0,
            "mtare_records_consumed": 0
        },
        "estimated_cost": {
            "disk_gb": 0.5,
            "wall_time_hours": 0.1,
            "compute": "Serial CPU manifest rewrite and deterministic association-pair regeneration; zero GPU, raycast, sensor export, inference or optimizer steps."
        },
        "evidence": {
            "machine_metrics": "Exact event/identity/pair/action counts, all observations/traversals, immutable-field equality, source verification, logs and seal.",
            "complete_visual_review": "No new scientific visualization is required; the sealed causal proof paper figure remains the visual evidence. Preserve concise README and machine-readable summaries.",
            "failure_policy": "Any count, identity, immutable field, source, priority, split, environment or resource drift seals FAIL. Do not patch or retry the run."
        },
        "retention": "Retain the full corrected observations, traversal manifest, regenerated pairs, identity summaries, label audit, world summaries, metrics, environment, raw log, RUN_STATE and complete SHA-256 seal."
    }
    CARD_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_json(CARD_PATH, card)

    tools = {
        "data_card": _record(str(CARD_PATH.relative_to(PROJECT_ROOT))),
        "runner": _record("tools/v3/run_gse_corrected_causal_teacher_manifest_v1.py"),
        "executor": _record("tools/v3/execute_gse_corrected_causal_teacher_manifest_v1.py"),
        "manifest_builder": _record("src/mtare_topo/data/gse_corrected_teacher_manifest.py"),
        "association_teacher": _record("src/mtare_topo/teacher/gse_association_teacher.py"),
        "governance": _record("src/mtare_topo/governance.py"),
        "preflight": _record("tools/v3/preflight.py"),
        "freezer": _record("tools/v3/freeze_gse_corrected_causal_teacher_manifest_spec_v1.py")
    }
    input_paths = (
        "configs/v3/gate3/data_cards/gse_directional_structural_event_training_v1.json",
        "results/gate2_representation/gate2_20260824_gse_teacher_manifest_v1_seed0/RUN_STATE.json",
        "results/gate2_representation/gate2_20260824_gse_teacher_manifest_v1_seed0/artifacts/evidence_sha256.txt",
        "results/gate2_representation/gate2_20260824_gse_teacher_manifest_v1_seed0/artifacts/teacher_observations.jsonl",
        "results/gate2_representation/gate2_20260824_gse_teacher_manifest_v1_seed0/artifacts/traversal_manifest.jsonl",
        "results/gate3_semantics/gate3_20260826_gse_causal_change_point_proof_v1_seed0/RUN_STATE.json",
        "results/gate3_semantics/gate3_20260826_gse_causal_change_point_proof_v1_seed0/artifacts/evidence_sha256.txt",
        "results/gate3_semantics/gate3_20260826_gse_causal_change_point_proof_v1_seed0/artifacts/causal_change_point_labels.jsonl"
    )
    spec = {
        "schema_version": "v3_run_spec_v1",
        "gate": 2,
        "execution_phase": 3,
        "date": "20260826",
        "slug": "gse_corrected_causal_teacher_manifest_v1",
        "operation": "teacher_generation",
        "question": "Can the sealed causal change-point proof be integrated into a complete C01-C08 Teacher manifest with exact priority, identity and association consistency and no non-Teacher field drift?",
        "method": "Read every sealed C01-C08 observation once. Reset every historical geometry-transition event/identity to corridor/none. Apply the 1090 sealed proof labels by traversal and sequence key, suppress exactly 39 junction and 20 terminal overlaps, and apply exactly 1031 corrected geometry-transition labels. Preserve every other field. Regenerate all association pairs and uniform structural identity summaries deterministically within each parent.",
        "baseline": "The sealed V1 Teacher manifest with 16867 short-frame transition observations and 3035 transition identities remains an immutable failed-label baseline.",
        "fallback": "If any exact count, priority, identity, pair, immutable-field or source check fails, seal FAIL and stop before training. Do not edit the old manifest or tune the proof.",
        "seed": 0,
        "config_path": str(CARD_PATH.relative_to(PROJECT_ROOT)),
        "data_card": str(CARD_PATH.relative_to(PROJECT_ROOT)),
        "working_directory": str(PROJECT_ROOT),
        "python_executable": "/home/zeng-workstation/anaconda3/bin/python",
        "command": [
            "/usr/bin/timeout", "--signal=INT", "--kill-after=60s", "600s",
            "/home/zeng-workstation/anaconda3/bin/python",
            "tools/v3/run_gse_corrected_causal_teacher_manifest_v1.py",
            "--spec", str(SPEC_PATH),
            "--run-dir", str(PROJECT_ROOT / "results/gate2_representation" / RUN_ID)
        ],
        "user_authorization": {
            "status": "APPROVED",
            "approved_by": "user",
            "approved_at": "2026-08-26T23:58:00+08:00",
            "scope": "One immutable corrected C01-C08 Teacher generation bound to research decision A; 80 worlds, 16078 traversals, 188126 observations, zero C09/C10/M-TARE/model/training.",
            "confirmation_reference": "User explicitly selected A/a and requested autonomous continuation of the fixed GSE-Graph paper pipeline."
        },
        "data_scope": {
            "fit_worlds_C01_C06": 60,
            "selection_worlds_C07_C08": 20,
            "c09_worlds_consumed": 0,
            "strict_test_worlds_read": 0,
            "mtare_worlds_read": 0,
            "directed_traversals": 16078,
            "observations": 188126,
            "proof_labels": 1090,
            "applied_change_point_labels": 1031,
            "change_point_identities": 76,
            "training_samples": 0,
            "model_inference_frames": 0,
            "optimizer_steps": 0
        },
        "resource_contract": {
            "executor_timeout_seconds": 300,
            "outer_timeout_seconds": 600,
            "disk_limit_bytes": 536870912,
            "minimum_free_disk_bytes": 2147483648,
            "cpu_threads": 2,
            "gpu_count": 0,
            "sensor_payload_generated": False
        },
        "acceptance_criteria": [
            "Exact 80 worlds, 16078 traversals, 188126 unique observation/global-sequence identities and no non-event field changes.",
            "Exact event counts corridor/junction/terminal/turn/transition = 150964/26608/7525/1998/1031.",
            "Exact identity counts junction/terminal/turn/transition = 563/503/392/76; no old geometry-transition identity remains.",
            "All 1090 proof labels are consumed once, exactly 1031 apply and exactly 39 junction plus 20 terminal labels suppress by priority.",
            "Regenerated association pairs exactly equal 37029 cross-traversal positive, 63 same-traversal revisit positive and 36972 hard negative, total 74064.",
            "Sources verify before/after; C09/C10/M-TARE/model/training/optimizer reads or steps are zero; result remains within 512 MiB and seals completely."
        ],
        "stop_conditions": [
            "Any source/tool/card/environment/count/schema/identity/priority/pair drift or non-event field change.",
            "Any C09/C10/M-TARE/model/training access, timeout, disk overrun, retry or old-run overwrite."
        ],
        "expected_evidence": [
            "Complete corrected observations, traversal manifest, regenerated association pairs and typed structural identity summaries.",
            "Every proof-label application/suppression row and eighty world summaries.",
            "Environment, command, raw log, exact metrics, RUN_STATE and complete SHA-256 seal."
        ],
        "estimated_cost": {
            "disk_gb": 0.5,
            "wall_time_hours": 0.1,
            "compute": "Serial CPU manifest rewrite and pair regeneration; zero GPU/raycast/sensor/inference/training."
        },
        "frozen_tools": tools,
        "frozen_inputs": {path: _sha256(PROJECT_ROOT / path) for path in input_paths}
    }
    write_json(SPEC_PATH, spec)
    print(CARD_PATH)
    print(SPEC_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
