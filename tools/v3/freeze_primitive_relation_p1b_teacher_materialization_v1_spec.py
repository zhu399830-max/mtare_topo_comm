#!/usr/bin/env python3
"""Freeze P1b only after the immutable P1a export has scientifically passed."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from _bootstrap import PROJECT_ROOT


CARD = PROJECT_ROOT / "configs/v3/gate3/data_cards/primitive_relation_p1b_teacher_materialization_v1.json"
SPEC = PROJECT_ROOT / "configs/v3/gate3/primitive_relation_p1b_teacher_materialization_v1.json"
RUN_ID = "gate3_20260830_primitive_relation_p1b_teacher_materialization_v1_seed0"
CARD_ID = "primitive_relation_p1b_teacher_materialization_v1"
SLUG = "primitive_relation_p1b_teacher_materialization_v1"
RUNNER = "tools/v3/run_primitive_relation_p1b_teacher_materialization_v1.py"
REAL_SHARD_PILOT = "tools/v3/check_primitive_relation_p1b_real_shard_contract.py"
CORRECTIVE_OF: Path | None = None
P1A_DIRECT = PROJECT_ROOT / "results/gate3_semantics/gate3_20260830_primitive_relation_p1a_sensor_provenance_export_v1_seed0"
P1A_CORRECTED = PROJECT_ROOT / "results/gate3_semantics/gate3_20260830_primitive_relation_p1a_lossless_storage_corrective_v1_seed0"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def canonical_p1a_run() -> Path:
    for candidate in (P1A_CORRECTED, P1A_DIRECT):
        summary_path = candidate / "metrics/summary.json"; state_path = candidate / "RUN_STATE.json"
        if not summary_path.is_file() or not state_path.is_file():
            continue
        summary = json.loads(summary_path.read_text()); state = json.loads(state_path.read_text())
        if summary.get("scientific_pass") and state.get("state") == "COMPLETED" and state.get("error") is None:
            return candidate
    raise RuntimeError("cannot freeze P1b before a sealed direct or losslessly corrected P1a scientific PASS")


def main() -> None:
    p1a_run = canonical_p1a_run()
    p1a_summary = json.loads((p1a_run / "metrics/summary.json").read_text())
    p1a_state = json.loads((p1a_run / "RUN_STATE.json").read_text())
    if not p1a_summary.get("scientific_pass") or p1a_state.get("state") != "COMPLETED":
        raise RuntimeError("cannot freeze P1b before sealed P1a scientific PASS")
    p1a_card = json.loads((p1a_run / "config/data_card.json").read_text())
    approval = {
        "status": "APPROVED", "approved_by": "user-standing-authorization",
        "approved_at": "2026-08-30T00:00:00+08:00", "authorized_gates": [3],
        "authorized_operations": ["teacher_generation"],
        "confirmation_reference": "User instructed continuous autonomous execution and selection of the evidence-strongest in-plan method; sealed P1a is the required prerequisite.",
        "scope": "One immutable P1b derivation from the exact sealed C01-C08 P1a shards. No new rendering, C09/C10, model, optimizer, graph, planner or M-TARE.",
    }
    card = {
        "schema_version": "v3_data_card_v1", "card_id": CARD_ID,
        "status": "APPROVED_FOR_ONE_IMMUTABLE_P1B_TEACHER_MATERIALIZATION", "approval": approval,
        "purpose": "Convert P1a ray-level construction provenance into permutation-safe five-frame supervision for bottom-up primitive geometry and primitive-relation learning.",
        "worlds": p1a_card["worlds"],
        "source": {
            "raw_sources": ["sealed P1a LiDAR/provenance shards", "sealed P1a local source-set codebooks", "sealed P1a realized construction programs", "sealed causal traversal references"],
            "license_or_allowed_use": p1a_card["source"]["license_or_allowed_use"],
            "worlds": ["80 independent topology parents, C01-C08 only"],
        },
        "trajectories": p1a_card["trajectories"],
        "sampling": {
            "independent_sampling_units": "80 topology parents; three paired geometry realizations remain repeated measures inside each parent.",
            "raw_frame_count": 757290, "raw_five_frame_sequence_count": 564378, "effective_sample_count": 80,
            "effective_structure_event_count": p1a_card["sampling"]["effective_structure_event_count"],
            "structure_event_counts": p1a_card["sampling"]["structure_event_counts"],
            "spatial_interval_m": 1.0, "temporal_window_frames": 5, "maximum_primitive_slots": 32,
            "partition_sequences": {"fit_c01_c06": 426552, "selection_c07": 64644, "development_transfer_c08": 73182},
            "rule": "For every causal five-frame sequence, preserve every visible source primitive, crop its swept axis to observed support, regress its endpoint cross-section, and label temporal correspondence, endpoint attachment, and disconnected angular overlap without truncation.",
        },
        "split": p1a_card["split"],
        "teacher": {
            "source": "Exact procedural construction identity propagated through the qualified CSG ray exits; no human junction rule or predicted exit class generates the labels.",
            "labels": "32-slot primitive axis/shape/support targets, five-frame visibility/correspondence, endpoint attachment, disconnected angular overlap, and causal relative odometry derived without retaining absolute pose in the student input.",
            "student_forbidden_inputs": "Pose, parent/world, traversal, primitive identity, TNG, construction graph and future frames remain Teacher/evaluation metadata only.",
            "unexplored_port_policy": "Not a perception label: explored/unexplored is deterministic robot execution state and will be added only by the online graph state machine.",
            "valid_mask": "A primitive slot is valid iff at least one qualified P1a return from that exact construction primitive supports the causal five-frame window; unused slots are masked and never treated as negative geometry.",
            "planner_consistency_plan": "No graph or planner executes in P1b; planner consistency remains deferred until the learned primitive relations pass offline perception and graph qualification.",
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
            "population": "Exactly 240 tasks, 80 parents, 757,290 source frames, 564,378 paired-variant five-frame sequences and 24,117 realized primitives.",
            "capacity": "Every sequence contains at most the frozen 32 visible primitive slots; any overflow fails without truncation.",
            "relations": "Endpoint attachment, disconnected visual overlap and temporal dustbin labels are each nonempty globally.",
            "losslessness": "Geometry and relation slot indices agree exactly; source membership, endpoint incidence and overlap are compacted without information loss.",
            "provenance": "Every P1a evidence-manifest entry is rehashed before derivation; every P1b shard is tree-hashed and first-row write/reopen exact.",
            "resources": "12 CPU workers, compressed P1b output <=12 GiB, no GPU/model/graph/planner and zero C09/C10 reads.",
        },
        "estimated_cost": {"compute": "12-process CPU provenance reduction and five-frame target materialization", "wall_time_hours": 24.0, "host_ram_gb": 48, "gpu": 0, "disk_gb": 12},
        "retention": "Keep compact teacher shards, task manifest, unit log, P1a seal copy, environment, RUN_STATE and SHA-256 seal. Frame-support caches are transient and not retained.",
        "failure_policy": "Any P1a drift, identity/order mismatch, 32-slot overflow, missing relation class, count/resource failure or worker exception stops and seals FAIL. Do not truncate slots, infer nearest identities, add future traversal state or resume in place.",
    }
    write(CARD, card)
    tools = {
        "runner": RUNNER,
        "real_shard_pilot": REAL_SHARD_PILOT,
        "frame_support": "src/mtare_topo/data/primitive_frame_support.py",
        "relation_targets": "src/mtare_topo/data/primitive_relation_targets.py",
        "relation_storage": "src/mtare_topo/data/primitive_relation_storage.py",
        "sequence_refs": "src/mtare_topo/data/primitive_relation_sequences.py",
        "materialization_loader": "src/mtare_topo/data/primitive_relation_materialization.py",
        "membership_contract": "src/mtare_topo/data/primitive_relation_dataset.py",
        "shape_field": "src/mtare_topo/teacher/swept_superellipse_field.py",
        "governance": "src/mtare_topo/governance.py",
        "frame_support_tests": "tests/v3/unit/test_primitive_frame_support.py",
        "relation_target_tests": "tests/v3/unit/test_primitive_relation_targets.py",
        "storage_tests": "tests/v3/unit/test_primitive_relation_storage.py",
        "sequence_tests": "tests/v3/unit/test_primitive_relation_sequences.py",
        "loader_tests": "tests/v3/unit/test_primitive_relation_materialization.py",
        "preflight": "tools/v3/preflight.py", "create_run": "tools/v3/create_run.py",
    }
    if RUNNER != "tools/v3/run_primitive_relation_p1b_teacher_materialization_v1.py":
        tools["runner_base"] = "tools/v3/run_primitive_relation_p1b_teacher_materialization_v1.py"
    input_paths = [
        p1a_run / "artifacts/evidence_sha256.txt", p1a_run / "artifacts/task_manifest.json",
        p1a_run / "artifacts/traversal_manifest.jsonl", p1a_run / "metrics/summary.json", p1a_run / "RUN_STATE.json",
    ]
    if CORRECTIVE_OF is not None:
        predecessor_summary = json.loads((CORRECTIVE_OF / "metrics/summary.json").read_text())
        predecessor_state = json.loads((CORRECTIVE_OF / "RUN_STATE.json").read_text())
        if (
            predecessor_state.get("state") != "FAILED"
            or predecessor_summary.get("error") != "RuntimeError: real P1b repeated-shard contract failed"
        ):
            raise RuntimeError("P1b V1R requires the exact sealed V1 source_run interface failure")
        input_paths.extend([
            CORRECTIVE_OF / "artifacts/evidence_sha256.txt",
            CORRECTIVE_OF / "metrics/summary.json",
            CORRECTIVE_OF / "RUN_STATE.json",
        ])
        card["corrective_of"] = {
            "run": str(CORRECTIVE_OF.relative_to(PROJECT_ROOT)),
            "failure": predecessor_summary["error"],
            "scope": "Add the omitted source_run field to the real-shard readiness task only; all Teacher, data, split, metric and resource contracts are unchanged.",
        }
        write(CARD, card)
    spec = {
        "schema_version": "v3_run_spec_v1", "gate": 3, "execution_phase": 3,
        "operation": "teacher_generation", "date": "20260830", "slug": SLUG, "seed": 0,
        "question": "Can exact causal ray provenance be converted losslessly into compact 32-slot primitive geometry and relation supervision over the full paired C01-C08 population?",
        "method": "Reduce each P1a frame once to per-primitive support/extrema/azimuth bitsets, then aggregate every exact five-frame window into permutation-safe geometry, temporal correspondence, endpoint attachment and disconnected-overlap targets.",
        "baseline": "P1a ray provenance is the frozen source; historical exit-only and rule graphs do not provide bottom-up primitive-relation labels.",
        "fallback": "Any failure stops P1b. Do not label from future topology, nearest-only hit assignment, fixed junction classes, slot truncation or historical predictions.",
        "source_p1a_run": str(p1a_run.relative_to(PROJECT_ROOT)),
        "corrective_of": None if CORRECTIVE_OF is None else str(CORRECTIVE_OF.relative_to(PROJECT_ROOT)),
        "data_card": str(CARD.relative_to(PROJECT_ROOT)), "config_path": str(CARD.relative_to(PROJECT_ROOT)),
        "user_authorization": approval, "acceptance_criteria": list(card["metrics_and_pre_registered_gates"].values()),
        "expected_counts": {"worlds": 80, "tasks": 240, "frames": 757290, "sequences": 564378, "fit_sequences": 426552, "c07_sequences": 64644, "c08_sequences": 73182, "realized_primitives": 24117, "maximum_slots": 32, "unit_tests": 10, "real_shard_repeat_pilots": 1, "workers": 12, "optimizer_steps": 0, "model_inference_frames": 0, "graph_replays": 0, "c09_worlds_read": 0, "c10_worlds_read": 0},
        "expected_evidence": ["240 compact relation Teacher shards, task/tree hashes, P1a seal verification, tests/environment, RUN_STATE and SHA-256 seal."],
        "estimated_cost": card["estimated_cost"],
        "frozen_tools": {name: {"path": path, "sha256": sha(PROJECT_ROOT / path)} for name, path in tools.items()},
        "frozen_inputs": {str(path.relative_to(PROJECT_ROOT)): sha(path) for path in input_paths},
        "working_directory": str(PROJECT_ROOT),
        "command": [
            "/usr/bin/systemd-inhibit", "--what=sleep:shutdown", "--why=GSE P1b primitive relation Teacher materialization", "--mode=block",
            "/usr/bin/timeout", "--signal=INT", "--kill-after=600s", "259200s",
            "/tmp/mtare_gate4_meshing_sidecar_v1/bin/python", RUNNER,
            "--spec", str(SPEC), "--run-dir", str(PROJECT_ROOT / f"results/gate3_semantics/{RUN_ID}"), "--workers", "12",
        ],
    }
    write(SPEC, spec); print(SPEC)


if __name__ == "__main__":
    main()
