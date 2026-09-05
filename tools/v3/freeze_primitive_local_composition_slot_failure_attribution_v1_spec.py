#!/usr/bin/env python3
"""Freeze one C07-only local composition-slot factor attribution."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE_CARD = ROOT / "configs/v3/gate3/data_cards/primitive_local_composition_slot_three_seed_training_v1.json"
CARD = ROOT / "configs/v3/gate3/data_cards/primitive_local_composition_slot_failure_attribution_v1.json"
SPEC = ROOT / "configs/v3/gate3/primitive_local_composition_slot_failure_attribution_v1.json"
RUN_ID = "gate3_20260904_primitive_local_composition_slot_failure_attribution_v1_seed0"
CARD_STATUS = "APPROVED_FOR_ONE_IMMUTABLE_PRIMITIVE_LOCAL_COMPOSITION_SLOT_FAILURE_ATTRIBUTION_V1"
PYTHON = "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python"
SOURCE_RUN = ROOT / "results/gate3_semantics/gate3_20260903_primitive_local_composition_slot_three_seed_training_v1_seed0"
P1A = ROOT / "results/gate3_semantics/gate3_20260830_primitive_relation_p1a_lossless_storage_corrective_v1_seed0"
P1B = ROOT / "results/gate3_semantics/gate3_20260830_primitive_relation_p1b_teacher_materialization_v1r_seed0"
OBS = ROOT / "results/gate3_semantics/gate3_20260902_primitive_attachment_observability_sidecar_v1_seed0"
THRESHOLDS = ROOT / "results/gate3_semantics/gate3_20260903_primitive_predicted_geometry_association_diagnostic_v1_seed0"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    if CARD.exists() or SPEC.exists():
        raise RuntimeError("local-slot attribution card/spec already exists")
    source_state = _load(SOURCE_RUN / "RUN_STATE.json")
    source_summary = _load(SOURCE_RUN / "metrics/summary.json")
    evaluation = _load(SOURCE_RUN / "metrics/evaluation/summary.json")
    if not (
        source_state.get("state") == "COMPLETED" and source_state.get("error") is None
        and source_summary.get("error") is None and source_summary.get("scientific_pass") is False
        and evaluation.get("c07", {}).get("passing_seeds") == 1
        and evaluation.get("c08_rows_read") == 0
    ):
        raise RuntimeError("attribution requires the exact valid 1/3 C07 source failure")
    card = copy.deepcopy(_load(SOURCE_CARD))
    card.update({
        "card_id": "primitive_local_composition_slot_failure_attribution_v1",
        "status": CARD_STATUS,
        "purpose": "Use the three frozen local-slot checkpoints to decompose the failed C07 structured score into slot identity, dustbin/second-slot margin, slot presence, primitive existence, endpoint evidence and entropy factors without training or changing the deployment gate.",
        "teacher_source": "The same sealed P1b construction attachments and dual-endpoint observability mask used by the failed source gate; labels are read only for C07 scoring and per-factor attribution.",
        "estimated_cost": {
            "compute": "Three deterministic frozen-checkpoint C07 inference passes plus exact tie-safe ranking of 18 pre-registered factor scores.",
            "disk_gb": 1, "gpu": 1, "gpu_memory_gb": 16,
            "host_ram_gb": 4, "wall_time_hours": 4,
        },
        "failure_policy": "Any source metric reproduction, count, hash, environment, symmetry, resource or isolation failure stops and seals system FAIL. Diagnostic factor scores cannot be promoted to a new method, threshold or graph result inside this run.",
        "retention": "Keep per-seed and per-task factor metrics, endpoint zero-rate/assignment statistics, comparison PNG/PDF/SVG, environment, commands, logs, RUN_STATE and SHA-256 seal; retain no raw pair arrays.",
    })
    card["sampling"] = {
        "raw_frame_count": 64_644 * 5,
        "effective_sample_count": 64_644,
        "effective_structure_event_count": 442_936,
        "independent_sampling_units": "10 disjoint C07 topology parents and 30 paired geometry tasks; the three geometry variants are repeated measures within each parent.",
        "rule": "Read every one of the 64644 C07 five-frame sequences exactly once per frozen seed. All 442936 observable positive attachment pairs remain in the recall denominator; no row, pair, task or seed is selected or removed.",
        "spatial_interval_m": 1.0,
        "temporal_window_frames": 5,
        "structure_event_counts": {
            "fit_parent_worlds_read": 0, "c07_parent_worlds": 10,
            "c07_tasks": 30, "c07_sequences_per_seed": 64_644,
            "c07_inference_sequences": 193_932,
            "c07_observable_positive_attachments_per_seed": 442_936,
            "seeds": 3, "optimizer_steps": 0, "c08_rows_read": 0,
            "graph_replays": 0, "mtare_worlds_read": 0,
        },
    }
    card["split"] = {
        "fit": "C01-C06 are not read; training is finished and all checkpoints are immutable.",
        "selection": "C07 is used only to attribute the already declared source failure. It cannot authorize a factor formula as a replacement deployment method.",
        "development_transfer": "C08 remains unopened.",
        "strict_test": "C08/C09/C10 and all M-TARE benchmark worlds remain unread.",
        "world_disjoint": True, "trajectory_disjoint": True,
        "historical_pollution_audit": "The factor list and diagnostic decision tree are frozen before execution; no result-dependent factor is added.",
    }
    card["teacher"] = {
        "source": "Sealed P1b construction attachment cliques plus sealed endpoint observability mask.",
        "labels": "Per-observed-endpoint local composition clique/dustbin and disconnected-overlap hard-negative relation.",
        "valid_mask": "Only dual-observed cross-primitive upper-triangle endpoint pairs are candidates; all observable positives remain in the recall denominator even when primitive existence rejects them.",
        "student_forbidden_inputs": "Teacher/world/node/TNG/identity/absolute pose/future frames never enter model forward. No C08+ or planner state is read.",
        "planner_consistency_plan": "No graph or planner is executed in this attribution.",
    }
    card["leakage_audit"] = {
        "model_inference_count": 193_932, "optimizer_step_count": 0,
        "future_sensor_frames_excluded": True, "absolute_pose_not_retained_in_student_representation": True,
        "test_excluded_from_supervised_training": True, "test_excluded_from_ssl": True,
        "test_excluded_from_normalization": True, "test_excluded_from_teacher_calibration": True,
        "test_excluded_from_threshold_calibration": True, "test_excluded_from_checkpoint_selection": True,
        "test_excluded_from_augmentation_tuning": True, "mtare_benchmark_excluded": True,
    }
    card["metrics_and_pre_registered_gates"] = {
        "reproduction": "The deployed structured score must exactly reproduce source best/safe threshold, TP, FP, FN and selected-pair counts for all three seeds.",
        "factors": "Report 18 frozen factor scores: hard slot identity, each endpoint confidence component/product, soft slot affinity/presence/safe scores, and the exact deployed score; all use tie-safe thresholds.",
        "classification": "If hard slot identity is safely nonempty in at least 2/3 seeds, diagnose multiplicative confidence collapse; else if soft affinity is safely nonempty in at least 2/3, diagnose hard-argmax fragmentation; otherwise diagnose slot identity and confidence as not cross-seed safe.",
        "isolation": "Exactly 193932 C07 inference rows, zero optimizer steps, zero C08/C09/C10, graph and M-TARE reads.",
        "resources": "Host RSS <=4 GiB, result <=1 GiB and wall time <=4 h; GPU process memory remains below16 GiB.",
    }
    approval = {
        "approved_at": "2026-09-04T09:05:00+08:00", "approved_by": "user-standing-authorization",
        "authorized_gates": [3], "authorized_operations": ["audit"],
        "confirmation_reference": "The user said to continue and previously authorized automatic evidence-optimal execution within the frozen paper plan.",
        "scope": "One immutable C07-only zero-training local-slot factor attribution; no C08, graph, retraining or threshold change.",
        "status": "APPROVED",
    }
    card["approval"] = approval
    _write(CARD, card)

    run_dir = ROOT / "results/gate3_semantics" / RUN_ID
    diagnostic_command = [
        PYTHON, str(ROOT / "tools/v3/execute_primitive_local_composition_slot_failure_attribution_v1.py"),
        "--models-root", str(SOURCE_RUN / "artifacts/models"),
        "--sensor-root", str(P1A / "artifacts/dataset"),
        "--teacher-root", str(P1B / "artifacts/teacher"),
        "--observability-root", str(OBS / "artifacts/endpoint_observability"),
        "--source-evaluation-summary", str(SOURCE_RUN / "metrics/evaluation/summary.json"),
        "--existence-threshold-source", str(THRESHOLDS / "metrics/diagnostic/summary.json"),
        "--output-dir", str(run_dir / "metrics/attribution"),
    ]
    spec = {
        "schema_version": "v3_run_spec_v1", "slug": "primitive_local_composition_slot_failure_attribution_v1",
        "date": "20260904", "seed": 0, "gate": 3, "execution_phase": 3,
        "operation": "audit", "working_directory": str(ROOT),
        "config_path": str(CARD.relative_to(ROOT)), "data_card": str(CARD.relative_to(ROOT)),
        "question": "Is the local-slot C07 failure caused primarily by incorrect slot identity or by multiplicative endpoint-confidence collapse?",
        "method": "Freeze all three checkpoints and decompose the exact structured score into its existing assignment, dustbin margin, slot presence, existence, evidence and entropy factors, plus hard versus soft slot relation diagnostics.",
        "baseline": "The exact sealed deployed structured score and the frozen non-learning C07 attachment baseline; no diagnostic score is treated as a deployable method.",
        "fallback": card["failure_policy"], "user_authorization": approval,
        "acceptance_criteria": list(card["metrics_and_pre_registered_gates"].values()),
        "expected_counts": {
            "c07_parent_worlds": 10, "c07_tasks": 30, "c07_rows_per_seed": 64_644,
            "c07_model_forward_rows": 193_932, "c07_positive_pairs_per_seed": 442_936,
            "seeds": 3, "factor_scores": 18, "tests": 16,
            "optimizer_steps": 0, "c08_rows": 0, "graph_replays": 0, "mtare_worlds": 0,
        },
        "estimated_cost": card["estimated_cost"],
        "expected_evidence": [
            "Exact three-seed reproduction of the source deployed C07 metrics.",
            "Per-seed and per-task factor zero rates, assignment statistics, best-F1, precision>=0.98 TP and overlap FP.",
            "Pre-registered mechanism classification, comparison PNG/PDF/SVG, environment, logs, RUN_STATE and SHA-256 seal.",
        ],
        "diagnostic_command": diagnostic_command,
        "command": [
            "/usr/bin/systemd-inhibit", "--what=sleep:shutdown", "--why=Local slot C07 failure attribution",
            "--mode=block", "/usr/bin/timeout", "--signal=INT", "--kill-after=120s", "18000s",
            "/usr/bin/env", f"PYTHONPATH={ROOT / 'src'}:{ROOT / 'tools/v3'}:{ROOT}",
            "PYTHONHASHSEED=0", "CUBLAS_WORKSPACE_CONFIG=:4096:8", PYTHON,
            "tools/v3/run_primitive_local_composition_slot_failure_attribution_v1.py",
            "--spec", str(SPEC), "--run-dir", str(run_dir),
        ],
    }
    inputs = [
        CARD, SOURCE_CARD, SOURCE_RUN / "RUN_STATE.json", SOURCE_RUN / "metrics/summary.json",
        SOURCE_RUN / "metrics/evaluation/summary.json", SOURCE_RUN / "artifacts/evidence_sha256.txt",
        *[SOURCE_RUN / f"artifacts/models/seed{seed}/selected.pt" for seed in range(3)],
        P1A / "RUN_STATE.json", P1A / "metrics/summary.json", P1A / "artifacts/task_manifest.json", P1A / "artifacts/evidence_sha256.txt",
        P1B / "RUN_STATE.json", P1B / "metrics/summary.json", P1B / "artifacts/task_manifest.json", P1B / "artifacts/evidence_sha256.txt",
        OBS / "RUN_STATE.json", OBS / "metrics/summary.json", OBS / "artifacts/task_manifest.json", OBS / "artifacts/evidence_sha256.txt",
        THRESHOLDS / "RUN_STATE.json", THRESHOLDS / "metrics/diagnostic/summary.json", THRESHOLDS / "artifacts/evidence_sha256.txt",
    ]
    spec["frozen_inputs"] = {str(path.relative_to(ROOT)): _sha(path) for path in inputs}
    tools = {
        "factorization": "src/mtare_topo/evaluation/primitive_local_composition_slot_failure_attribution.py",
        "model": "src/mtare_topo/representation/primitive_local_composition_slot_model.py",
        "decoding": "src/mtare_topo/representation/primitive_local_composition_slot_decoding.py",
        "tie_safe_metrics": "src/mtare_topo/evaluation/primitive_relation_sparse_port_failure_attribution.py",
        "executor": "tools/v3/execute_primitive_local_composition_slot_failure_attribution_v1.py",
        "runner": "tools/v3/run_primitive_local_composition_slot_failure_attribution_v1.py",
        "freezer": "tools/v3/freeze_primitive_local_composition_slot_failure_attribution_v1_spec.py",
        "factor_tests": "tests/v3/unit/test_primitive_local_composition_slot_failure_attribution.py",
        "decode_tests": "tests/v3/unit/test_primitive_local_composition_slot_decoding.py",
        "metric_tests": "tests/v3/unit/test_primitive_relation_sparse_port_failure_attribution.py",
        "governance": "src/mtare_topo/governance.py", "preflight": "tools/v3/preflight.py",
        "create_run": "tools/v3/create_run.py",
    }
    spec["frozen_tools"] = {name: {"path": path, "sha256": _sha(ROOT / path)} for name, path in tools.items()}
    _write(SPEC, spec); print(CARD); print(SPEC)


if __name__ == "__main__":
    main()
