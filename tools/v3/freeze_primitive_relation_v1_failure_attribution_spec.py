#!/usr/bin/env python3
"""Freeze the sole full-C07 primitive-relation V1 failure attribution."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from _bootstrap import PROJECT_ROOT


CARD = PROJECT_ROOT / "configs/v3/gate3/data_cards/primitive_relation_v1_failure_attribution_v1.json"
SPEC = PROJECT_ROOT / "configs/v3/gate3/primitive_relation_v1_failure_attribution_v1.json"
RUN_ID = "gate3_20260831_primitive_relation_v1_failure_attribution_v1_seed0"
PYTHON = "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python"
P1A = PROJECT_ROOT / "results/gate3_semantics/gate3_20260830_primitive_relation_p1a_lossless_storage_corrective_v1_seed0"
P1B = PROJECT_ROOT / "results/gate3_semantics/gate3_20260830_primitive_relation_p1b_teacher_materialization_v1r_seed0"
TRAINING = PROJECT_ROOT / "results/gate3_semantics/gate3_20260830_primitive_relation_three_seed_training_v1_seed0"
BASELINE = PROJECT_ROOT / "results/gate3_semantics/gate3_20260830_primitive_relation_nonlearning_c07_v1_seed0"


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
    if CARD.exists() or SPEC.exists():
        raise RuntimeError("primitive relation attribution card/spec already exists")
    training_state = json.loads((TRAINING / "RUN_STATE.json").read_text())
    training_summary = json.loads((TRAINING / "metrics/summary.json").read_text())
    if training_state.get("state") != "COMPLETED" or training_state.get("error") is not None:
        raise RuntimeError("primitive relation attribution source run is not cleanly completed")
    if training_summary.get("scientific_pass") is not False or training_summary.get("c08_rows_read") != 0:
        raise RuntimeError("primitive relation attribution source is not the sealed C07 failure")
    approval = {
        "status": "APPROVED", "approved_by": "user-standing-authorization",
        "approved_at": "2026-08-31T21:20:00+08:00", "authorized_gates": [3],
        "authorized_operations": ["audit"],
        "scope": "One immutable full-population C07-only frozen-checkpoint inference attribution: 10 parents, 30 paired geometry tasks, 64644 sequences and seeds 0/1/2; zero optimizer, threshold change, C08/C09/C10, graph or M-TARE.",
        "confirmation_reference": "User repeatedly authorized continuous autonomous best-evidence execution without routine approval prompts; this operation is the exact in-plan scientific-failure attribution announced after V1 stopped.",
    }
    families = ((1, "flat_tree_small"), (2, "3d_tree_small"), (3, "flat_unicyclic_small"), (4, "3d_unicyclic_small"), (5, "flat_branch_medium"), (6, "3d_branch_medium"), (7, "flat_loop_rich"), (8, "3d_loop_rich"), (9, "flat_complex"), (10, "3d_complex"))
    c07 = [f"S{family:02d}_{name}_C07" for family, name in families]
    strict = [f"S{family:02d}_{name}_C{condition:02d}" for condition in (8, 9, 10) for family, name in families]
    card = {
        "schema_version": "v3_data_card_v1",
        "card_id": "primitive_relation_v1_failure_attribution_v1",
        "status": "APPROVED_FOR_ONE_IMMUTABLE_PRIMITIVE_RELATION_V1_FAILURE_ATTRIBUTION_V1",
        "approval": approval,
        "purpose": "Determine whether V1 attachment failure is caused primarily by over-active primitive proposals or by relation logits that remain inadequate even when evaluation is restricted to Teacher-matched primitive slots.",
        "source": {
            "raw_sources": [
                "sealed P1a corrected C07 five-frame 16x720 range/valid and relative odometry",
                "sealed P1b C07 32-slot visible primitive and relation Teacher",
                "three selected checkpoints and formal C07 metrics from the sealed failed V1 run",
            ],
            "license_or_allowed_use": "Local research use of project-generated Cano assets; redistribution remains subject to upstream licensing.",
        },
        "worlds": {
            "train": [], "validation": c07, "ssl": [], "normalization": [],
            "teacher_calibration": [], "threshold_calibration": [], "augmentation_tuning": [],
            "checkpoint_selection": [], "strict_test": strict + ["all_M-TARE_benchmark_worlds"],
        },
        "trajectories": [{
            "id": "all_C07_directed_traversals_three_geometries_frozen_replay",
            "world": c07[0], "split": "validation", "independent": True,
            "duration_s": 27421.160671011814, "distance_m": 27421.160671011814,
            "spatial_coverage_m": 27421.160671011814,
        }],
        "sampling": {
            "independent_sampling_units": "10 C07 topology parents; three geometry realizations are paired repeated measures and remain within parent.",
            "raw_frame_count": 88140, "effective_sample_count": 21548,
            "effective_structure_event_count": 64644, "spatial_interval_m": 1.0,
            "structure_event_counts": {
                "c07_parent_worlds": 10, "c07_geometry_tasks": 30,
                "c07_unique_source_sequences": 21548, "c07_paired_sequences": 64644,
                "frozen_seeds": 3, "model_inference_rows": 193932,
            },
            "rule": "Read every C07 row once per frozen seed in task and row order. Accumulate actual-candidate and Teacher-matched proposal-oracle metrics in one forward pass; no sampling, deletion, persistence of raw pair tensors or C08 access.",
        },
        "split": {
            "world_disjoint": True, "trajectory_disjoint": True,
            "fit": "No fit, optimizer, normalization or model update.",
            "selection": "No new checkpoint or threshold selection. The original formal C07 existence threshold is reproduced; relation threshold sweeps are diagnostic only.",
            "development_transfer": "C08 is forbidden because V1 failed C07; rows read must remain zero.",
            "strict_test": "C08/C09/C10 and all M-TARE worlds remain unread.",
            "historical_pollution_audit": "Only the already-exposed C07 selection population diagnoses the already-failed V1. No strict-test or transfer claim is made.",
        },
        "teacher": {
            "source": "Sealed P1b construction-program Teacher cropped to actual five-frame ray support.",
            "valid_mask": "Teacher primitive mask defines an evaluation-only proposal oracle after unchanged Hungarian alignment; relation logits are never replaced by Teacher values.",
            "planner_consistency_plan": "No graph, planner or deployment path. Oracle results only select which model component requires a new readiness proof.",
            "student_forbidden_inputs": "The model forward still receives only five causal range/valid scans and relative odometry. Teacher mask is applied after inference only for attribution.",
        },
        "leakage_audit": {
            "optimizer_step_count": 0, "model_inference_count": 193932,
            "future_sensor_frames_excluded": True,
            "absolute_pose_not_retained_in_student_representation": True,
            "mtare_benchmark_excluded": True,
            "test_excluded_from_supervised_training": True,
            "test_excluded_from_ssl": True,
            "test_excluded_from_normalization": True,
            "test_excluded_from_teacher_calibration": True,
            "test_excluded_from_threshold_calibration": True,
            "test_excluded_from_augmentation_tuning": True,
            "test_excluded_from_checkpoint_selection": True,
        },
        "metrics_and_pre_registered_gates": {
            "implementation": "Ten frozen CPU tests pass; formal C07 existence/attachment/overlap counts reproduce exactly for all three seeds.",
            "population": "Exactly 64644 rows and 30 tasks per seed, 193932 inference rows total, three selected epochs unchanged, C08 rows read zero.",
            "proposal": "Report active/target/redundant slots, active-count histograms and actual/oracle attachment and overlap pair spaces per seed and task.",
            "relation": "Compare unchanged relation logits under actual existence masks and Teacher-matched proposal-oracle masks using the original 91 thresholds and precision>=0.98 safe selection.",
            "decision": "If at least two seeds reach baseline attachment F1+0.05 and nonzero safe true positives under the proposal oracle, allow only a sparse-existence/set-decoder corrective readiness. Otherwise require a new sparse-port relation architecture readiness; never treat oracle as deployable performance.",
            "resources": "Wall time <=3h, CUDA reserved <=16GiB, host RAM <=16GiB and output <=0.2GiB; zero optimizer/C08/C09/C10/graph/M-TARE.",
        },
        "estimated_cost": {
            "compute": "One RTX 5090, three frozen selected checkpoints, one streaming pass over full C07 each.",
            "wall_time_hours": 1.0, "host_ram_gb": 16, "gpu": 1,
            "gpu_memory_gb": 16, "disk_gb": 0.2,
        },
        "retention": "Retain per-seed and per-task attribution, actual/oracle pair statistics, PNG/PDF/SVG/source, environment, raw logs, RUN_STATE and seal as paper failure analysis.",
        "failure_policy": "Any population, formal-metric reproduction, checkpoint, source/tool/environment, resource or isolation drift fails closed. Do not modify thresholds, checkpoint, model, data, Teacher or use C08 to resolve the diagnosis.",
    }
    write(CARD, card)
    tools = {
        "executor": "tools/v3/execute_primitive_relation_v1_failure_attribution.py",
        "runner": "tools/v3/run_primitive_relation_v1_failure_attribution.py",
        "freezer": "tools/v3/freeze_primitive_relation_v1_failure_attribution_spec.py",
        "attribution_module": "src/mtare_topo/evaluation/primitive_relation_failure_attribution.py",
        "metrics": "src/mtare_topo/evaluation/primitive_relation_metrics.py",
        "batch_reader": "src/mtare_topo/data/primitive_relation_batches.py",
        "training_conversion": "src/mtare_topo/representation/primitive_relation_training.py",
        "model": "src/mtare_topo/representation/primitive_relation_model.py",
        "loss_alignment": "src/mtare_topo/representation/primitive_relation_losses.py",
        "attribution_tests": "tests/v3/unit/test_primitive_relation_failure_attribution.py",
        "metric_tests": "tests/v3/unit/test_primitive_relation_metrics.py",
        "evaluation_tests": "tests/v3/unit/test_evaluate_primitive_relation_three_seed_v1.py",
        "governance": "src/mtare_topo/governance.py",
        "preflight": "tools/v3/preflight.py", "create_run": "tools/v3/create_run.py",
    }
    inputs = [
        CARD,
        P1A / "RUN_STATE.json", P1A / "metrics/summary.json", P1A / "artifacts/evidence_sha256.txt", P1A / "artifacts/task_manifest.json",
        P1B / "RUN_STATE.json", P1B / "metrics/summary.json", P1B / "artifacts/evidence_sha256.txt", P1B / "artifacts/task_manifest.json",
        TRAINING / "RUN_STATE.json", TRAINING / "metrics/summary.json", TRAINING / "artifacts/evidence_sha256.txt", TRAINING / "config/source_integrity_after.json",
        BASELINE / "RUN_STATE.json", BASELINE / "metrics/summary.json", BASELINE / "artifacts/evidence_sha256.txt",
        PROJECT_ROOT / "configs/v3/gate2/environments/phase3_torch290_cu129_zarr2187_v1r2.json",
        PROJECT_ROOT / "configs/v3/gate2/environments/phase3_torch290_cu129_zarr2187_pip_freeze.txt",
    ]
    for seed in range(3):
        inputs.extend((
            TRAINING / f"artifacts/models/seed{seed}/selected.pt",
            TRAINING / f"metrics/evaluation/c07_seed{seed}.json",
        ))
    spec = {
        "schema_version": "v3_run_spec_v1", "gate": 3, "execution_phase": 3,
        "operation": "audit", "date": "20260831",
        "slug": "primitive_relation_v1_failure_attribution_v1", "seed": 0,
        "question": "Does V1 fail attachment learning mainly because primitive proposals are over-active, or do unchanged relation logits fail even when evaluation is restricted to Teacher-matched proposals?",
        "method": "Replay each frozen selected checkpoint once on the complete C07 population; reproduce formal metrics, compare actual existence masks with an evaluation-only Teacher-matched proposal oracle, and stratify slot/pair expansion and relation precision-recall by seed, topology and geometry family.",
        "baseline": "The sealed same-input non-learning attachment F1=0.005819 and the exact actual-candidate C07 metrics from failed V1.",
        "fallback": "Any drift or unresolved diagnosis stops. No C08, threshold relaxation, checkpoint selection, training, graph rule or planner compensation.",
        "data_card": str(CARD.relative_to(PROJECT_ROOT)), "config_path": str(CARD.relative_to(PROJECT_ROOT)),
        "user_authorization": approval,
        "acceptance_criteria": list(card["metrics_and_pre_registered_gates"].values()),
        "expected_counts": {
            "c07_parent_worlds": 10, "c07_geometry_tasks": 30,
            "c07_rows_per_seed": 64644, "frozen_seeds": 3,
            "model_inference_rows": 193932, "optimizer_steps": 0,
            "unit_tests": 10, "c08_rows_read": 0, "c09_c10_worlds_read": 0,
            "graph_replays": 0, "mtare_worlds_read": 0,
        },
        "expected_evidence": [
            "Exact formal metric reproduction and C07 population proof.",
            "Per-seed actual/oracle slot, pair, attachment, overlap and safe-recall metrics.",
            "Per-task topology/geometry stratification and one resolved diagnosis.",
            "PNG/PDF/SVG/source, environment, commands, raw logs, RUN_STATE and SHA-256 seal.",
        ],
        "estimated_cost": card["estimated_cost"],
        "frozen_tools": {name: {"path": path, "sha256": sha(PROJECT_ROOT / path)} for name, path in tools.items()},
        "frozen_inputs": {str(path.relative_to(PROJECT_ROOT)): sha(path) for path in inputs},
        "working_directory": str(PROJECT_ROOT),
        "command": [
            "/usr/bin/systemd-inhibit", "--what=sleep:shutdown",
            "--why=Primitive relation V1 full C07 failure attribution", "--mode=block",
            "/usr/bin/timeout", "--signal=INT", "--kill-after=60s", "10800s",
            "/usr/bin/env", "CUBLAS_WORKSPACE_CONFIG=:4096:8",
            f"PYTHONPATH={PROJECT_ROOT / 'src'}:{PROJECT_ROOT / 'tools/v3'}",
            PYTHON, "tools/v3/run_primitive_relation_v1_failure_attribution.py",
            "--spec", str(SPEC),
            "--run-dir", str(PROJECT_ROOT / f"results/gate3_semantics/{RUN_ID}"),
        ],
    }
    write(SPEC, spec); print(SPEC)


if __name__ == "__main__":
    main()
