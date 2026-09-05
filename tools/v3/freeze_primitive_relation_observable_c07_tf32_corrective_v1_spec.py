#!/usr/bin/env python3
"""Freeze the one same-checkpoint observable C07 TF32-off corrective."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from _bootstrap import PROJECT_ROOT


CARD = PROJECT_ROOT / "configs/v3/gate3/data_cards/primitive_relation_observable_c07_tf32_corrective_v1.json"
SPEC = PROJECT_ROOT / "configs/v3/gate3/primitive_relation_observable_c07_tf32_corrective_v1.json"
SOURCE_CARD = PROJECT_ROOT / "configs/v3/gate3/data_cards/primitive_relation_observable_three_seed_training_v1.json"
RUN_ID = "gate3_20260902_primitive_relation_observable_c07_tf32_corrective_v1_seed0"
PYTHON = "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python"
SOURCE = PROJECT_ROOT / "results/gate3_semantics/gate3_20260902_primitive_relation_observable_three_seed_training_v1_seed0"
P1A = PROJECT_ROOT / "results/gate3_semantics/gate3_20260830_primitive_relation_p1a_lossless_storage_corrective_v1_seed0"
P1B = PROJECT_ROOT / "results/gate3_semantics/gate3_20260830_primitive_relation_p1b_teacher_materialization_v1r_seed0"
SIDECAR = PROJECT_ROOT / "results/gate3_semantics/gate3_20260902_primitive_attachment_observability_sidecar_v1_seed0"
BASELINE = PROJECT_ROOT / "results/gate3_semantics/gate3_20260902_primitive_relation_observable_nonlearning_c07_v1_seed0"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    if CARD.exists() or SPEC.exists():
        raise RuntimeError("observable C07 TF32 corrective card/spec already exists")
    source_state = json.loads((SOURCE / "RUN_STATE.json").read_text())
    source_summary = json.loads((SOURCE / "metrics/summary.json").read_text())
    if (
        source_state.get("state") != "COMPLETED"
        or source_summary.get("error") is not None
        or int(source_summary.get("optimizer_steps", -1)) != 240_624
        or int(source_summary.get("c08_rows_read", -1)) != 0
    ):
        raise RuntimeError("observable C07 TF32 corrective source drift")

    approval = {
        "status": "APPROVED", "approved_by": "user-standing-authorization",
        "approved_at": "2026-09-02T21:20:00+08:00",
        "authorized_gates": [3], "authorized_operations": ["audit"],
        "scope": "One immutable same-checkpoint C07-only matmul/cuDNN TF32-off corrective: 10 parents, 30 paired geometry tasks, 64644 sequences, three frozen checkpoints, two complete passes per checkpoint and 387864 forward rows; zero optimizer, checkpoint change, C08/C09/C10, graph or M-TARE.",
        "confirmation_reference": "The user authorized uninterrupted best-evidence execution; post-run audit found the final evaluator omitted the declared cuDNN TF32-off contract, making same-checkpoint C07-only correction the unique minimal valid option.",
    }
    gates = {
        "numerical_contract": "Before any CUDA forward, deterministic algorithms are enabled, cuda.matmul.allow_tf32 and cudnn.allow_tf32 are false, and float32 matmul precision is highest; the actual values are written to the result.",
        "population": "Each of the same three selected checkpoints performs the unchanged two-pass evaluator over exactly 64644 C07 rows and 30 tasks; total forward rows are 387864 and selected epochs remain 2/1/2.",
        "science": "Use the unchanged surface>=10%, geometry-macro>=10%, attachment-F1 gain>=5 points, primitive/coverage non-regression and nonempty precision>=0.98 gates; at least two of three seeds must pass all checks.",
        "comparison": "Preserve the old sealed output and publish per-seed old-versus-corrected deltas without using old TF32-on counts as a reproduction target.",
        "isolation": "Optimizer steps, checkpoint writes, C08/C09/C10 rows/worlds, graph replays and M-TARE reads are exactly zero.",
        "resources": "Wall time <=3 h, host RSS/GPU process/PyTorch allocated/reserved each <=16 GiB and result <=0.25 GiB.",
    }
    card = copy.deepcopy(json.loads(SOURCE_CARD.read_text()))
    card.update({
        "card_id": "primitive_relation_observable_c07_tf32_corrective_v1",
        "status": "APPROVED_FOR_ONE_IMMUTABLE_PRIMITIVE_RELATION_OBSERVABLE_C07_TF32_CORRECTIVE_V1",
        "approval": approval,
        "purpose": "Restore the declared deterministic TF32-off numerical contract for the final C07 gate without changing checkpoints, data, Teacher, thresholds, metrics or method.",
        "metrics_and_pre_registered_gates": gates,
        "estimated_cost": {
            "compute": "One RTX 5090 D; three frozen checkpoints and two serial complete C07 passes per checkpoint.",
            "wall_time_hours": 1.0, "host_ram_gb": 16, "gpu": 1,
            "gpu_memory_gb": 16, "disk_gb": 0.25,
        },
        "retention": "Retain corrected per-seed metrics, old-versus-corrected deltas, PNG/PDF/SVG, numerical contract, commands, environment, resource monitor, logs, RUN_STATE and seal.",
        "failure_policy": "Any checkpoint/input/tool, population, numerical setting, resource or isolation drift fails closed. Do not retrain, change thresholds, read C08, select seeds or reinterpret the old noncompliant output as final.",
    })
    card["source"]["raw_sources"] = [
        "sealed P1a C07 causal LiDAR shards",
        "sealed P1b C07 construction primitive/relation Teacher shards",
        "sealed C07 endpoint-observability sidecars",
        "three selected observable checkpoints and the old sealed final evaluation",
        "sealed same-input observable non-learning C07 baseline",
    ]
    card["sampling"].update({
        "independent_sampling_units": "10 disjoint C07 topology parents; three geometry realizations are paired repeated measures within each parent.",
        "raw_frame_count": 88_140, "effective_sample_count": 21_548,
        "effective_structure_event_count": 64_644,
        "spatial_interval_m": 1.0, "temporal_window_frames": 5,
        "geometry_realizations": 3,
        "rule": "Run the unchanged two-pass final evaluator under explicit deterministic TF32-off settings for seed0/1/2 selected checkpoints. No sampling, data deletion, new threshold family, checkpoint selection or raw prediction persistence.",
    })
    card["sampling"]["structure_event_counts"] = {
        "c07_parent_worlds": 10, "c07_geometry_tasks": 30,
        "c07_unique_source_sequences": 21_548,
        "c07_paired_sequences_per_seed_pass": 64_644,
        "passes_per_seed": 2, "frozen_seeds": 3,
        "model_forward_rows": 387_864,
        "observable_positive_attachments": 442_936,
    }
    card["teacher"] = {
        "source": "The same sealed P1b construction Teacher and 0.25 m endpoint-support sidecar used by the original final C07 evaluator.",
        "labels": "Swept geometry, temporal visibility, physical attachment, disconnected overlap and endpoint observability.",
        "valid_mask": "Exactly the original dual-endpoint-observed policy; no label or mask changes are allowed.",
        "student_forbidden_inputs": "Teacher support, construction identity, world/TNG identity, absolute pose and future frames never enter model forward.",
        "planner_consistency_plan": "No graph or planner runs. Corrected C07 only decides whether the frozen method may proceed to a separately sealed C08 transfer or failure attribution.",
    }
    card["split"].update({
        "fit": "No fit, optimizer, normalization or checkpoint update.",
        "selection": "The same complete C07 population and registered threshold grid are recomputed only to correct the numerical execution contract.",
        "development_transfer": "C08 remains unread until a corrected 2-of-3 all-gates PASS.",
        "strict_test": "C08/C09/C10 and all M-TARE benchmark worlds remain unread.",
        "historical_pollution_audit": "The old sealed C07 output is read only for reporting numerical deltas; it cannot select a checkpoint, threshold family or method change.",
    })
    card["leakage_audit"].update({
        "optimizer_step_count": 0, "model_inference_count": 387_864,
        "test_excluded_from_checkpoint_selection": True,
    })
    card["worlds"]["normalization"] = []
    card["worlds"]["checkpoint_selection"] = []
    card["worlds"]["augmentation_tuning"] = []
    card["worlds"]["ssl"] = []
    card["worlds"]["train"] = ["NONE_FROZEN_CHECKPOINT_AUDIT_ONLY"]
    card["worlds"]["validation"] = [f"S{index:02d}_C07" for index in range(1, 11)]
    card["worlds"]["strict_test"] = [
        *[f"S{index:02d}_C08" for index in range(1, 11)],
        *[f"S{index:02d}_C09" for index in range(1, 11)],
        *[f"S{index:02d}_C10" for index in range(1, 11)],
        "all_M-TARE_benchmark_worlds",
    ]
    card["trajectories"] = [{
        "id": "all_C07_directed_traversals_three_geometries_same_checkpoint_corrective",
        "world": "S01-S10_C07", "split": "validation", "independent": True,
        "distance_m": 27_421.160671011814,
        "duration_s": 27_421.160671011814,
        "spatial_coverage_m": 27_421.160671011814,
    }]
    write(CARD, card)

    inputs = [
        CARD,
        SOURCE / "RUN_STATE.json", SOURCE / "metrics/summary.json",
        SOURCE / "metrics/evaluation/summary.json", SOURCE / "artifacts/evidence_sha256.txt",
        *[SOURCE / f"artifacts/models/seed{seed}/selected.pt" for seed in (0, 1, 2)],
        *[
            root / relative
            for root in (P1A, P1B, SIDECAR, BASELINE)
            for relative in ("RUN_STATE.json", "metrics/summary.json", "artifacts/evidence_sha256.txt")
        ],
        P1A / "artifacts/task_manifest.json", P1B / "artifacts/task_manifest.json",
        SIDECAR / "artifacts/task_manifest.json",
        PROJECT_ROOT / "configs/v3/gate2/environments/phase3_torch290_cu129_zarr2187_v1r2.json",
        PROJECT_ROOT / "configs/v3/gate2/environments/phase3_torch290_cu129_zarr2187_pip_freeze.txt",
    ]
    tools = {
        "runner": "tools/v3/run_primitive_relation_observable_c07_tf32_corrective_v1.py",
        "freezer": "tools/v3/freeze_primitive_relation_observable_c07_tf32_corrective_v1_spec.py",
        "corrective_evaluator": "tools/v3/evaluate_primitive_relation_observable_three_seed_c07_tf32_corrective_v1.py",
        "base_evaluator": "tools/v3/evaluate_primitive_relation_observable_three_seed_v1.py",
        "reader": "src/mtare_topo/data/primitive_relation_observable_batches.py",
        "sidecar": "src/mtare_topo/data/primitive_attachment_observability_sidecar.py",
        "model": "src/mtare_topo/representation/primitive_relation_observable_model.py",
        "training_converter": "src/mtare_topo/representation/primitive_relation_observable_training.py",
        "metrics": "src/mtare_topo/evaluation/primitive_relation_metrics.py",
        "base_comparison": "tools/v3/evaluate_primitive_relation_three_seed_v1.py",
        "test_corrective": "tests/v3/unit/test_evaluate_primitive_relation_observable_three_seed_c07_tf32_corrective_v1.py",
        "test_base": "tests/v3/unit/test_evaluate_primitive_relation_observable_three_seed_v1.py",
        "governance": "src/mtare_topo/governance.py",
        "preflight": "tools/v3/preflight.py", "create_run": "tools/v3/create_run.py",
    }
    run_dir = PROJECT_ROOT / f"results/gate3_semantics/{RUN_ID}"
    spec = {
        "schema_version": "v3_run_spec_v1", "gate": 3,
        "execution_phase": 3, "operation": "audit", "date": "20260902",
        "slug": "primitive_relation_observable_c07_tf32_corrective_v1", "seed": 0,
        "question": "Under the declared deterministic matmul/cuDNN TF32-off contract, do the same three frozen observable-relation checkpoints pass every C07 gate in at least two seeds?",
        "method": "Run the unchanged two-pass complete C07 evaluator on selected epochs 2/1/2 after explicitly setting deterministic algorithms, both TF32 flags false and float32 precision highest; preserve old output and report exact deltas.",
        "baseline": "The same sealed observable non-learning C07 baseline and the old sealed noncompliant final output for numerical deltas only.",
        "fallback": "Any system drift fails closed. Corrected 2-of-3 PASS only authorizes a separate C08 run; corrected FAIL only authorizes sealed C07 failure attribution. No retraining or threshold change.",
        "corrective_of": str(SOURCE.relative_to(PROJECT_ROOT)),
        "data_card": str(CARD.relative_to(PROJECT_ROOT)),
        "config_path": str(CARD.relative_to(PROJECT_ROOT)),
        "user_authorization": approval,
        "acceptance_criteria": list(gates.values()),
        "expected_counts": {
            "c07_parent_worlds": 10, "c07_tasks": 30,
            "c07_rows_per_seed": 64_644, "passes_per_seed": 2,
            "frozen_seeds": 3, "model_forward_rows": 387_864,
            "selected_epochs": [2, 1, 2], "unit_tests": 3,
            "optimizer_steps": 0, "checkpoint_writes": 0,
            "c08_rows_read": 0, "c09_c10_worlds_read": 0,
            "graph_replays": 0, "mtare_worlds_read": 0,
        },
        "estimated_cost": card["estimated_cost"],
        "expected_evidence": [
            "Actual deterministic/TF32 state before any forward.",
            "Corrected per-seed C07 geometry, relation, evidence, safe metrics and 2-of-3 decision.",
            "Old-versus-corrected per-seed deltas and unchanged selected epochs.",
            "Resource monitor, three-format figure, logs, source integrity, RUN_STATE and seal.",
        ],
        "frozen_inputs": {str(path.relative_to(PROJECT_ROOT)): sha(path) for path in inputs},
        "frozen_tools": {
            name: {"path": path, "sha256": sha(PROJECT_ROOT / path)}
            for name, path in tools.items()
        },
        "working_directory": str(PROJECT_ROOT),
        "command": [
            "/usr/bin/systemd-inhibit", "--what=sleep:shutdown",
            "--why=Observable relation C07 TF32 contract corrective",
            "--mode=block", "/usr/bin/timeout", "--signal=INT",
            "--kill-after=60s", "10800s", "/usr/bin/env",
            "CUBLAS_WORKSPACE_CONFIG=:4096:8",
            f"PYTHONPATH={PROJECT_ROOT / 'src'}:{PROJECT_ROOT / 'tools/v3'}:{PROJECT_ROOT}",
            PYTHON, "tools/v3/run_primitive_relation_observable_c07_tf32_corrective_v1.py",
            "--spec", str(SPEC), "--run-dir", str(run_dir),
        ],
    }
    write(SPEC, spec)
    print(SPEC)


if __name__ == "__main__":
    main()
