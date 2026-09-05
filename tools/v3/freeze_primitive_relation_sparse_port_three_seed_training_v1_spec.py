#!/usr/bin/env python3
"""Freeze the sole sparse-port primitive-relation three-seed run."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from _bootstrap import PROJECT_ROOT


CARD = PROJECT_ROOT / "configs/v3/gate3/data_cards/primitive_relation_sparse_port_three_seed_training_v1.json"
SPEC = PROJECT_ROOT / "configs/v3/gate3/primitive_relation_sparse_port_three_seed_training_v1.json"
OLD_CARD = PROJECT_ROOT / "configs/v3/gate3/data_cards/primitive_relation_three_seed_training_v1.json"
RUN_ID = "gate3_20260831_primitive_relation_sparse_port_three_seed_training_v1_seed0"
PYTHON = (
    "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/"
    "phase3_torch290_cu129_zarr2187_v1/bin/python"
)
P1A = PROJECT_ROOT / "results/gate3_semantics/gate3_20260830_primitive_relation_p1a_lossless_storage_corrective_v1_seed0"
P1B = PROJECT_ROOT / "results/gate3_semantics/gate3_20260830_primitive_relation_p1b_teacher_materialization_v1r_seed0"
V2_READINESS = PROJECT_ROOT / "results/gate3_semantics/gate3_20260831_primitive_relation_sparse_port_readiness_v1_seed0"
V1_ATTRIBUTION = PROJECT_ROOT / "results/gate3_semantics/gate3_20260831_primitive_relation_v1_failure_attribution_v1r2_seed0"
BASELINE_READINESS = PROJECT_ROOT / "results/gate3_semantics/gate3_20260830_primitive_relation_nonlearning_readiness_v1_seed0"
BASELINE_C07 = PROJECT_ROOT / "results/gate3_semantics/gate3_20260830_primitive_relation_nonlearning_c07_v1_seed0"


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
    prerequisites = (
        P1A, P1B, V2_READINESS, V1_ATTRIBUTION,
        BASELINE_READINESS, BASELINE_C07,
    )
    for source in prerequisites:
        state = json.loads((source / "RUN_STATE.json").read_text())
        summary = json.loads((source / "metrics/summary.json").read_text())
        if (
            state.get("state") != "COMPLETED"
            or state.get("error") is not None
            or not summary.get("scientific_pass")
        ):
            raise RuntimeError(f"sparse-port prerequisite failed: {source.name}")
    approval = {
        "status": "APPROVED",
        "approved_by": "user-standing-authorization",
        "approved_at": "2026-08-31T23:59:00+08:00",
        "authorized_gates": [3],
        "authorized_operations": [
            "training", "checkpoint_selection", "threshold_calibration",
        ],
        "confirmation_reference": (
            "User explicitly authorized continuous autonomous execution, "
            "automatic in-plan choices, and preservation of paper evidence."
        ),
        "scope": (
            "One immutable V2 sparse-port three-seed run: C01-C06 gradients, "
            "C07 checkpoint/threshold selection, and C08 only after at least "
            "two seeds pass the complete C07 geometry, relation, coverage and "
            "non-vacuous >=0.98-precision attachment gate; no C09/C10/graph/M-TARE."
        ),
    }
    card = copy.deepcopy(json.loads(OLD_CARD.read_text(encoding="utf-8")))
    card.update({
        "card_id": "primitive_relation_sparse_port_three_seed_training_v1",
        "status": (
            "APPROVED_FOR_ONE_IMMUTABLE_PRIMITIVE_RELATION_"
            "SPARSE_PORT_THREE_SEED_TRAINING_V1"
        ),
        "approval": approval,
        "purpose": (
            "Train the construction-supervised sparse primitive-set and "
            "endpoint-relation Transformer with seeds 0/1/2, select only on "
            "C07, and test whether it fixes V1's 14x relation-pair expansion "
            "while improving explicit geometry over the same-input robust fitter."
        ),
        "retention": (
            "Keep all 21 epoch checkpoints and histories, three selected "
            "checkpoints, C07 calibrated thresholds/metrics, conditional compact "
            "C08 outputs, comparisons, paper figures, environment, raw logs, "
            "RUN_STATE and SHA-256 seal."
        ),
        "failure_policy": (
            "Any source/tool/environment drift, nonfinite gradient, row/task "
            "mismatch, resource overrun, vacuous safe relation, C07 scientific "
            "failure, C08 regression or forbidden-world read stops. Do not add "
            "losses, epochs, rules, frames or planner tuning after seeing results."
        ),
    })
    card["sampling"]["rule"] = (
        "Every C01-C06 row appears once per epoch in deterministic shard-local "
        "shuffle for seven epochs. C07 is fixed-order selection. C08 is not "
        "instantiated unless at least two seeds pass every frozen C07 check, "
        "including a nonzero true attachment at precision >=0.98."
    )
    card["leakage_audit"]["optimizer_step_count"] = 561_456
    card["metrics_and_pre_registered_gates"] = {
        "implementation": (
            "Exactly 44 frozen unit tests pass; deterministic full-FP32 CUDA, "
            "TF32 off; 2,629,870 parameters and V2 architecture/losses unchanged "
            "from the sealed readiness run."
        ),
        "training": (
            "Seeds 0/1/2, seven epochs, batch16, AdamW lr=3e-4 "
            "weight_decay=1e-4, clip=1.0. Stages: geometry; sparse set/MDL; "
            "endpoint relations; temporal+uncertainty; three equal joint epochs. "
            "Expected 187,152 steps per seed and 561,456 total."
        ),
        "selection": (
            "Choose each seed's minimum C07 equal-six-family loss checkpoint; "
            "calibrate only the frozen 91-point thresholds on C07. An ensemble "
            "cannot rescue fewer than two independently passing seeds."
        ),
        "c07_perception_gate": (
            "At least two seeds each improve surface Chamfer >=10%, macro axis/"
            "width/height/exponent/slope/curvature MAE >=10%, and attachment F1 "
            ">=5 points over the sealed non-learning baseline, without primitive "
            "F1 or target-coverage regression."
        ),
        "c07_safety_gate": (
            "Every passing seed must additionally recover at least one true "
            "attachment at calibrated precision >=0.98 using "
            "p(attachment)*(1-relation uncertainty). Zero accepted relations "
            "is explicitly a failure, not perfect precision."
        ),
        "c08_transfer_gate": (
            "Only after C07 passes, unchanged checkpoints and thresholds must "
            "satisfy the same complete gate on C08 in at least two seeds; otherwise "
            "stop before graph construction."
        ),
        "resources": (
            "Full run <=40 h, GPU allocation <=16 GiB, host process memory "
            "<=16 GiB, output <=6 GiB; zero C09/C10/graph/M-TARE."
        ),
    }
    card["estimated_cost"] = {
        "compute": (
            "One RTX 5090, three seeds sequentially. V1 measured about 21.7 h "
            "for six epochs; the added sparse relation Transformer and seventh "
            "epoch are budgeted under the hard 40 h outer limit."
        ),
        "wall_time_hours": 36,
        "host_ram_gb": 16,
        "gpu": 1,
        "gpu_memory_gb": 16,
        "disk_gb": 6,
    }
    write(CARD, card)

    tools = {
        "runner": "tools/v3/run_primitive_relation_sparse_port_three_seed_training_v1.py",
        "trainer": "tools/v3/train_primitive_relation_sparse_port_v1.py",
        "evaluator": "tools/v3/evaluate_primitive_relation_sparse_port_three_seed_v1.py",
        "base_trainer": "tools/v3/train_primitive_relation_model_v1.py",
        "base_evaluator": "tools/v3/evaluate_primitive_relation_three_seed_v1.py",
        "batch_reader": "src/mtare_topo/data/primitive_relation_batches.py",
        "single_reader": "src/mtare_topo/data/primitive_relation_training.py",
        "base_training_contract": "src/mtare_topo/representation/primitive_relation_training.py",
        "training_contract": "src/mtare_topo/representation/primitive_relation_sparse_port_training.py",
        "base_model": "src/mtare_topo/representation/primitive_relation_model.py",
        "model": "src/mtare_topo/representation/primitive_relation_sparse_port_model.py",
        "base_loss": "src/mtare_topo/representation/primitive_relation_losses.py",
        "loss": "src/mtare_topo/representation/primitive_relation_sparse_port_losses.py",
        "metrics": "src/mtare_topo/evaluation/primitive_relation_metrics.py",
        "nonlearning": "src/mtare_topo/semantics/primitive_relation_nonlearning.py",
        "exit_baseline": "src/mtare_topo/semantics/range_exit_baseline.py",
        "reader_tests": "tests/v3/unit/test_primitive_relation_training.py",
        "base_model_tests": "tests/v3/unit/test_primitive_relation_model.py",
        "baseline_tests": "tests/v3/unit/test_primitive_relation_nonlearning.py",
        "metric_tests": "tests/v3/unit/test_primitive_relation_metrics.py",
        "base_schedule_tests": "tests/v3/unit/test_primitive_relation_training_schedule.py",
        "base_evaluation_tests": "tests/v3/unit/test_evaluate_primitive_relation_three_seed_v1.py",
        "model_tests": "tests/v3/unit/test_primitive_relation_sparse_port.py",
        "schedule_tests": "tests/v3/unit/test_primitive_relation_sparse_port_training.py",
        "evaluation_tests": "tests/v3/unit/test_evaluate_primitive_relation_sparse_port_three_seed_v1.py",
        "governance": "src/mtare_topo/governance.py",
        "preflight": "tools/v3/preflight.py",
        "create_run": "tools/v3/create_run.py",
    }
    inputs = [
        CARD,
        *[
            source / name
            for source in prerequisites
            for name in (
                "RUN_STATE.json", "metrics/summary.json",
                "artifacts/evidence_sha256.txt",
            )
        ],
        P1A / "artifacts/task_manifest.json",
        P1B / "artifacts/task_manifest.json",
        PROJECT_ROOT / "configs/v3/gate2/environments/phase3_torch290_cu129_zarr2187_v1r2.json",
        PROJECT_ROOT / "configs/v3/gate2/environments/phase3_torch290_cu129_zarr2187_pip_freeze.txt",
    ]
    spec = {
        "schema_version": "v3_run_spec_v1",
        "gate": 3,
        "execution_phase": 3,
        "operation": "training",
        "date": "20260831",
        "slug": "primitive_relation_sparse_port_three_seed_training_v1",
        "seed": 0,
        "question": (
            "Can a sparse primitive-set plus endpoint-relation Transformer recover "
            "explicit swept geometry and non-vacuous safe attachments from five "
            "causal LiDAR frames on unseen topologies?"
        ),
        "method": (
            "Train the frozen V2 causal circular encoder, sparse 32-query swept-"
            "primitive decoder and invariant endpoint/primitive relation "
            "Transformers with the seven-stage schedule; select only on C07 and "
            "conditionally evaluate C08 once."
        ),
        "baseline": (
            "Sealed same-input non-learning robust fitter; V1 dense all-pairs "
            "model is retained as the architecture ablation and failure evidence."
        ),
        "fallback": (
            "Scientific failure stops before C08 or graph. Preserve checkpoints "
            "and failure attribution; do not tune the planner or relax gates."
        ),
        "data_card": str(CARD.relative_to(PROJECT_ROOT)),
        "config_path": str(CARD.relative_to(PROJECT_ROOT)),
        "user_authorization": approval,
        "acceptance_criteria": list(card["metrics_and_pre_registered_gates"].values()),
        "expected_counts": {
            "fit_parent_worlds": 60,
            "fit_geometry_tasks": 180,
            "fit_rows_per_epoch": 426_552,
            "epochs_per_seed": 7,
            "seeds": 3,
            "optimizer_steps_per_seed": 187_152,
            "optimizer_steps_total": 561_456,
            "c07_parent_worlds": 10,
            "c07_rows": 64_644,
            "conditional_c08_parent_worlds": 10,
            "conditional_c08_rows_per_method": 73_182,
            "unit_tests": 44,
            "c09_c10_worlds_read": 0,
            "graph_replays": 0,
            "mtare_worlds_read": 0,
        },
        "expected_evidence": [
            "Three complete seven-epoch histories and selected checkpoints; C07 "
            "loss, geometry, relation, uncertainty and non-vacuous safe-threshold "
            "comparisons; conditional C08 compact outputs; raw logs, environment, "
            "paper figure, RUN_STATE and SHA-256 seal."
        ],
        "estimated_cost": card["estimated_cost"],
        "frozen_tools": {
            name: {"path": path, "sha256": sha(PROJECT_ROOT / path)}
            for name, path in tools.items()
        },
        "frozen_inputs": {
            str(path.relative_to(PROJECT_ROOT)): sha(path) for path in inputs
        },
        "working_directory": str(PROJECT_ROOT),
        "command": [
            "/usr/bin/systemd-inhibit", "--what=sleep:shutdown",
            "--why=GSE sparse-port primitive relation three-seed training",
            "--mode=block", "/usr/bin/timeout", "--signal=INT",
            "--kill-after=60s", "144000s", "/usr/bin/env",
            "CUBLAS_WORKSPACE_CONFIG=:4096:8",
            f"PYTHONPATH={PROJECT_ROOT / 'src'}:{PROJECT_ROOT / 'tools/v3'}",
            PYTHON,
            "tools/v3/run_primitive_relation_sparse_port_three_seed_training_v1.py",
            "--spec", str(SPEC),
            "--run-dir", str(PROJECT_ROOT / f"results/gate3_semantics/{RUN_ID}"),
        ],
    }
    write(SPEC, spec)
    print(SPEC)


if __name__ == "__main__":
    main()
