#!/usr/bin/env python3
from __future__ import annotations

import hashlib

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import write_json


RUN_ID = "gate3_20260828_gse_geometry_anchored_joint_capacity_v1_seed0"
SPEC = PROJECT_ROOT / "configs/v3/gate3/gse_geometry_anchored_joint_capacity_v1.json"
DATA_CARD = "configs/v3/gate3/data_cards/gse_geometry_anchored_joint_training_v1.json"
PYTHON = "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    if SPEC.exists():
        raise RuntimeError("joint capacity V1 spec already exists")
    dataset = "results/gate2_representation/gate2_20260824_gse_deduplicated_dataset_export_v1_seed0"
    teacher = "results/gate2_representation/gate2_20260828_gse_observable_spatial_event_teacher_v2_seed0"
    readiness = "results/gate3_semantics/gate3_20260828_gse_geometry_anchored_spatial_event_readiness_v2_seed0"
    old_capacity = "results/gate3_semantics/gate3_20260828_gse_spatial_event_set_capacity_v1_seed0"
    inputs = [
        DATA_CARD,
        f"{dataset}/RUN_STATE.json",
        f"{dataset}/metrics/summary.json",
        f"{dataset}/artifacts/evidence_sha256.txt",
        f"{dataset}/artifacts/shard_manifest.json",
        f"{teacher}/RUN_STATE.json",
        f"{teacher}/metrics/summary.json",
        f"{teacher}/artifacts/evidence_sha256.txt",
        f"{teacher}/artifacts/export/summary.json",
        f"{teacher}/artifacts/export/teacher_shard_manifest.jsonl",
        f"{readiness}/RUN_STATE.json",
        f"{readiness}/metrics/summary.json",
        f"{readiness}/artifacts/evidence_sha256.txt",
        f"{old_capacity}/RUN_STATE.json",
        f"{old_capacity}/metrics/summary.json",
        f"{old_capacity}/artifacts/evidence_sha256.txt",
        f"{old_capacity}/metrics/capacity/baseline_selection_outputs.npz",
    ]
    for seed in (0, 1, 2):
        inputs.extend(
            (
                f"{old_capacity}/artifacts/models/seed{seed}/summary.json",
                f"{old_capacity}/artifacts/models/seed{seed}/selection_outputs.npz",
            )
        )
    tools = {
        "model": "src/mtare_topo/representation/gse_geometry_anchored_event.py",
        "set_loss": "src/mtare_topo/representation/gse_spatial_event_set.py",
        "dataset_loader": "src/mtare_topo/data/gse_observable_spatial_event_dataset.py",
        "metrics": "src/mtare_topo/evaluation/gse_spatial_event_set_metrics.py",
        "trainer": "tools/v3/train_gse_geometry_anchored_spatial_event_v1.py",
        "evaluator": "tools/v3/evaluate_gse_geometry_anchored_joint_capacity_v1.py",
        "runner": "tools/v3/run_gse_geometry_anchored_joint_capacity_v1.py",
        "freezer": "tools/v3/freeze_gse_geometry_anchored_joint_capacity_v1_spec.py",
        "model_tests": "tests/v3/unit/test_gse_geometry_anchored_event.py",
        "set_tests": "tests/v3/unit/test_gse_spatial_event_set.py",
        "dataset_tests": "tests/v3/unit/test_gse_observable_spatial_event_dataset.py",
        "governance": "src/mtare_topo/governance.py",
        "preflight": "tools/v3/preflight.py",
        "create_run": "tools/v3/create_run.py",
    }
    authorization = {
        "status": "APPROVED",
        "approved_by": "user-standing-authorization",
        "approved_at": "2026-08-28T22:20:00+08:00",
        "authorized_gates": [3],
        "authorized_operations": ["training"],
        "scope": "One immutable three-seed C01-C08 GeometryAnchoredSpatialEventEncoder joint capacity run.",
        "confirmation_reference": "User instructed automatic best-choice execution without routine approval prompts.",
    }
    spec = {
        "schema_version": "v3_run_spec_v1",
        "gate": 3,
        "execution_phase": 3,
        "date": "20260828",
        "slug": "gse_geometry_anchored_joint_capacity_v1",
        "seed": 0,
        "operation": "training",
        "data_card": DATA_CARD,
        "question": "Can end-to-end five-frame LiDAR geometry learning improve C07-C08 spatial-event macro-F1 by at least five points over every sealed development baseline while retaining safe precision, multi-event recall and localization?",
        "method": "Train all 264134 parameters of the GeometryAnchoredSpatialEventEncoder for eight fixed epochs on every C01-C06 observation, for seeds 0/1/2; select minimum C07-C08 set-loss checkpoints and confidence only on the fixed 0.05 grid.",
        "baseline": "Recompute the sealed exclusive single-center exit model, nonlearning geometric event estimator and three frozen-encoder set-prediction seeds on the same observable Teacher V2 C07-C08 population.",
        "fallback": "If any seed misses a frozen gate, stop before topology replay and attribute the failure to representation, objective or sample support; do not tune the graph, planner, Teacher, thresholds or strict-test worlds.",
        "user_authorization": authorization,
        "acceptance_criteria": [
            "Exactly 142184 C01-C06 fit rows and 45942 C07-C08 selection rows; 98279/33145 target tokens; zero C09/C10/M-TARE access.",
            "Three seeds, eight epochs and exactly 9096 optimizer steps per seed with all 264134 parameters trainable.",
            "Every seed precision >=0.90, recall >=0.25, terminal and junction recall >=0.20, and matched localization MAE <=4m.",
            "Every seed overall and macro F1 exceeds the best recomputed baseline by >=0.05 and multi-event recall by >=0.10.",
            "Inputs and tools remain unchanged; checkpoints, selection outputs, history, logs, metrics, publication figure, RUN_STATE and SHA-256 seal are complete.",
        ],
        "expected_counts": {
            "worlds": 80,
            "fit_observations": 142184,
            "selection_observations": 45942,
            "fit_tokens": 98279,
            "selection_tokens": 33145,
            "seeds": 3,
            "epochs_per_seed": 8,
            "optimizer_steps_per_seed": 9096,
            "optimizer_steps_total": 27288,
            "parameters_per_seed": 264134,
            "c09_worlds_read": 0,
            "c10_worlds_read": 0,
            "mtare_worlds_read": 0,
        },
        "expected_evidence": [
            "Three selected checkpoints, histories and complete C07-C08 token predictions with descriptors and uncertainty.",
            "Recomputed common-population baseline table, per-seed gates, PNG/PDF/SVG paper figure and exact source JSON.",
            "Environment, commands, raw logs, outer summary, RUN_STATE and exact SHA-256 evidence seal.",
        ],
        "estimated_cost": {
            "compute": "Three sequential deterministic RTX 5090 D trainings; smoke measured 1.84 seconds and 176 MB peak allocation for one batch including validation plumbing.",
            "wall_time_hours": 6.0,
            "host_ram_gb": 8,
            "gpu_memory_gb": 16,
            "disk_gb": 3.0,
            "gpu": "one NVIDIA GeForce RTX 5090 D, sequential seeds",
        },
        "frozen_inputs": {path: sha256(PROJECT_ROOT / path) for path in sorted(inputs)},
        "frozen_tools": {
            name: {"path": path, "sha256": sha256(PROJECT_ROOT / path)}
            for name, path in tools.items()
        },
        "working_directory": str(PROJECT_ROOT),
        "command": [
            "/usr/bin/timeout",
            "23400s",
            PYTHON,
            "tools/v3/run_gse_geometry_anchored_joint_capacity_v1.py",
            "--spec",
            str(SPEC),
            "--run-dir",
            str(PROJECT_ROOT / f"results/gate3_semantics/{RUN_ID}"),
        ],
    }
    write_json(SPEC, spec)
    print(SPEC.relative_to(PROJECT_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
