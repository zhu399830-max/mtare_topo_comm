#!/usr/bin/env python3
"""Freeze circular peak geometry three-seed training."""

from __future__ import annotations

import hashlib

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import write_json


RUN_ID = "gate3_20260829_gse_circular_peak_geometry_three_seed_training_v1_seed0"
SPEC = PROJECT_ROOT / "configs/v3/gate3/gse_circular_peak_geometry_three_seed_training_v1.json"
DATA_CARD = "configs/v3/gate3/data_cards/gse_circular_peak_geometry_three_seed_training_v1.json"
PYTHON = "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    if SPEC.exists():
        raise RuntimeError("circular peak three-seed training V1 spec already exists")
    dataset = "results/gate2_representation/gate2_20260824_gse_deduplicated_dataset_export_v1_seed0"
    teacher = "results/gate2_representation/gate2_20260829_gse_circular_exit_geometry_field_teacher_export_v1_seed0"
    readiness = "results/gate3_semantics/gate3_20260829_gse_circular_peak_geometry_model_readiness_v1_seed0"
    inputs = [
        DATA_CARD,
        f"{dataset}/RUN_STATE.json", f"{dataset}/metrics/summary.json", f"{dataset}/artifacts/shard_manifest.json", f"{dataset}/artifacts/evidence_sha256.txt",
        f"{teacher}/RUN_STATE.json", f"{teacher}/metrics/summary.json", f"{teacher}/artifacts/export/teacher_shard_manifest.jsonl", f"{teacher}/artifacts/evidence_sha256.txt",
        f"{readiness}/RUN_STATE.json", f"{readiness}/metrics/summary.json", f"{readiness}/artifacts/evidence_sha256.txt",
        "configs/v3/gate2/environments/phase3_torch290_cu129_zarr2187_v1r2.json",
    ]
    tools = {
        "model": "src/mtare_topo/representation/gse_circular_peak_geometry_model.py",
        "circular_layers": "src/mtare_topo/representation/phase3_structural_semantics.py",
        "metrics": "src/mtare_topo/evaluation/gse_circular_peak_metrics.py",
        "trainer": "tools/v3/train_gse_circular_peak_geometry_v1.py",
        "evaluator": "tools/v3/evaluate_gse_circular_peak_geometry_selection_v1.py",
        "runner": "tools/v3/run_gse_circular_peak_geometry_three_seed_training_v1.py",
        "freezer": "tools/v3/freeze_gse_circular_peak_geometry_three_seed_training_spec_v1.py",
        "model_tests": "tests/v3/unit/test_gse_circular_peak_geometry_model.py",
        "metric_tests": "tests/v3/unit/test_gse_circular_peak_metrics.py",
        "governance": "src/mtare_topo/governance.py", "preflight": "tools/v3/preflight.py", "create_run": "tools/v3/create_run.py",
    }
    spec = {
        "schema_version": "v3_run_spec_v1", "gate": 3, "execution_phase": 3,
        "date": "20260829", "slug": "gse_circular_peak_geometry_three_seed_training_v1", "seed": 0,
        "operation": "training", "data_card": DATA_CARD,
        "question": "Can the C01-C06-trained circular model recover high-precision executable peak sets and full geometry on C07 and transfer the frozen checkpoint/threshold once to C08?",
        "method": "Three from-scratch 769268-parameter seeds, 10 epochs each, exact circular augmentation, C07 minimum-loss checkpointing and one probability-mean ensemble threshold.",
        "baseline": "Frozen raw-range AP 0.051493 and historical free-query high-precision recall <=0.093888.",
        "fallback": "Any C07/C08 safe peak, action-set or geometry failure stops before descriptor association and graph; no retry or planner compensation.",
        "user_authorization": {"status": "APPROVED", "approved_by": "user-standing-authorization", "approved_at": "2026-08-29T04:05:00+08:00", "authorized_gates": [3], "authorized_operations": ["training"], "scope": "One immutable three-seed C01-C06 training with C07 selection and one C08 transfer.", "confirmation_reference": "User explicitly delegated best-method decisions and continuous execution."},
        "acceptance_criteria": [
            "Exactly three seeds, 10 epochs/11370 steps each and 34110 total optimizer steps; no old checkpoint initialization.",
            "C07-selected ensemble precision>=0.995 and recall>=0.50 on both C07 and zero-adaptation C08; AP gain>=0.10 over raw range.",
            "Terminal/corridor/junction macro-F1>=0.80 on both splits.",
            "Peak heading/width/profile and global axis/width/height/slope/curvature satisfy the frozen Data Card bounds on both splits.",
            "C08 checkpoint observations/updates zero; all seeds, per-world results and one ensemble threshold retained.",
            "Zero C09/C10/M-TARE/graph/planner and all frozen sources unchanged."
        ],
        "expected_counts": {"fit_worlds": 60, "c07_worlds": 10, "c08_worlds": 10, "fit_observations": 142184, "c07_observations": 21548, "c08_observations": 24394, "fit_peaks": 299872, "c07_peaks": 45504, "c08_peaks": 51537, "parameters": 769268, "seeds": 3, "epochs_per_seed": 10, "optimizer_steps_per_seed": 11370, "optimizer_steps": 34110, "c08_checkpoint_observations": 0, "c09_worlds_read": 0, "c10_worlds_read": 0, "mtare_worlds_read": 0, "graph_replays": 0},
        "expected_evidence": ["Three checkpoints, histories, seed summaries and compact C07/C08 predictions.", "C07 threshold selection, C08 transfer, seed diagnostics and per-world metrics.", "PNG/PDF/SVG/source, environment, commands, raw logs, RUN_STATE and seal."],
        "estimated_cost": {"compute": "Three sequential CUDA trainings and development evaluation", "wall_time_hours": 6, "host_ram_gb": 16, "gpu_memory_gb": 16, "disk_gb": 2, "gpu": "one NVIDIA GeForce RTX 5090 D"},
        "frozen_inputs": {path: sha256(PROJECT_ROOT / path) for path in sorted(inputs)},
        "frozen_tools": {name: {"path": path, "sha256": sha256(PROJECT_ROOT / path)} for name, path in tools.items()},
        "working_directory": str(PROJECT_ROOT),
        "command": ["/usr/bin/timeout", "86400s", PYTHON, "tools/v3/run_gse_circular_peak_geometry_three_seed_training_v1.py", "--spec", str(SPEC), "--run-dir", str(PROJECT_ROOT / f"results/gate3_semantics/{RUN_ID}")],
    }
    write_json(SPEC, spec)
    print(SPEC.relative_to(PROJECT_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
