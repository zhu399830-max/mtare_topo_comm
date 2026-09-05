#!/usr/bin/env python3
"""Freeze stable-phase JCGS readiness V4."""
from __future__ import annotations

import hashlib

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, validate_data_card, write_json


RUN_ID = "gate3_20260829_gse_joint_cyclic_gap_simplex_readiness_v4_seed0"
SPEC = PROJECT_ROOT / "configs/v3/gate3/gse_joint_cyclic_gap_simplex_readiness_v4.json"
DATA_CARD = "configs/v3/gate3/data_cards/gse_joint_cyclic_gap_simplex_readiness_v4.json"
PYTHON = "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    if SPEC.exists():
        raise RuntimeError("gap-simplex V4 spec already exists")
    card = load_json(PROJECT_ROOT / DATA_CARD)
    validation = validate_data_card(card)
    if not validation.passed:
        raise RuntimeError(f"gap-simplex V4 Data Card invalid: {validation.errors}")
    dataset = "results/gate2_representation/gate2_20260824_gse_deduplicated_dataset_export_v1_seed0"
    teacher = "results/gate2_representation/gate2_20260829_gse_circular_exit_geometry_field_teacher_export_v1_seed0"
    readiness_v3 = "results/gate3_semantics/gate3_20260829_gse_joint_cyclic_gap_simplex_readiness_v3_seed0"
    failed_training = "results/gate3_semantics/gate3_20260829_gse_joint_cyclic_gap_simplex_three_seed_training_v1_seed0"
    inputs = [
        DATA_CARD,
        f"{dataset}/RUN_STATE.json", f"{dataset}/metrics/summary.json", f"{dataset}/artifacts/evidence_sha256.txt",
        f"{teacher}/RUN_STATE.json", f"{teacher}/metrics/summary.json", f"{teacher}/artifacts/evidence_sha256.txt",
        f"{readiness_v3}/RUN_STATE.json", f"{readiness_v3}/metrics/summary.json", f"{readiness_v3}/artifacts/evidence_sha256.txt",
        f"{failed_training}/RUN_STATE.json", f"{failed_training}/metrics/summary.json", f"{failed_training}/artifacts/evidence_sha256.txt",
        "configs/v3/gate2/environments/phase3_torch290_cu129_zarr2187_v1r2.json",
    ]
    tools = {
        "gap_simplex": "src/mtare_topo/representation/gse_joint_cyclic_gap_simplex.py",
        "slot_transport": "src/mtare_topo/representation/gse_cardinality_conditioned_circular_slot_transport.py",
        "circular_backbone": "src/mtare_topo/representation/gse_circular_peak_geometry_model.py",
        "training_core": "tools/v3/train_gse_circular_peak_geometry_v1.py",
        "readiness_helpers": "tools/v3/execute_gse_circular_peak_geometry_model_readiness_v1.py",
        "executor_v1": "tools/v3/execute_gse_joint_cyclic_gap_simplex_readiness_v1.py",
        "executor": "tools/v3/execute_gse_joint_cyclic_gap_simplex_readiness_v2.py",
        "runner_helpers": "tools/v3/run_gse_joint_cyclic_gap_simplex_readiness_v3.py",
        "runner": "tools/v3/run_gse_joint_cyclic_gap_simplex_readiness_v4.py",
        "freezer": "tools/v3/freeze_gse_joint_cyclic_gap_simplex_readiness_spec_v4.py",
        "tests": "tests/v3/unit/test_gse_joint_cyclic_gap_simplex.py",
        "governance": "src/mtare_topo/governance.py",
        "preflight": "tools/v3/preflight.py",
        "create_run": "tools/v3/create_run.py",
    }
    spec = {
        "schema_version": "v3_run_spec_v1", "gate": 3, "execution_phase": 3,
        "date": "20260829", "slug": "gse_joint_cyclic_gap_simplex_readiness_v4", "seed": 0,
        "operation": "audit", "data_card": DATA_CARD,
        "question": "Does bounded low-resultant phase backward plus explicit finite guards remove the observed numerical failure without changing JCGS forward semantics or readiness properties?",
        "method": "Ordinary atan2 forward; backward denominator floored at float32 phase-mass machine precision; reject non-finite bearing before periodic gather and non-finite gradient norm before optimizer step; repeat all V3 checks.",
        "baseline": "First three-seed run stopped when seed1 triggered CUDA gather out-of-bounds during epoch4 after finite checkpoints; seed0 completed.",
        "fallback": "Stop JCGS before retraining if any stability regression or V3 scientific check fails.",
        "user_authorization": card["approval"],
        "acceptance_criteria": [
            "Exact V3 population, rows, singleton closure and 789650 parameters.",
            "Stable phase forward equals atan2 away from zero and has finite zero-resultant gradient.",
            "Non-finite bearing and gradient are rejected before gather/optimizer.",
            "All V2/V3 phase, gap, symmetry and real finite-backward checks remain PASS.",
            "Zero optimizer/checkpoint/threshold/C09/C10/M-TARE/graph/planner and frozen inputs unchanged."
        ],
        "expected_counts": {"worlds": 80, "observations": 188126, "visible_exits": 396913, "real_rows": 8, "parameters": 789650, "optimizer_steps": 0, "checkpoint_writes": 0, "c09_worlds_read": 0, "c10_worlds_read": 0, "mtare_worlds_read": 0, "graph_replays": 0},
        "expected_evidence": ["Stable phase and finite fail-closed regression log.", "Repeated readiness summary, figures and selected rows.", "Environment, commands, RUN_STATE and SHA-256 seal."],
        "estimated_cost": {"compute": "CPU metadata audit plus eight-row forward/backward", "wall_time_hours": 0.1, "host_ram_gb": 6, "gpu_memory_gb": 0, "disk_gb": 0.1},
        "frozen_inputs": {path: sha256(PROJECT_ROOT / path) for path in sorted(inputs)},
        "frozen_tools": {name: {"path": path, "sha256": sha256(PROJECT_ROOT / path)} for name, path in tools.items()},
        "working_directory": str(PROJECT_ROOT),
        "command": ["/usr/bin/timeout", "1800s", PYTHON, "tools/v3/run_gse_joint_cyclic_gap_simplex_readiness_v4.py", "--spec", str(SPEC), "--run-dir", str(PROJECT_ROOT / f"results/gate3_semantics/{RUN_ID}")]
    }
    write_json(SPEC, spec)
    print(SPEC.relative_to(PROJECT_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
