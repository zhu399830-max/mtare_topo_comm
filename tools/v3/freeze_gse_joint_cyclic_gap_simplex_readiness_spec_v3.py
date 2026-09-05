#!/usr/bin/env python3
"""Freeze singleton-corrected JCGS readiness V3."""
from __future__ import annotations

import hashlib

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, validate_data_card, write_json


RUN_ID = "gate3_20260829_gse_joint_cyclic_gap_simplex_readiness_v3_seed0"
SPEC = PROJECT_ROOT / "configs/v3/gate3/gse_joint_cyclic_gap_simplex_readiness_v3.json"
DATA_CARD = "configs/v3/gate3/data_cards/gse_joint_cyclic_gap_simplex_readiness_v3.json"
PYTHON = "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    if SPEC.exists():
        raise RuntimeError("gap-simplex V3 spec already exists")
    card = load_json(PROJECT_ROOT / DATA_CARD)
    validation = validate_data_card(card)
    if not validation.passed:
        raise RuntimeError(f"gap-simplex V3 Data Card invalid: {validation.errors}")
    dataset = "results/gate2_representation/gate2_20260824_gse_deduplicated_dataset_export_v1_seed0"
    teacher = "results/gate2_representation/gate2_20260829_gse_circular_exit_geometry_field_teacher_export_v1_seed0"
    readiness_v2 = "results/gate3_semantics/gate3_20260829_gse_joint_cyclic_gap_simplex_readiness_v2_seed0"
    inputs = [
        DATA_CARD,
        f"{dataset}/RUN_STATE.json", f"{dataset}/metrics/summary.json", f"{dataset}/artifacts/evidence_sha256.txt",
        f"{teacher}/RUN_STATE.json", f"{teacher}/metrics/summary.json", f"{teacher}/artifacts/evidence_sha256.txt",
        f"{readiness_v2}/RUN_STATE.json", f"{readiness_v2}/metrics/summary.json", f"{readiness_v2}/artifacts/evidence_sha256.txt",
        "configs/v3/gate2/environments/phase3_torch290_cu129_zarr2187_v1r2.json",
    ]
    tools = {
        "gap_simplex": "src/mtare_topo/representation/gse_joint_cyclic_gap_simplex.py",
        "slot_transport": "src/mtare_topo/representation/gse_cardinality_conditioned_circular_slot_transport.py",
        "circular_backbone": "src/mtare_topo/representation/gse_circular_peak_geometry_model.py",
        "readiness_helpers": "tools/v3/execute_gse_circular_peak_geometry_model_readiness_v1.py",
        "executor_v1": "tools/v3/execute_gse_joint_cyclic_gap_simplex_readiness_v1.py",
        "executor": "tools/v3/execute_gse_joint_cyclic_gap_simplex_readiness_v2.py",
        "runner": "tools/v3/run_gse_joint_cyclic_gap_simplex_readiness_v3.py",
        "freezer": "tools/v3/freeze_gse_joint_cyclic_gap_simplex_readiness_spec_v3.py",
        "tests": "tests/v3/unit/test_gse_joint_cyclic_gap_simplex.py",
        "governance": "src/mtare_topo/governance.py",
        "preflight": "tools/v3/preflight.py",
        "create_run": "tools/v3/create_run.py",
    }
    spec = {
        "schema_version": "v3_run_spec_v1",
        "gate": 3,
        "execution_phase": 3,
        "date": "20260829",
        "slug": "gse_joint_cyclic_gap_simplex_readiness_v3",
        "seed": 0,
        "operation": "audit",
        "data_card": DATA_CARD,
        "question": "Does the singleton-corrected joint cyclic gap representation retain every V2 readiness property while encoding the one-exit closing gap as 360 degrees?",
        "method": "Repeat V2 readiness on the same population and rows, plus a frozen K=1 circular closure regression. No training or threshold selection.",
        "baseline": "V2 passed eleven readiness checks but mapped a singleton target's rolled angular difference to 0 rather than its mathematically required 360-degree closing gap.",
        "fallback": "Stop before training if the singleton regression or any V2 readiness check fails.",
        "user_authorization": card["approval"],
        "acceptance_criteria": [
            "Exact V2 population, eight rows and 789650 parameters.",
            "K=1 target closing gap equals 360 degrees for arbitrary bearings.",
            "All V2 phase likelihood, refusal, closure, gradient, symmetry and finite-backward checks remain PASS.",
            "Zero optimizer/checkpoint/threshold/C09/C10/M-TARE/graph/planner and frozen inputs unchanged."
        ],
        "expected_counts": {"worlds": 80, "observations": 188126, "visible_exits": 396913, "cardinality_1": 7525, "real_rows": 8, "parameters": 789650, "optimizer_steps": 0, "checkpoint_writes": 0, "c09_worlds_read": 0, "c10_worlds_read": 0, "mtare_worlds_read": 0, "graph_replays": 0},
        "expected_evidence": ["Singleton closure regression log.", "Repeated V2 readiness summary, figures and selected rows.", "Environment, commands, RUN_STATE and SHA-256 seal."],
        "estimated_cost": {"compute": "CPU metadata audit plus eight-row forward/backward", "wall_time_hours": 0.1, "host_ram_gb": 6, "gpu_memory_gb": 0, "disk_gb": 0.1},
        "frozen_inputs": {path: sha256(PROJECT_ROOT / path) for path in sorted(inputs)},
        "frozen_tools": {name: {"path": path, "sha256": sha256(PROJECT_ROOT / path)} for name, path in tools.items()},
        "working_directory": str(PROJECT_ROOT),
        "command": ["/usr/bin/timeout", "1800s", PYTHON, "tools/v3/run_gse_joint_cyclic_gap_simplex_readiness_v3.py", "--spec", str(SPEC), "--run-dir", str(PROJECT_ROOT / f"results/gate3_semantics/{RUN_ID}")]
    }
    write_json(SPEC, spec)
    print(SPEC.relative_to(PROJECT_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
