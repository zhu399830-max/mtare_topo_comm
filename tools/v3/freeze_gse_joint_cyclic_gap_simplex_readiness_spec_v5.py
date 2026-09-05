#!/usr/bin/env python3
"""Freeze the attributed circular-boundary JCGS readiness V5 spec."""
from __future__ import annotations

import hashlib

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, validate_data_card, write_json


RUN_ID = "gate3_20260829_gse_joint_cyclic_gap_simplex_readiness_v5_seed0"
SPEC = PROJECT_ROOT / "configs/v3/gate3/gse_joint_cyclic_gap_simplex_readiness_v5.json"
DATA_CARD = "configs/v3/gate3/data_cards/gse_joint_cyclic_gap_simplex_readiness_v5.json"
PYTHON = "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    if SPEC.exists():
        raise RuntimeError("gap-simplex V5 spec already exists")
    card = load_json(PROJECT_ROOT / DATA_CARD)
    validation = validate_data_card(card)
    if not validation.passed:
        raise RuntimeError(f"gap-simplex V5 Data Card invalid: {validation.errors}")
    dataset = "results/gate2_representation/gate2_20260824_gse_deduplicated_dataset_export_v1_seed0"
    teacher = "results/gate2_representation/gate2_20260829_gse_circular_exit_geometry_field_teacher_export_v1_seed0"
    readiness_v4 = "results/gate3_semantics/gate3_20260829_gse_joint_cyclic_gap_simplex_readiness_v4_seed0"
    attribution = "results/gate3_semantics/gate3_20260829_gse_joint_cyclic_gap_simplex_seed1_cuda_index_attribution_v1_seed1"
    inputs = [
        DATA_CARD,
        f"{dataset}/RUN_STATE.json", f"{dataset}/metrics/summary.json", f"{dataset}/artifacts/evidence_sha256.txt",
        f"{teacher}/RUN_STATE.json", f"{teacher}/metrics/summary.json", f"{teacher}/artifacts/evidence_sha256.txt",
        f"{readiness_v4}/RUN_STATE.json", f"{readiness_v4}/metrics/summary.json", f"{readiness_v4}/artifacts/evidence_sha256.txt",
        f"{attribution}/RUN_STATE.json", f"{attribution}/metrics/summary.json", f"{attribution}/artifacts/evidence_sha256.txt",
        "configs/v3/gate2/environments/phase3_torch290_cu129_zarr2187_v1r2.json",
    ]
    tools = {
        "gap_simplex": "src/mtare_topo/representation/gse_joint_cyclic_gap_simplex.py",
        "circular_backbone": "src/mtare_topo/representation/gse_circular_peak_geometry_model.py",
        "training_core": "tools/v3/train_gse_circular_peak_geometry_v1.py",
        "boundary_executor": "tools/v3/execute_gse_joint_cyclic_gap_simplex_boundary_index_readiness_v1.py",
        "prior_executor": "tools/v3/execute_gse_joint_cyclic_gap_simplex_readiness_v2.py",
        "prior_executor_helpers": "tools/v3/execute_gse_joint_cyclic_gap_simplex_readiness_v1.py",
        "runner_helpers": "tools/v3/run_gse_joint_cyclic_gap_simplex_readiness_v3.py",
        "runner": "tools/v3/run_gse_joint_cyclic_gap_simplex_readiness_v5.py",
        "freezer": "tools/v3/freeze_gse_joint_cyclic_gap_simplex_readiness_spec_v5.py",
        "tests": "tests/v3/unit/test_gse_joint_cyclic_gap_simplex.py",
        "governance": "src/mtare_topo/governance.py",
        "preflight": "tools/v3/preflight.py",
        "create_run": "tools/v3/create_run.py",
    }
    spec = {
        "schema_version": "v3_run_spec_v1", "gate": 3, "execution_phase": 3,
        "date": "20260829", "slug": "gse_joint_cyclic_gap_simplex_readiness_v5", "seed": 0,
        "operation": "audit", "data_card": DATA_CARD,
        "question": "Does canonical circular endpoint indexing remove the exactly attributed seed-1 phase-gather failure while preserving JCGS semantics and finite gradients?",
        "method": "Reconstruct seed1/epoch4/S10_3d_complex_C01/batch9/shift360, prove raw index 180 at row 180664, map the circular endpoint modulo 180 to bin0, run exact CUDA loss forward/backward, then repeat all prior readiness checks.",
        "baseline": "Synchronous CUDA attribution reproduced one ScatterGatherKernel failure at phase-log-mass gather after 5673 optimizer steps.",
        "fallback": "Stop JCGS before retraining if exact attribution, corrected range, CUDA backward, finite gradients or prior readiness fails.",
        "user_authorization": card["approval"],
        "acceptance_criteria": [
            "Exact formal batch contains global sequence 180664 and reproduces exactly one raw lower index 180.",
            "Canonical periodic coordinates map the endpoint to 0 and every corrected index is 0..179 without clipping.",
            "Exact 128-row CUDA JCGS loss forward/backward completes with finite gradients.",
            "All 12 unit regressions and prior readiness checks pass.",
            "Zero optimizer/checkpoint/threshold/C09/C10/M-TARE/graph/planner and frozen inputs unchanged."
        ],
        "expected_counts": {"worlds": 80, "observations": 188126, "visible_exits": 396913, "attributed_batch_rows": 128, "attributed_global_sequence_index": 180664, "raw_out_of_bounds": 1, "canonical_out_of_bounds": 0, "parameters": 789650, "optimizer_steps": 0, "checkpoint_writes": 0, "c09_worlds_read": 0, "c10_worlds_read": 0, "mtare_worlds_read": 0, "graph_replays": 0},
        "expected_evidence": ["Exact boundary coordinate and CUDA replay summary.", "Unit and prior readiness logs/figures.", "Environment, commands, RUN_STATE and SHA-256 seal."],
        "estimated_cost": {"compute": "One CUDA batch forward/backward plus CPU readiness", "wall_time_hours": 0.15, "host_ram_gb": 6, "gpu_memory_gb": 4, "disk_gb": 0.1},
        "frozen_inputs": {path: sha256(PROJECT_ROOT / path) for path in sorted(inputs)},
        "frozen_tools": {name: {"path": path, "sha256": sha256(PROJECT_ROOT / path)} for name, path in tools.items()},
        "working_directory": str(PROJECT_ROOT),
        "command": ["/usr/bin/timeout", "1800s", PYTHON, "tools/v3/run_gse_joint_cyclic_gap_simplex_readiness_v5.py", "--spec", str(SPEC), "--run-dir", str(PROJECT_ROOT / f"results/gate3_semantics/{RUN_ID}")]
    }
    write_json(SPEC, spec)
    print(SPEC.relative_to(PROJECT_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
