#!/usr/bin/env python3
"""Freeze synchronous seed1 CUDA index attribution spec."""
from __future__ import annotations
import hashlib
from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, validate_data_card, write_json

RUN_ID = "gate3_20260829_gse_joint_cyclic_gap_simplex_seed1_cuda_index_attribution_v1_seed1"
SPEC = PROJECT_ROOT / "configs/v3/gate3/gse_joint_cyclic_gap_simplex_seed1_cuda_index_attribution_v1.json"
DATA_CARD = "configs/v3/gate3/data_cards/gse_joint_cyclic_gap_simplex_seed1_cuda_index_attribution_v1.json"
PYTHON = "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python"

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()

def main() -> int:
    if SPEC.exists(): raise RuntimeError("seed1 attribution spec already exists")
    card = load_json(PROJECT_ROOT / DATA_CARD); validation = validate_data_card(card)
    if not validation.passed: raise RuntimeError(f"seed1 attribution card invalid: {validation.errors}")
    dataset = "results/gate2_representation/gate2_20260824_gse_deduplicated_dataset_export_v1_seed0"
    teacher = "results/gate2_representation/gate2_20260829_gse_circular_exit_geometry_field_teacher_export_v1_seed0"
    readiness = "results/gate3_semantics/gate3_20260829_gse_joint_cyclic_gap_simplex_readiness_v4_seed0"
    failed_v1 = "results/gate3_semantics/gate3_20260829_gse_joint_cyclic_gap_simplex_three_seed_training_v1_seed0"
    failed_v1r = "results/gate3_semantics/gate3_20260829_gse_joint_cyclic_gap_simplex_three_seed_training_v1r_seed0"
    inputs = [DATA_CARD, f"{dataset}/RUN_STATE.json", f"{dataset}/metrics/summary.json", f"{dataset}/artifacts/evidence_sha256.txt", f"{teacher}/RUN_STATE.json", f"{teacher}/metrics/summary.json", f"{teacher}/artifacts/evidence_sha256.txt", f"{readiness}/RUN_STATE.json", f"{readiness}/metrics/summary.json", f"{readiness}/artifacts/evidence_sha256.txt", f"{failed_v1}/RUN_STATE.json", f"{failed_v1}/metrics/summary.json", f"{failed_v1}/artifacts/evidence_sha256.txt", f"{failed_v1r}/RUN_STATE.json", f"{failed_v1r}/metrics/summary.json", f"{failed_v1r}/artifacts/evidence_sha256.txt", "configs/v3/gate2/environments/phase3_torch290_cu129_zarr2187_v1r2.json"]
    tools = {
        "gap_simplex": "src/mtare_topo/representation/gse_joint_cyclic_gap_simplex.py", "parent_slot_transport": "src/mtare_topo/representation/gse_cardinality_conditioned_circular_slot_transport.py", "backbone": "src/mtare_topo/representation/gse_circular_peak_geometry_model.py", "training_core": "tools/v3/train_gse_circular_peak_geometry_v1.py", "trainer": "tools/v3/train_gse_joint_cyclic_gap_simplex_v1.py", "runner_helpers": "tools/v3/run_gse_joint_cyclic_gap_simplex_three_seed_training_v1.py", "runner": "tools/v3/run_gse_joint_cyclic_gap_simplex_seed1_cuda_index_attribution_v1.py", "freezer": "tools/v3/freeze_gse_joint_cyclic_gap_simplex_seed1_cuda_index_attribution_spec_v1.py", "model_tests": "tests/v3/unit/test_gse_joint_cyclic_gap_simplex.py", "governance": "src/mtare_topo/governance.py", "preflight": "tools/v3/preflight.py", "create_run": "tools/v3/create_run.py"
    }
    spec = {
        "schema_version": "v3_run_spec_v1", "gate": 3, "execution_phase": 3, "date": "20260829", "slug": "gse_joint_cyclic_gap_simplex_seed1_cuda_index_attribution_v1", "seed": 1, "operation": "training", "data_card": DATA_CARD,
        "question": "Which exact gather call and finite index context causes the repeatable JCGS seed1 epoch4 CUDA out-of-bounds failure?",
        "method": "Exact seed1 first-five-epoch replay with CUDA_LAUNCH_BLOCKING=1, pre-batch JSON trace, unchanged model/loss/order/optimizer and forced stop before development inference.",
        "baseline": "V1 and stable-phase V1R both fail asynchronously in seed1 epoch4 at about 16m25s.",
        "fallback": "If not reproduced, seal five-epoch non-reproduction and stop; no correction or C08 read.",
        "user_authorization": card["approval"],
        "acceptance_criteria": ["Exact seed1 order/model/loss and at most five epochs/5685 steps.", "CUDA_LAUNCH_BLOCKING=1 and context before every batch.", "Reproduction records exact stack line and last epoch/world/batch/shift/index ranges; non-reproduction stops before C08.", "C08/C09/C10/M-TARE/graph/planner zero; sources unchanged."],
        "expected_counts": {"fit_worlds": 60, "c07_worlds": 10, "fit_observations": 142184, "c07_observations": 21548, "seed": 1, "maximum_epochs": 5, "maximum_optimizer_steps": 5685, "c08_worlds_read": 0, "c09_worlds_read": 0, "c10_worlds_read": 0, "mtare_worlds_read": 0, "graph_replays": 0},
        "expected_evidence": ["Synchronous raw log with per-batch context and exact first failing stack.", "Parsed diagnosis with attempted/successful step counts and bounds.", "Environment, command, RUN_STATE and seal."],
        "estimated_cost": {"compute": "One seed1 CUDA replay for at most five epochs", "wall_time_hours": 0.4, "host_ram_gb": 8, "gpu_memory_gb": 12, "disk_gb": 0.2, "gpu": "one NVIDIA GeForce RTX 5090 D"},
        "frozen_inputs": {path: sha256(PROJECT_ROOT / path) for path in sorted(inputs)}, "frozen_tools": {name: {"path": path, "sha256": sha256(PROJECT_ROOT / path)} for name, path in tools.items()}, "working_directory": str(PROJECT_ROOT),
        "command": ["/usr/bin/timeout", "7200s", PYTHON, "tools/v3/run_gse_joint_cyclic_gap_simplex_seed1_cuda_index_attribution_v1.py", "--spec", str(SPEC), "--run-dir", str(PROJECT_ROOT / f"results/gate3_semantics/{RUN_ID}")]
    }
    write_json(SPEC, spec); print(SPEC.relative_to(PROJECT_ROOT)); return 0

if __name__ == "__main__": raise SystemExit(main())
