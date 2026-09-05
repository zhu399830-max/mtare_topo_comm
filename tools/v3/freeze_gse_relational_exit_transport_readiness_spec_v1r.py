#!/usr/bin/env python3
"""Freeze the immutable V1R readiness system corrective."""

from __future__ import annotations

import hashlib

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import write_json


RUN_ID = "gate3_20260829_gse_relational_exit_transport_readiness_v1r_seed0"
SPEC = PROJECT_ROOT / "configs/v3/gate3/gse_relational_exit_transport_readiness_v1r.json"
DATA_CARD = "configs/v3/gate3/data_cards/gse_relational_exit_transport_readiness_v1r.json"
PYTHON = "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    if SPEC.exists():
        raise RuntimeError("V1R readiness spec already exists")
    action = "results/gate3_semantics/gate3_20260828_gse_action_set_node_training_v1_seed0"
    cache = f"{action}/scratch/action_set_cache"
    unified = "results/gate3_semantics/gate3_20260827_gse_factorized_association_capacity_v1r_seed0"
    feasibility = "results/gate3_semantics/gate3_20260828_gse_exit_action_transport_feasibility_v1_seed0"
    failed = "results/gate3_semantics/gate3_20260828_gse_relational_exit_transport_readiness_v1_seed0"
    inputs = [
        DATA_CARD,
        f"{action}/RUN_STATE.json", f"{action}/metrics/summary.json", f"{action}/artifacts/evidence_sha256.txt", f"{cache}/manifest.json",
        f"{unified}/RUN_STATE.json", f"{unified}/metrics/summary.json", f"{unified}/artifacts/evidence_sha256.txt",
        f"{feasibility}/RUN_STATE.json", f"{feasibility}/metrics/summary.json", f"{feasibility}/artifacts/evidence_sha256.txt",
        f"{failed}/RUN_STATE.json", f"{failed}/metrics/summary.json", f"{failed}/artifacts/evidence_sha256.txt",
    ]
    inputs.extend(f"{cache}/{name}.npy" for name in (
        "raw_tokens", "history_references", "history_mask", "normalization_mean", "normalization_scale",
        "decision_target", "decision_episode_id", "partition_code", "traversal_id", "sequence_index",
    ))
    for seed in range(3):
        inputs.append(f"{unified}/artifacts/unified_observation/seed{seed}_unified_observation_features.npy")
    tools = {
        "model": "src/mtare_topo/representation/gse_relational_exit_transport_event.py",
        "v1_executor": "tools/v3/execute_gse_relational_exit_transport_readiness_v1.py",
        "v1r_executor": "tools/v3/execute_gse_relational_exit_transport_readiness_v1r.py",
        "runner": "tools/v3/run_gse_relational_exit_transport_readiness_v1r.py",
        "freezer": "tools/v3/freeze_gse_relational_exit_transport_readiness_spec_v1r.py",
        "tests_model": "tests/v3/unit/test_gse_relational_exit_transport_event.py",
        "tests_corrective": "tests/v3/unit/test_gse_relational_exit_transport_readiness_v1r.py",
        "governance": "src/mtare_topo/governance.py",
        "preflight": "tools/v3/preflight.py",
        "create_run": "tools/v3/create_run.py",
    }
    spec = {
        "schema_version": "v3_run_spec_v1", "gate": 3, "execution_phase": 3,
        "date": "20260829", "slug": "gse_relational_exit_transport_readiness_v1r", "seed": 0,
        "operation": "audit", "data_card": DATA_CARD,
        "question": "Does unchanged relational readiness pass after replacing the unavailable torch.flatnonzero call with its exact supported equivalent?",
        "method": "Bind the immutable V1 system FAIL and repeat the same executor through a one-line flatnonzero compatibility wrapper; model, data, random seed, checks and tolerances are unchanged.",
        "baseline": "V1 stopped before loss/backward/check publication with AttributeError: torch.flatnonzero.",
        "fallback": "Any remaining program or science failure stops before training; no silent repair.",
        "user_authorization": {"status": "APPROVED", "approved_by": "user-standing-authorization", "approved_at": "2026-08-29T00:00:00+08:00", "authorized_gates": [3], "authorized_operations": ["audit"], "scope": "One immutable V1R readiness system corrective.", "confirmation_reference": "User delegated best-method decisions and continuous execution."},
        "acceptance_criteria": [
            "Exact V1 data/model/random seed/checks/tolerances and exact V1 FAIL seal binding.",
            "flatnonzero compatibility regression returns exact indices and rejects non-vectors.",
            "Exact 188126 observations, 240101 parameters, causal references, permutation <=1e-6, masked/repeat zero, transport <=1e-6 and finite backward.",
            "Zero optimizer/trained inference/threshold selection/graph/C09/C10/M-TARE; source unchanged and complete seal."
        ],
        "expected_counts": {"worlds": 80, "observations": 188126, "fit_observations": 142184, "selection_observations": 45942, "real_readiness_rows": 8, "history_observations": 5, "perception_seeds": 3, "tokens_per_seed": 6, "parameters": 240101, "optimizer_steps": 0, "model_inference_frames": 0, "threshold_selection_steps": 0, "graph_replays": 0, "c09_worlds_read": 0, "c10_worlds_read": 0, "mtare_worlds_read": 0},
        "expected_evidence": ["V1 FAIL binding and exact system corrective.", "Full causal/interface/numerical readiness evidence.", "CSV, PNG/PDF/SVG/source, log, environment, RUN_STATE and seal."],
        "estimated_cost": {"compute": "CPU full causal audit and eight-row forward/backward", "wall_time_hours": 0.05, "host_ram_gb": 4, "gpu_memory_gb": 0, "disk_gb": 0.1, "gpu": "none"},
        "frozen_inputs": {path: sha256(PROJECT_ROOT / path) for path in sorted(inputs)},
        "frozen_tools": {name: {"path": path, "sha256": sha256(PROJECT_ROOT / path)} for name, path in tools.items()},
        "working_directory": str(PROJECT_ROOT),
        "command": ["/usr/bin/timeout", "900s", PYTHON, "tools/v3/run_gse_relational_exit_transport_readiness_v1r.py", "--spec", str(SPEC), "--run-dir", str(PROJECT_ROOT / f"results/gate3_semantics/{RUN_ID}")],
    }
    write_json(SPEC, spec)
    print(SPEC.relative_to(PROJECT_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
