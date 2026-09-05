#!/usr/bin/env python3
"""Freeze one immutable relational commit-policy feasibility audit."""

from __future__ import annotations

import hashlib

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import write_json


RUN_ID = "gate3_20260829_gse_relational_commit_policy_feasibility_v1_seed0"
SPEC = PROJECT_ROOT / "configs/v3/gate3/gse_relational_commit_policy_feasibility_v1.json"
DATA_CARD = "configs/v3/gate3/data_cards/gse_relational_commit_policy_feasibility_v1.json"
PYTHON = "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    if SPEC.exists():
        raise RuntimeError("commit-policy feasibility spec already exists")
    action = "results/gate3_semantics/gate3_20260828_gse_action_set_node_training_v1_seed0"
    cache = f"{action}/scratch/action_set_cache"
    training = "results/gate3_semantics/gate3_20260829_gse_relational_exit_transport_training_v1_seed0"
    inputs = [
        DATA_CARD,
        f"{action}/RUN_STATE.json", f"{action}/metrics/summary.json", f"{action}/artifacts/evidence_sha256.txt",
        f"{training}/RUN_STATE.json", f"{training}/metrics/summary.json", f"{training}/artifacts/evidence_sha256.txt",
    ]
    inputs.extend(f"{cache}/{name}.npy" for name in (
        "partition_code", "decision_target", "decision_episode_id", "traversal_id", "sequence_index", "parent_id",
    ))
    inputs.extend(f"{training}/artifacts/models/seed{seed}/selection_outputs.npz" for seed in range(3))
    tools = {
        "policy": "src/mtare_topo/evaluation/gse_relational_commit_policy.py",
        "trigger_metrics": "src/mtare_topo/evaluation/gse_causal_episode_metrics.py",
        "executor": "tools/v3/execute_gse_relational_commit_policy_feasibility_v1.py",
        "runner": "tools/v3/run_gse_relational_commit_policy_feasibility_v1.py",
        "freezer": "tools/v3/freeze_gse_relational_commit_policy_feasibility_spec_v1.py",
        "tests": "tests/v3/unit/test_gse_relational_commit_policy.py",
        "governance": "src/mtare_topo/governance.py",
        "preflight": "tools/v3/preflight.py",
        "create_run": "tools/v3/create_run.py",
    }
    spec = {
        "schema_version": "v3_run_spec_v1", "gate": 3, "execution_phase": 3,
        "date": "20260829", "slug": "gse_relational_commit_policy_feasibility_v1", "seed": 0,
        "operation": "audit", "data_card": DATA_CARD,
        "question": "Can a fixed past-only stable-trigger, duplicate-suppression and seed-consensus refusal state machine rescue the sealed relational outputs without retraining?",
        "method": "Enumerate exactly 336 per-event causal policies on C07, select joint safety then macro-F1, and apply the selected policy once unchanged to C08.",
        "baseline": "Sealed stateless ensemble precision/recall/macro-F1 0.946612/0.405810/0.664556 and pooled-token macro-F1 0.743847.",
        "fallback": "If either split misses original safety or +0.05 gain, reject a deterministic state-machine rescue and require structured one-commit learning or stop the route.",
        "user_authorization": {"status": "APPROVED", "approved_by": "user-standing-authorization", "approved_at": "2026-08-29T00:20:00+08:00", "authorized_gates": [3], "authorized_operations": ["audit"], "scope": "One immutable zero-training C07/C08 frozen-output commit-policy audit.", "confirmation_reference": "User explicitly delegated best-method choices and continuous execution."},
        "acceptance_criteria": [
            "Exact 21548 C07/24394 C08 observations and 533/603 decision episodes.",
            "Exactly 336 policies per event; C07-only selection and one unchanged C08 application.",
            "Both splits independently achieve aggregate precision>=0.995, false<=0.005, recall>=0.25 and per-event precision/recall>=0.99/0.25.",
            "Both splits independently achieve macro-F1>=0.793847, five points over the exact pooled baseline.",
            "Exact stateless attribution 461 correct/18 duplicate/5 corridor false/3 wrong event; zero training/new inference/C09/C10/M-TARE/graph/planner."
        ],
        "expected_counts": {"worlds": 20, "c07_worlds": 10, "c08_worlds": 10, "selection_observations": 45942, "c07_observations": 21548, "c08_observations": 24394, "c07_decision_episodes": 533, "c08_decision_episodes": 603, "policies_per_event": 336, "optimizer_steps": 0, "model_inference_frames": 0, "graph_replays": 0, "c09_worlds_read": 0, "c10_worlds_read": 0, "mtare_worlds_read": 0},
        "expected_evidence": ["Full candidate grid and selected typed policy.", "C07 selection/C08 transfer metrics and exact stateless failure attribution.", "PNG/PDF/SVG/source, raw log, environment, RUN_STATE and seal."],
        "estimated_cost": {"compute": "CPU-only finite policy replay", "wall_time_hours": 0.05, "host_ram_gb": 2, "gpu_memory_gb": 0, "disk_gb": 0.1, "gpu": "none"},
        "frozen_inputs": {path: sha256(PROJECT_ROOT / path) for path in sorted(inputs)},
        "frozen_tools": {name: {"path": path, "sha256": sha256(PROJECT_ROOT / path)} for name, path in tools.items()},
        "working_directory": str(PROJECT_ROOT),
        "command": ["/usr/bin/timeout", "900s", PYTHON, "tools/v3/run_gse_relational_commit_policy_feasibility_v1.py", "--spec", str(SPEC), "--run-dir", str(PROJECT_ROOT / f"results/gate3_semantics/{RUN_ID}")],
    }
    write_json(SPEC, spec)
    print(SPEC.relative_to(PROJECT_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
