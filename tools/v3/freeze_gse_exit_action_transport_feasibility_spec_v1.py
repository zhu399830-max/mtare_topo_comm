#!/usr/bin/env python3
"""Freeze the one immutable exit/action-token transport feasibility spec."""

from __future__ import annotations

import hashlib

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import write_json


RUN_ID = "gate3_20260828_gse_exit_action_transport_feasibility_v1_seed0"
SPEC = PROJECT_ROOT / "configs/v3/gate3/gse_exit_action_transport_feasibility_v1.json"
DATA_CARD = "configs/v3/gate3/data_cards/gse_exit_action_transport_feasibility_v1.json"
PYTHON = "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    if SPEC.exists():
        raise RuntimeError("exit/action transport spec already exists")
    dataset = "results/gate2_representation/gate2_20260824_gse_deduplicated_dataset_export_v1_seed0"
    action = "results/gate3_semantics/gate3_20260828_gse_action_set_node_training_v1_seed0"
    cache = f"{action}/scratch/action_set_cache"
    pooled = "results/gate3_semantics/gate3_20260828_gse_action_set_node_selection_corrective_v1_seed0"
    fixed = "results/gate3_semantics/gate3_20260827_gse_action_conditioned_state_feasibility_v1_seed0"
    inputs = [
        DATA_CARD,
        f"{dataset}/RUN_STATE.json", f"{dataset}/metrics/summary.json",
        f"{dataset}/artifacts/shard_manifest.json", f"{dataset}/artifacts/evidence_sha256.txt",
        f"{action}/RUN_STATE.json", f"{action}/metrics/summary.json",
        f"{action}/artifacts/evidence_sha256.txt", f"{cache}/manifest.json",
        f"{pooled}/RUN_STATE.json", f"{pooled}/metrics/summary.json", f"{pooled}/artifacts/evidence_sha256.txt",
        f"{fixed}/RUN_STATE.json", f"{fixed}/metrics/summary.json", f"{fixed}/artifacts/evidence_sha256.txt",
    ]
    inputs.extend(f"{cache}/{name}.npy" for name in (
        "raw_tokens", "partition_code", "decision_target", "decision_episode_id",
        "traversal_id", "sequence_index", "global_sequence_index",
    ))
    tools = {
        "transport_metrics": "src/mtare_topo/evaluation/gse_exit_action_transport.py",
        "executor": "tools/v3/execute_gse_exit_action_transport_feasibility_v1.py",
        "runner": "tools/v3/run_gse_exit_action_transport_feasibility_v1.py",
        "freezer": "tools/v3/freeze_gse_exit_action_transport_feasibility_spec_v1.py",
        "tests": "tests/v3/unit/test_gse_exit_action_transport.py",
        "governance": "src/mtare_topo/governance.py",
        "preflight": "tools/v3/preflight.py",
        "create_run": "tools/v3/create_run.py",
    }
    spec = {
        "schema_version": "v3_run_spec_v1",
        "gate": 3,
        "execution_phase": 3,
        "date": "20260828",
        "slug": "gse_exit_action_transport_feasibility_v1",
        "seed": 0,
        "operation": "audit",
        "data_card": DATA_CARD,
        "question": "Can full executable exit/action tokens retain causal structure and cross-frame physical-exit identity strongly enough to replace pooled action summaries?",
        "method": "Verify every C01-C08 dataset shard, measure an executable visible-exit-set Teacher upper bound, then attach identities after frozen heading/width matching and score adjacent-frame descriptor-only token transport for all three seeds.",
        "baseline": "Sealed fixed five-dimensional change point and sealed per-frame mean/max pooled ActionSetNodeDetector.",
        "fallback": "Stop before training if action-set support or all-seed token transport fails; otherwise implement a relational token-transport event model with explicit objectness/refusal.",
        "user_authorization": {
            "status": "APPROVED",
            "approved_by": "user-standing-authorization",
            "approved_at": "2026-08-28T23:59:00+08:00",
            "authorized_gates": [3],
            "authorized_operations": ["audit"],
            "scope": "One immutable C01-C08 exit/action-token transport feasibility audit.",
            "confirmation_reference": "User delegated best-method decisions and continuous execution.",
        },
        "acceptance_criteria": [
            "Exact 80 worlds, 252430 frames, 188126 observations, 396913 visible tokens, 130080/41970 adjacent pairs and 3282/1136 decision episodes.",
            "C07-C08 executable-set decision macro-F1 and episode support coverage are each at least 0.98.",
            "Every frozen seed has C07-C08 adjacent-frame visible-token identity precision and recall at least 0.98.",
            "All 80 shard tree hashes match and zero optimizer/new inference/threshold selection/graph/C09/C10/M-TARE operations occur.",
        ],
        "expected_counts": {
            "worlds": 80,
            "raw_frames": 252430,
            "observations": 188126,
            "fit_observations": 142184,
            "selection_observations": 45942,
            "visible_exit_tokens": 396913,
            "fit_adjacent_pairs": 130080,
            "selection_adjacent_pairs": 41970,
            "fit_decision_episodes": 3282,
            "selection_decision_episodes": 1136,
            "seed_archives": 3,
            "optimizer_steps": 0,
            "model_inference_frames": 0,
            "threshold_selection_steps": 0,
            "graph_replays": 0,
            "c09_worlds_read": 0,
            "c10_worlds_read": 0,
            "mtare_worlds_read": 0,
        },
        "expected_evidence": [
            "All-shard integrity and exact population evidence.",
            "Teacher action-set upper bound, episode support and per-seed descriptor transport metrics.",
            "Baseline comparison CSV, PNG/PDF/SVG and exact JSON source.",
            "Commands, environment, raw log, RUN_STATE, input/tool hashes and seal.",
        ],
        "estimated_cost": {"compute": "CPU read-only full shard and token audit", "wall_time_hours": 0.1, "host_ram_gb": 4, "gpu_memory_gb": 0, "disk_gb": 0.1, "gpu": "none"},
        "frozen_inputs": {path: sha256(PROJECT_ROOT / path) for path in sorted(inputs)},
        "frozen_tools": {name: {"path": path, "sha256": sha256(PROJECT_ROOT / path)} for name, path in tools.items()},
        "working_directory": str(PROJECT_ROOT),
        "command": [
            "/usr/bin/timeout", "1200s", PYTHON,
            "tools/v3/run_gse_exit_action_transport_feasibility_v1.py",
            "--spec", str(SPEC),
            "--run-dir", str(PROJECT_ROOT / f"results/gate3_semantics/{RUN_ID}"),
        ],
    }
    write_json(SPEC, spec)
    print(SPEC.relative_to(PROJECT_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
