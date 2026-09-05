#!/usr/bin/env python3
"""Freeze one immutable structured exact-one three-seed capacity run."""

from __future__ import annotations

import hashlib

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import write_json


RUN_ID = "gate3_20260829_gse_structured_exact_one_event_training_v1_seed0"
SPEC = PROJECT_ROOT / "configs/v3/gate3/gse_structured_exact_one_event_training_v1.json"
DATA_CARD = "configs/v3/gate3/data_cards/gse_structured_exact_one_event_training_v1.json"
PYTHON = "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    if SPEC.exists():
        raise RuntimeError("structured exact-one training spec already exists")
    action = "results/gate3_semantics/gate3_20260828_gse_action_set_node_training_v1_seed0"
    cache = f"{action}/scratch/action_set_cache"
    unified = "results/gate3_semantics/gate3_20260827_gse_factorized_association_capacity_v1r_seed0"
    v1 = "results/gate3_semantics/gate3_20260829_gse_relational_exit_transport_training_v1_seed0"
    commit = "results/gate3_semantics/gate3_20260829_gse_relational_commit_policy_feasibility_v1_seed0"
    readiness = "results/gate3_semantics/gate3_20260829_gse_structured_exact_one_event_loss_readiness_v1_seed0"
    inputs = [
        DATA_CARD,
        f"{action}/RUN_STATE.json", f"{action}/metrics/summary.json", f"{action}/artifacts/evidence_sha256.txt", f"{cache}/manifest.json",
        f"{unified}/RUN_STATE.json", f"{unified}/metrics/summary.json", f"{unified}/artifacts/evidence_sha256.txt",
        f"{v1}/RUN_STATE.json", f"{v1}/metrics/summary.json", f"{v1}/artifacts/evidence_sha256.txt",
        f"{commit}/RUN_STATE.json", f"{commit}/metrics/summary.json", f"{commit}/artifacts/evidence_sha256.txt",
        f"{readiness}/RUN_STATE.json", f"{readiness}/metrics/summary.json", f"{readiness}/artifacts/evidence_sha256.txt",
    ]
    inputs.extend(f"{cache}/{name}.npy" for name in (
        "raw_tokens", "history_references", "history_mask", "normalization_mean", "normalization_scale",
        "decision_target", "decision_episode_id", "partition_code", "traversal_id", "sequence_index", "parent_id",
    ))
    inputs.extend(f"{unified}/artifacts/unified_observation/seed{seed}_unified_observation_features.npy" for seed in range(3))
    tools = {
        "model": "src/mtare_topo/representation/gse_relational_exit_transport_event.py",
        "loss": "src/mtare_topo/representation/gse_structured_exact_one_event.py",
        "dataset": "src/mtare_topo/data/gse_relational_exit_transport_cache.py",
        "sampler": "src/mtare_topo/data/gse_causal_episode_sampler.py",
        "commit_policy": "src/mtare_topo/evaluation/gse_relational_commit_policy.py",
        "trigger_metrics": "src/mtare_topo/evaluation/gse_causal_episode_metrics.py",
        "trainer": "tools/v3/train_gse_structured_exact_one_event_v1.py",
        "evaluator": "tools/v3/evaluate_gse_structured_exact_one_event_selection_v1.py",
        "runner": "tools/v3/run_gse_structured_exact_one_event_training_v1.py",
        "freezer": "tools/v3/freeze_gse_structured_exact_one_event_training_spec_v1.py",
        "tests_loss": "tests/v3/unit/test_gse_structured_exact_one_event.py",
        "tests_training": "tests/v3/unit/test_gse_structured_exact_one_event_training.py",
        "tests_policy": "tests/v3/unit/test_gse_relational_commit_policy.py",
        "governance": "src/mtare_topo/governance.py",
        "preflight": "tools/v3/preflight.py",
        "create_run": "tools/v3/create_run.py",
    }
    spec = {
        "schema_version": "v3_run_spec_v1", "gate": 3, "execution_phase": 3,
        "date": "20260829", "slug": "gse_structured_exact_one_event_training_v1", "seed": 0,
        "operation": "training", "data_card": DATA_CARD,
        "question": "Can exact-one likelihood make the unchanged relational model satisfy safe structural-node commit and macro-F1 gates on C07 and transfer unchanged to C08?",
        "method": "Train three unchanged models on C01-C06 for six epochs with exact-one likelihood; C07 alone selects checkpoint and one finite-grid causal commit policy; apply once to C08.",
        "baseline": "V1 max-MIL stateful C07/C08 precision 0.996241/0.997093, recall 0.497186/0.568823 and macro-F1 0.753519/0.796339; pooled macro-F1 0.743847.",
        "fallback": "Any C07/C08 safety, F1, family, split or system failure stops V2 before C09 or graph.",
        "user_authorization": {"status": "APPROVED", "approved_by": "user-standing-authorization", "approved_at": "2026-08-29T00:45:00+08:00", "authorized_gates": [3], "authorized_operations": ["training"], "scope": "One immutable C01-C06/C07/C08 three-seed exact-one capacity run.", "confirmation_reference": "User explicitly delegated best-method choices and continuous execution."},
        "acceptance_criteria": [
            "Exactly 142184 fit, 21548 C07 and 24394 C08 observations and 3282/533/603 decision episodes.",
            "Exactly 6666 steps/seed and 19998 total; unchanged 240101 parameters; zero perception updates and zero C08 checkpoint reads.",
            "C07 and C08 each achieve aggregate precision>=0.995, false<=0.005, recall>=0.25, per-event precision/recall>=0.99/0.25 and all ten families correct.",
            "C07 and C08 each achieve macro-F1>=0.793847; one C07-selected policy is transferred unchanged.",
            "Zero C09/C10/M-TARE/graph/planner; source unchanged and complete checkpoints/grid/figures/logs/seal."
        ],
        "expected_counts": {"worlds": 80, "fit_observations": 142184, "c07_observations": 21548, "c08_observations": 24394, "fit_decision_episodes": 3282, "c07_decision_episodes": 533, "c08_decision_episodes": 603, "model_seeds": 3, "parameters_per_seed": 240101, "epochs_per_seed": 6, "batches_per_epoch": 1111, "optimizer_steps_per_seed": 6666, "optimizer_steps": 19998, "perception_backbone_optimizer_steps": 0, "c08_checkpoint_observations": 0, "graph_replays": 0, "c09_worlds_read": 0, "c10_worlds_read": 0, "mtare_worlds_read": 0},
        "expected_evidence": ["Three checkpoints, histories and row-aligned C07-C08 outputs.", "C07-selected typed policy, full grid, unchanged C08 transfer and stateless/stateful metrics.", "PNG/PDF/SVG/source, raw logs, environment, RUN_STATE and seal."],
        "estimated_cost": {"compute": "Three serial deterministic CUDA trainings plus CPU policy evaluation", "wall_time_hours": 0.3, "host_ram_gb": 8, "gpu_memory_gb": 8, "disk_gb": 1, "gpu": "one RTX 5090 D"},
        "frozen_inputs": {path: sha256(PROJECT_ROOT / path) for path in sorted(inputs)},
        "frozen_tools": {name: {"path": path, "sha256": sha256(PROJECT_ROOT / path)} for name, path in tools.items()},
        "working_directory": str(PROJECT_ROOT),
        "command": ["/usr/bin/timeout", "43200s", PYTHON, "tools/v3/run_gse_structured_exact_one_event_training_v1.py", "--spec", str(SPEC), "--run-dir", str(PROJECT_ROOT / f"results/gate3_semantics/{RUN_ID}")],
    }
    write_json(SPEC, spec)
    print(SPEC.relative_to(PROJECT_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
