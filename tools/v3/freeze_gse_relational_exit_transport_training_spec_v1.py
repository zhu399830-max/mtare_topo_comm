#!/usr/bin/env python3
"""Freeze one immutable three-seed relational transport capacity run."""

from __future__ import annotations

import hashlib

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import write_json


RUN_ID = "gate3_20260829_gse_relational_exit_transport_training_v1_seed0"
SPEC = PROJECT_ROOT / "configs/v3/gate3/gse_relational_exit_transport_training_v1.json"
DATA_CARD = "configs/v3/gate3/data_cards/gse_relational_exit_transport_training_v1.json"
PYTHON = "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    if SPEC.exists():
        raise RuntimeError("relational transport training spec already exists")
    action = "results/gate3_semantics/gate3_20260828_gse_action_set_node_training_v1_seed0"
    cache = f"{action}/scratch/action_set_cache"
    unified = "results/gate3_semantics/gate3_20260827_gse_factorized_association_capacity_v1r_seed0"
    readiness = "results/gate3_semantics/gate3_20260829_gse_relational_exit_transport_readiness_v1r_seed0"
    feasibility = "results/gate3_semantics/gate3_20260828_gse_exit_action_transport_feasibility_v1_seed0"
    baseline = "results/gate3_semantics/gate3_20260828_gse_action_set_node_selection_corrective_v1_seed0"
    inputs = [
        DATA_CARD,
        f"{action}/RUN_STATE.json", f"{action}/metrics/summary.json", f"{action}/artifacts/evidence_sha256.txt", f"{cache}/manifest.json",
        f"{unified}/RUN_STATE.json", f"{unified}/metrics/summary.json", f"{unified}/artifacts/evidence_sha256.txt",
        f"{readiness}/RUN_STATE.json", f"{readiness}/metrics/summary.json", f"{readiness}/artifacts/evidence_sha256.txt",
        f"{feasibility}/RUN_STATE.json", f"{feasibility}/metrics/summary.json", f"{feasibility}/artifacts/evidence_sha256.txt",
        f"{baseline}/RUN_STATE.json", f"{baseline}/metrics/summary.json", f"{baseline}/artifacts/evidence_sha256.txt",
    ]
    inputs.extend(f"{cache}/{name}.npy" for name in (
        "raw_tokens", "history_references", "history_mask", "normalization_mean", "normalization_scale",
        "decision_target", "decision_episode_id", "partition_code", "traversal_id", "sequence_index",
        "identity", "parent_id",
    ))
    for seed in range(3):
        inputs.append(f"{unified}/artifacts/unified_observation/seed{seed}_unified_observation_features.npy")
    tools = {
        "model": "src/mtare_topo/representation/gse_relational_exit_transport_event.py",
        "dataset": "src/mtare_topo/data/gse_relational_exit_transport_cache.py",
        "sampler": "src/mtare_topo/data/gse_causal_episode_sampler.py",
        "trainer": "tools/v3/train_gse_relational_exit_transport_event_v1.py",
        "selection_evaluator": "tools/v3/evaluate_gse_relational_exit_transport_selection_v1.py",
        "selection_evaluator_base": "tools/v3/evaluate_gse_action_set_node_selection_v1.py",
        "runner": "tools/v3/run_gse_relational_exit_transport_training_v1.py",
        "freezer": "tools/v3/freeze_gse_relational_exit_transport_training_spec_v1.py",
        "tests_model": "tests/v3/unit/test_gse_relational_exit_transport_event.py",
        "tests_training": "tests/v3/unit/test_gse_relational_exit_transport_training.py",
        "tests_sampler": "tests/v3/unit/test_gse_causal_episode_sampler.py",
        "governance": "src/mtare_topo/governance.py",
        "preflight": "tools/v3/preflight.py",
        "create_run": "tools/v3/create_run.py",
    }
    spec = {
        "schema_version": "v3_run_spec_v1", "gate": 3, "execution_phase": 3,
        "date": "20260829", "slug": "gse_relational_exit_transport_training_v1", "seed": 0,
        "operation": "training", "data_card": DATA_CARD,
        "question": "Can explicit causal exit-token transport learn safe junction/terminal node events with at least five macro-F1 points over the old pooled-token model?",
        "method": "Train three independent 240101-parameter relational transport event models on C01-C06 with episode-preserving MIL, select checkpoints and one fixed threshold on C07-C08 only, and average seed probabilities.",
        "baseline": "Sealed pooled-token ActionSetNodeDetector: decision episode macro-F1 0.743847, trigger precision 0.965870 and episode recall 0.498239.",
        "fallback": "If gain, safety or seed stability fails, seal scientific FAIL and stop this event-model route before any strict-test or graph experiment.",
        "user_authorization": {"status": "APPROVED", "approved_by": "user-standing-authorization", "approved_at": "2026-08-29T00:01:00+08:00", "authorized_gates": [3], "authorized_operations": ["training"], "scope": "One immutable C01-C06/C07-C08 three-seed relational event capacity run.", "confirmation_reference": "User explicitly requested continuous autonomous execution and delegated best-method choices."},
        "acceptance_criteria": [
            "Exactly 142184 fit and 45942 selection observations, 3282/1136 structural episodes and three seeds.",
            "Exactly 6666 optimizer steps per seed and 19998 total; perception outputs receive zero optimizer steps.",
            "Ensemble trigger precision >=0.995, false fraction <=0.005, episode recall >=0.25 and per-event precision/recall >=0.99/0.25.",
            "Decision episode macro-F1 >=0.793847, every seed precision >=0.98 and recall >=0.25, and all ten families have a correct trigger.",
            "Zero C09/C10/M-TARE/graph/planner reads; exact source replay, complete checkpoints, predictions, figures, logs and seal."
        ],
        "expected_counts": {"worlds": 80, "fit_worlds": 60, "selection_worlds": 20, "observations": 188126, "fit_observations": 142184, "selection_observations": 45942, "fit_decision_episodes": 3282, "selection_decision_episodes": 1136, "perception_seeds": 3, "model_seeds": 3, "parameters_per_seed": 240101, "epochs_per_seed": 6, "batches_per_epoch": 1111, "optimizer_steps_per_seed": 6666, "optimizer_steps": 19998, "perception_backbone_optimizer_steps": 0, "graph_replays": 0, "c09_worlds_read": 0, "c10_worlds_read": 0, "mtare_worlds_read": 0},
        "expected_evidence": ["Three checkpoints, histories and row-aligned selection outputs.", "Frozen ensemble threshold, aggregate/per-event/per-family/seed metrics and baseline gain.", "PNG/PDF/SVG paper figure and source, raw logs, environment, RUN_STATE and SHA-256 seal."],
        "estimated_cost": {"compute": "Three serial deterministic CUDA trainings plus one CPU selection evaluation", "wall_time_hours": 2.0, "host_ram_gb": 8, "gpu_memory_gb": 8, "disk_gb": 1, "gpu": "one RTX 5090 D"},
        "frozen_inputs": {path: sha256(PROJECT_ROOT / path) for path in sorted(inputs)},
        "frozen_tools": {name: {"path": path, "sha256": sha256(PROJECT_ROOT / path)} for name, path in tools.items()},
        "working_directory": str(PROJECT_ROOT),
        "command": ["/usr/bin/timeout", "43200s", PYTHON, "tools/v3/run_gse_relational_exit_transport_training_v1.py", "--spec", str(SPEC), "--run-dir", str(PROJECT_ROOT / f"results/gate3_semantics/{RUN_ID}")],
    }
    write_json(SPEC, spec)
    print(SPEC.relative_to(PROJECT_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
