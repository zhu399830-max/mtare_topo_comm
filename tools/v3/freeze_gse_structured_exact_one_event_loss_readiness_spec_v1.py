#!/usr/bin/env python3
"""Freeze one immutable structured exact-one loss readiness run."""

from __future__ import annotations

import hashlib

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import write_json


RUN_ID = "gate3_20260829_gse_structured_exact_one_event_loss_readiness_v1_seed0"
SPEC = PROJECT_ROOT / "configs/v3/gate3/gse_structured_exact_one_event_loss_readiness_v1.json"
DATA_CARD = "configs/v3/gate3/data_cards/gse_structured_exact_one_event_loss_readiness_v1.json"
PYTHON = "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    if SPEC.exists():
        raise RuntimeError("exact-one readiness spec already exists")
    action = "results/gate3_semantics/gate3_20260828_gse_action_set_node_training_v1_seed0"
    cache = f"{action}/scratch/action_set_cache"
    unified = "results/gate3_semantics/gate3_20260827_gse_factorized_association_capacity_v1r_seed0"
    training = "results/gate3_semantics/gate3_20260829_gse_relational_exit_transport_training_v1_seed0"
    commit = "results/gate3_semantics/gate3_20260829_gse_relational_commit_policy_feasibility_v1_seed0"
    inputs = [
        DATA_CARD,
        f"{action}/RUN_STATE.json", f"{action}/metrics/summary.json", f"{action}/artifacts/evidence_sha256.txt",
        f"{unified}/RUN_STATE.json", f"{unified}/metrics/summary.json", f"{unified}/artifacts/evidence_sha256.txt",
        f"{training}/RUN_STATE.json", f"{training}/metrics/summary.json", f"{training}/artifacts/evidence_sha256.txt",
        f"{commit}/RUN_STATE.json", f"{commit}/metrics/summary.json", f"{commit}/artifacts/evidence_sha256.txt",
    ]
    inputs.extend(f"{cache}/{name}.npy" for name in (
        "raw_tokens", "history_references", "history_mask", "normalization_mean", "normalization_scale",
        "decision_target", "decision_episode_id", "partition_code",
    ))
    inputs.extend(f"{unified}/artifacts/unified_observation/seed{seed}_unified_observation_features.npy" for seed in range(3))
    tools = {
        "model": "src/mtare_topo/representation/gse_relational_exit_transport_event.py",
        "loss": "src/mtare_topo/representation/gse_structured_exact_one_event.py",
        "dataset": "src/mtare_topo/data/gse_relational_exit_transport_cache.py",
        "sampler": "src/mtare_topo/data/gse_causal_episode_sampler.py",
        "executor": "tools/v3/execute_gse_structured_exact_one_event_loss_readiness_v1.py",
        "runner": "tools/v3/run_gse_structured_exact_one_event_loss_readiness_v1.py",
        "freezer": "tools/v3/freeze_gse_structured_exact_one_event_loss_readiness_spec_v1.py",
        "tests": "tests/v3/unit/test_gse_structured_exact_one_event.py",
        "governance": "src/mtare_topo/governance.py",
        "preflight": "tools/v3/preflight.py",
        "create_run": "tools/v3/create_run.py",
    }
    spec = {
        "schema_version": "v3_run_spec_v1", "gate": 3, "execution_phase": 3,
        "date": "20260829", "slug": "gse_structured_exact_one_event_loss_readiness_v1", "seed": 0,
        "operation": "audit", "data_card": DATA_CARD,
        "question": "Does the parameter-free exact-one likelihood directly penalize duplicate commit peaks while remaining invariant, stable and differentiable on complete real episodes?",
        "method": "Compare synthetic zero/one/two/wrong peaks, old/new second-peak gradients, episode permutations, extreme logits and one deterministic complete real batch with the unchanged relational model.",
        "baseline": "V1 max-MIL has exactly zero gradient on a non-maximum duplicate positive peak; the state-machine audit restores safety but misses the C07 F1 gain gate.",
        "fallback": "Any semantic, gradient, numerical or real-batch failure stops V2 before training.",
        "user_authorization": {"status": "APPROVED", "approved_by": "user-standing-authorization", "approved_at": "2026-08-29T00:35:00+08:00", "authorized_gates": [3], "authorized_operations": ["audit"], "scope": "One immutable zero-training exact-one loss readiness.", "confirmation_reference": "User explicitly delegated best-method choices and continuous execution."},
        "acceptance_criteria": [
            "Exact 188126 observations, 142184/45942 split, unchanged 240101 parameters and one complete 128-row real batch.",
            "One correct peak beats zero/two/wrong peaks; old duplicate gradient=0 and exact-one duplicate gradient>0.001.",
            "Episode permutation error<=1e-6; extreme and real losses/gradients finite.",
            "Zero optimizer/trained inference/checkpoint/threshold/C09/C10/M-TARE/graph/planner; source unchanged and complete evidence."
        ],
        "expected_counts": {"worlds": 80, "observations": 188126, "fit_observations": 142184, "selection_observations": 45942, "real_readiness_rows": 128, "parameters": 240101, "optimizer_steps": 0, "trained_model_inference_frames": 0, "graph_replays": 0, "c09_worlds_read": 0, "c10_worlds_read": 0, "mtare_worlds_read": 0},
        "expected_evidence": ["Synthetic loss ordering and old/new duplicate gradient attribution.", "Permutation/extreme-numeric and complete-real-batch finite backward evidence.", "PNG/PDF/SVG/source, raw log, environment, RUN_STATE and seal."],
        "estimated_cost": {"compute": "CPU full contract plus one untrained 128-row forward/backward", "wall_time_hours": 0.02, "host_ram_gb": 4, "gpu_memory_gb": 0, "disk_gb": 0.1, "gpu": "none"},
        "frozen_inputs": {path: sha256(PROJECT_ROOT / path) for path in sorted(inputs)},
        "frozen_tools": {name: {"path": path, "sha256": sha256(PROJECT_ROOT / path)} for name, path in tools.items()},
        "working_directory": str(PROJECT_ROOT),
        "command": ["/usr/bin/timeout", "600s", PYTHON, "tools/v3/run_gse_structured_exact_one_event_loss_readiness_v1.py", "--spec", str(SPEC), "--run-dir", str(PROJECT_ROOT / f"results/gate3_semantics/{RUN_ID}")],
    }
    write_json(SPEC, spec)
    print(SPEC.relative_to(PROJECT_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
