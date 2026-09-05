#!/usr/bin/env python3
"""Freeze the one C01--C08 GSE trace-commit capacity run."""

from __future__ import annotations

import hashlib

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import write_json


RUN_ID = "gate3_20260828_gse_trace_commit_capacity_v1_seed0"
SPEC = PROJECT_ROOT / "configs/v3/gate3/gse_trace_commit_capacity_v1.json"
PYTHON = "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python"


def _sha(path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    if SPEC.exists():
        raise RuntimeError("trace-commit capacity spec exists; overwrite is forbidden")
    teacher = "results/gate2_representation/gate2_20260827_gse_corrected_causal_teacher_manifest_v1r_seed0"
    dataset = "results/gate2_representation/gate2_20260824_gse_deduplicated_dataset_export_v1_seed0"
    source = "results/gate3_semantics/gate3_20260826_gse_exit_token_association_corrective_v2_seed0"
    capacity = "results/gate3_semantics/gate3_20260827_gse_factorized_association_capacity_v1r_seed0"
    action = "results/gate3_semantics/gate3_20260828_gse_action_set_node_training_v1_seed0"
    inputs = [
        f"{teacher}/artifacts/teacher_observations.jsonl",
        f"{dataset}/artifacts/sequence_manifest.jsonl",
        f"{source}/artifacts/pair_cache/pairs.npz",
        f"{action}/scratch/action_set_cache/manifest.json",
    ]
    inputs += [f"{action}/scratch/action_set_cache/{name}.npy" for name in (
        "raw_tokens", "history_references", "history_mask", "normalization_mean",
        "normalization_scale", "decision_target", "decision_episode_id", "partition_code",
    )]
    for seed in range(3):
        inputs += [
            f"{action}/artifacts/models/seed{seed}/best.pt",
            f"{source}/artifacts/models/seed{seed}/frozen_exit_token_outputs.npz",
            f"{capacity}/artifacts/unified_observation/seed{seed}_unified_observation_features.npy",
            f"{capacity}/artifacts/models/seed{seed}/full_route_conditioned/best.pt",
            f"{capacity}/artifacts/models/seed{seed}/full_route_conditioned/normalization.npz",
        ]
    tools = {
        "trace_graph": "src/mtare_topo/topology/gse_trace_commit_replay.py",
        "action_model": "src/mtare_topo/representation/gse_action_set_node.py",
        "action_cache": "src/mtare_topo/data/gse_action_set_cache.py",
        "association": "src/mtare_topo/representation/gse_factorized_association.py",
        "structured_rule": "src/mtare_topo/representation/gse_route_conditioned_node.py",
        "episode_metrics": "src/mtare_topo/evaluation/gse_causal_episode_metrics.py",
        "executor": "tools/v3/execute_gse_trace_commit_capacity_v1.py",
        "runner": "tools/v3/run_gse_trace_commit_capacity_v1.py",
        "freezer": "tools/v3/freeze_gse_trace_commit_capacity_spec_v1.py",
        "data_card": "configs/v3/gate3/data_cards/gse_trace_commit_capacity_v1.json",
        "preflight": "tools/v3/preflight.py",
        "create_run": "tools/v3/create_run.py",
    }
    run_dir = PROJECT_ROOT / f"results/gate3_semantics/{RUN_ID}"
    spec = {
        "schema_version": "v3_run_spec_v1", "gate": 3, "execution_phase": 3,
        "date": "20260828", "slug": "gse_trace_commit_capacity_v1", "seed": 0,
        "operation": "audit",
        "question": "Can learned geometry-semantic hypotheses improve a trace-verified topometric graph on disjoint C07-C08 development worlds?",
        "method": "A frozen three-seed action-set head proposes junction/terminal hypotheses. A frozen 2-of-3 route-conditioned association model with a 4 m cap may group exactly one prior hypothesis. Two distinct physical traversals are required to commit a node; a verified edge requires one completed traversal containing two committed endpoints.",
        "baseline": "The same learned proposals committed immediately (NTS-style ghost ablation), and the fixed route-conditioned token-count rule with the same trace commit.",
        "fallback": "None inside the run. Scientific failure stops this commit mechanism; no threshold, identity, test world or planner adaptation is allowed.",
        "data_card": "configs/v3/gate3/data_cards/gse_trace_commit_capacity_v1.json",
        "config_path": "configs/v3/gate3/data_cards/gse_trace_commit_capacity_v1.json",
        "user_authorization": {
            "status": "APPROVED", "approved_by": "user-standing-authorization",
            "approved_at": "2026-08-28T02:30:00+08:00", "authorized_gates": [3],
            "authorized_operations": ["audit"],
            "scope": "One immutable zero-training C01-C08 trace-commit graph capacity run; zero C09/C10/M-TARE.",
            "confirmation_reference": "User instructed automatic best-choice execution without routine approval prompts."
        },
        "hyperparameters": {
            "proposal_threshold_grid": list((.80, .85, .90, .925, .95, .97, .98, .99, .995)),
            "association_votes_required": 2, "association_distance_cap_m": 4.0,
            "independent_traversals_to_commit": 2,
            "structured_rule": {"confidence": .28, "incoming_half_angle_deg": 50, "persistence": 3, "seed_consensus": 2},
        },
        "acceptance_criteria": [
            "C01-C06 must contain a configuration with node precision >=0.98, false loop merge <=0.01, node recall >=0.25, edge precision >=0.98 and edge recall >=0.25.",
            "The once-selected C07-C08 result must independently satisfy the same node/edge safety and recall gates.",
            "C07-C08 node/edge macro-F1 must exceed both immediate learned-ghost and structured-rule trace-commit baselines by at least 0.05.",
            "Runtime candidate generation and commit decisions never consume Teacher identity; zero optimizer/model updates and zero C09/C10/M-TARE reads."
        ],
        "expected_counts": {
            "worlds": 80, "causal_observations": 188126, "fit_observations": 142184,
            "selection_observations": 45942, "directed_traversals": 16076,
            "fit_semantic_nodes": 792, "selection_semantic_nodes": 274,
            "fit_observed_semantic_trace_relations": 36,
            "selection_observed_semantic_trace_relations": 13,
            "optimizer_steps": 0, "model_updates": 0,
            "c09_worlds_read": 0, "c10_worlds_read": 0, "mtare_worlds_read": 0
        },
        "estimated_cost": {
            "compute": "one GPU inference of three frozen lightweight heads, then CPU association/replay",
            "wall_time_hours": 1.0, "gpu_memory_gb": 2, "host_ram_gb": 6, "disk_gb": 0.5
        },
        "expected_evidence": [
            "Fit threshold grid, frozen C07-C08 main/baseline metrics, raw proposal metrics, association pair scores, verified nodes/edges, decision trace, logs, source integrity and SHA-256 seal."
        ],
        "frozen_inputs": {relative: _sha(PROJECT_ROOT / relative) for relative in inputs},
        "frozen_tools": {name: {"path": relative, "sha256": _sha(PROJECT_ROOT / relative)} for name, relative in tools.items()},
        "command": [
            "/usr/bin/timeout", "--signal=INT", "--kill-after=30s", "3600s", PYTHON,
            "tools/v3/run_gse_trace_commit_capacity_v1.py", "--spec", str(SPEC),
            "--run-dir", str(run_dir),
        ],
        "working_directory": str(PROJECT_ROOT),
    }
    write_json(SPEC, spec)
    print({"spec": str(SPEC), "frozen_inputs": len(inputs), "frozen_tools": len(tools)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
