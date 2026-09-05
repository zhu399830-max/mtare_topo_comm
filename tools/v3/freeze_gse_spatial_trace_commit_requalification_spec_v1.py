#!/usr/bin/env python3
"""Freeze the single spatial-center trace-commit graph requalification."""

from __future__ import annotations

import hashlib
from pathlib import Path

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, write_json


RUN_ID = "gate3_20260828_gse_spatial_trace_commit_requalification_v1_seed0"
SPEC = PROJECT_ROOT / "configs/v3/gate3/gse_spatial_trace_commit_requalification_v1.json"
CARD = PROJECT_ROOT / "configs/v3/gate3/data_cards/gse_spatial_trace_commit_requalification_v1.json"
PYTHON = "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    if SPEC.exists() or CARD.exists():
        raise RuntimeError("spatial trace-commit card/spec exists; overwrite is forbidden")
    teacher = "results/gate2_representation/gate2_20260827_gse_corrected_causal_teacher_manifest_v1r_seed0"
    dataset = "results/gate2_representation/gate2_20260824_gse_deduplicated_dataset_export_v1_seed0"
    gse = "results/gate2_representation/gate2_20260824_gse_graph_three_seed_training_v1r_seed0"
    source = "results/gate3_semantics/gate3_20260826_gse_exit_token_association_corrective_v2_seed0"
    capacity = "results/gate3_semantics/gate3_20260827_gse_factorized_association_capacity_v1r_seed0"
    action = "results/gate3_semantics/gate3_20260828_gse_action_set_node_training_v1_seed0"
    center = "results/gate3_semantics/gate3_20260828_gse_event_center_offset_training_v1r3_seed0"
    scalar = "results/gate3_semantics/gate3_20260828_gse_event_center_pair_consistency_training_v1r3_seed0"
    spatial = "results/gate3_semantics/gate3_20260828_gse_spatial_event_center_training_v1_seed0"
    corrective = "results/gate3_semantics/gate3_20260828_gse_spatial_longitudinal_corrective_v1_seed0"
    predecessor = "results/gate3_semantics/gate3_20260828_gse_projected_trace_commit_capacity_v1_seed0"
    approval = {
        "status": "APPROVED", "approved_by": "user-standing-authorization",
        "approved_at": "2026-08-28T12:10:00+08:00", "authorized_gates": [3],
        "authorized_operations": ["audit"],
        "scope": "One immutable zero-training C01-C08 spatial-center trace-commit requalification; zero C09/C10/M-TARE.",
        "confirmation_reference": "User instructed automatic best-choice execution without routine approval prompts.",
    }
    card = load_json(PROJECT_ROOT / "configs/v3/gate3/data_cards/gse_projected_trace_commit_capacity_v1.json")
    card.update({
        "card_id": "gse_spatial_trace_commit_requalification_v1",
        "title": "Qualified spatial-center trace-commit graph requalification",
        "status": "APPROVED_FOR_ONE_IMMUTABLE_GSE_SPATIAL_TRACE_COMMIT_REQUALIFICATION_V1",
        "operation": "zero-training graph replay and audit",
        "purpose": "Test whether the qualified five-frame forward/left/up event center reduces duplicate junction and terminal nodes and recovers trace-verified edges under the unchanged graph contract.",
        "student_inputs": [
            "three frozen causal action-set model contexts",
            "three frozen five-frame spatial-longitudinal event-center predictions",
            "runtime route tangent and sensor pose",
            "three frozen learned association models and learned exit tokens",
            "executed traversal identity and causal within-world order",
        ],
        "split_logic": "The predecessor proposal threshold 0.97 is fixed before this run. C01-C06 and C07-C08 are replayed only for capacity and disjoint validation evidence; no threshold, 4 m cap, association vote, commit rule, model or Teacher changes are allowed.",
        "methods": {
            "main": "Use the arithmetic three-seed five-frame 3D structure center, then apply the frozen factorized association at the unchanged 4 m cap and require two independent traversals before node commit; edges require a completed physical trace.",
            "baselines": "Sealed scalar-projected trace-commit predecessor, immediate learned-ghost commit and fixed structured-rule trace commit.",
            "fallback": "None. Failure stops the current event-center graph route; no threshold, radius or planner tuning is allowed.",
        },
        "retention": "Retain three all-row seed projections, ensemble projection, qualification reproduction proof, graph nodes/edges/decision trace, pair scores, metrics, logs and seal. Delete each 2.526 GB deterministic spatial cache after use.",
        "approval": approval,
    })
    card["leakage_audit"].update({
        "objective_center_at_runtime": False,
        "event_valid_mask_at_runtime": False,
        "strict_test_worlds_read": 0,
    })
    card["metrics_and_pre_registered_gates"].update({
        "fixed_proposal_threshold": "Proposal threshold remains the predecessor value 0.97; no fit or validation reselection.",
        "center_reproduction": "All 8,839 C07-C08 event rows must reproduce the qualified seed and ensemble outputs bit-exactly.",
    })
    write_json(CARD, card)

    inputs = [
        f"{teacher}/RUN_STATE.json", f"{teacher}/metrics/summary.json", f"{teacher}/artifacts/evidence_sha256.txt", f"{teacher}/artifacts/teacher_observations.jsonl",
        f"{dataset}/RUN_STATE.json", f"{dataset}/metrics/summary.json", f"{dataset}/artifacts/evidence_sha256.txt", f"{dataset}/artifacts/sequence_manifest.jsonl",
        f"{source}/RUN_STATE.json", f"{source}/metrics/summary.json", f"{source}/artifacts/evidence_sha256.txt", f"{source}/artifacts/pair_cache/pairs.npz",
        f"{capacity}/RUN_STATE.json", f"{capacity}/metrics/summary.json", f"{capacity}/artifacts/evidence_sha256.txt",
        f"{action}/RUN_STATE.json", f"{action}/metrics/summary.json", f"{action}/artifacts/evidence_sha256.txt", f"{action}/scratch/action_set_cache/manifest.json",
        f"{center}/RUN_STATE.json", f"{center}/metrics/summary.json", f"{center}/artifacts/evidence_sha256.txt", f"{center}/artifacts/teacher/event_center_teacher.npz",
        f"{scalar}/RUN_STATE.json", f"{scalar}/metrics/summary.json", f"{scalar}/artifacts/evidence_sha256.txt",
        f"{spatial}/RUN_STATE.json", f"{spatial}/metrics/summary.json", f"{spatial}/artifacts/evidence_sha256.txt",
        f"{corrective}/RUN_STATE.json", f"{corrective}/metrics/summary.json", f"{corrective}/artifacts/evidence_sha256.txt", f"{corrective}/metrics/ensemble/ensemble_selection_outputs.npz",
        f"{predecessor}/RUN_STATE.json", f"{predecessor}/metrics/summary.json", f"{predecessor}/artifacts/evidence_sha256.txt", f"{predecessor}/artifacts/replay/summary.json",
    ]
    action_cache = PROJECT_ROOT / action / "scratch/action_set_cache"
    inputs.extend(str(path.relative_to(PROJECT_ROOT)) for path in sorted(action_cache.glob("*.npy")))
    for seed in range(3):
        inputs.extend([
            f"{gse}/artifacts/models/seed{seed}/best.pt",
            f"{action}/artifacts/models/seed{seed}/best.pt",
            f"{scalar}/artifacts/models/seed{seed}/best.pt",
            f"{spatial}/artifacts/models/seed{seed}/best.pt",
            f"{spatial}/artifacts/cache_manifests/seed{seed}_manifest.json",
            f"{corrective}/artifacts/models/seed{seed}/best.pt",
            f"{corrective}/artifacts/models/seed{seed}/selection_outputs.npz",
            f"{source}/artifacts/models/seed{seed}/frozen_exit_token_outputs.npz",
            f"{capacity}/artifacts/unified_observation/seed{seed}_unified_observation_features.npy",
            f"{capacity}/artifacts/models/seed{seed}/full_route_conditioned/best.pt",
            f"{capacity}/artifacts/models/seed{seed}/full_route_conditioned/normalization.npz",
        ])
    tools = {
        "data_card": str(CARD.relative_to(PROJECT_ROOT)),
        "spatial_encoder": "src/mtare_topo/representation/gse_graph.py",
        "spatial_projection": "src/mtare_topo/representation/gse_causal_episode_detector.py",
        "spatial_decoder": "src/mtare_topo/representation/gse_spatial_event_center.py",
        "spatial_cache": "src/mtare_topo/data/gse_spatial_event_center_cache.py",
        "center_projection": "src/mtare_topo/evaluation/gse_spatial_center_projection.py",
        "trace_graph": "src/mtare_topo/topology/gse_trace_commit_replay.py",
        "action_model": "src/mtare_topo/representation/gse_action_set_node.py",
        "action_cache": "src/mtare_topo/data/gse_action_set_cache.py",
        "scalar_model": "src/mtare_topo/representation/gse_event_center_offset.py",
        "center_teacher": "src/mtare_topo/teacher/gse_event_center_teacher.py",
        "association": "src/mtare_topo/representation/gse_factorized_association.py",
        "structured_rule": "src/mtare_topo/representation/gse_route_conditioned_node.py",
        "episode_metrics": "src/mtare_topo/evaluation/gse_causal_episode_metrics.py",
        "cache_builder": "tools/v3/build_gse_spatial_event_center_cache_v1.py",
        "center_inference": "tools/v3/infer_gse_spatial_longitudinal_center_all_v1.py",
        "center_combiner": "tools/v3/combine_gse_spatial_longitudinal_center_v1.py",
        "replay_executor": "tools/v3/execute_gse_trace_commit_capacity_v1.py",
        "runner": "tools/v3/run_gse_spatial_trace_commit_requalification_v1.py",
        "freezer": "tools/v3/freeze_gse_spatial_trace_commit_requalification_spec_v1.py",
        "preflight": "tools/v3/preflight.py", "create_run": "tools/v3/create_run.py",
    }
    run_dir = PROJECT_ROOT / "results/gate3_semantics" / RUN_ID
    spec = {
        "schema_version": "v3_run_spec_v1", "gate": 3, "execution_phase": 3,
        "date": "20260828", "slug": "gse_spatial_trace_commit_requalification_v1", "seed": 0,
        "operation": "audit",
        "question": "Does the qualified five-frame 3D event center reduce duplicate nodes and recover execution-verified topology under the unchanged trace-commit contract?",
        "method": card["methods"]["main"], "baseline": card["methods"]["baselines"], "fallback": card["methods"]["fallback"],
        "data_card": str(CARD.relative_to(PROJECT_ROOT)), "config_path": str(CARD.relative_to(PROJECT_ROOT)),
        "user_authorization": approval,
        "hyperparameters": {
            "fixed_proposal_threshold": 0.97,
            "center_ensemble": "arithmetic_mean_seeds_0_1_2",
            "association_votes_required": 2,
            "association_distance_cap_m": 4.0,
            "independent_traversals_to_commit": 2,
            "cache_batch_size": 128,
            "inference_batch_size": 128,
        },
        "acceptance_criteria": list(card["metrics_and_pre_registered_gates"].values()),
        "expected_counts": {
            "worlds": 80, "causal_observations": 188126,
            "fit_observations": 142184, "selection_observations": 45942,
            "selection_event_rows_reproduced": 8839,
            "directed_traversals": 16076, "fit_semantic_nodes": 792,
            "selection_semantic_nodes": 274, "fit_observed_semantic_trace_relations": 36,
            "selection_observed_semantic_trace_relations": 13,
            "optimizer_steps": 0, "model_updates": 0,
            "c09_worlds_read": 0, "c10_worlds_read": 0, "mtare_worlds_read": 0,
            "temporary_cache_bytes_per_seed": 2526043720,
        },
        "estimated_cost": {
            "compute": "three sequential frozen GPU feature/inference passes followed by CPU association/replay",
            "wall_time_hours": 2.0, "gpu_memory_gb": 3, "host_ram_gb": 8,
            "scratch_disk_gb": 3.0, "retained_disk_gb": 1.0, "disk_gb": 3.0,
        },
        "expected_evidence": [
            "Three exact cache reproduction manifests/deletion records and three all-row seed predictions.",
            "Bit-exact reproduction of all 8,839 qualified C07-C08 event outputs and an all-row ensemble projection.",
            "Fixed-threshold fit/selection graph metrics, pair scores, nodes, edges, decision trace, logs, source integrity and seal.",
        ],
        "frozen_inputs": {relative: _sha(PROJECT_ROOT / relative) for relative in inputs},
        "frozen_tools": {name: {"path": relative, "sha256": _sha(PROJECT_ROOT / relative)} for name, relative in tools.items()},
        "command": [
            "/usr/bin/timeout", "--signal=INT", "--kill-after=60s", "7200s", PYTHON,
            "tools/v3/run_gse_spatial_trace_commit_requalification_v1.py",
            "--spec", str(SPEC), "--run-dir", str(run_dir),
        ],
        "working_directory": str(PROJECT_ROOT),
    }
    write_json(SPEC, spec)
    print(SPEC.relative_to(PROJECT_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
