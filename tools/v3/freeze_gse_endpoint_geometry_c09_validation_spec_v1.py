#!/usr/bin/env python3
"""Freeze the Data Card and run spec for C09 endpoint-geometry validation."""

from __future__ import annotations

import hashlib

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, write_json


RUN_ID = "gate3_20260828_gse_endpoint_geometry_c09_validation_v1_seed0"
CARD = PROJECT_ROOT / "configs/v3/gate3/data_cards/gse_endpoint_geometry_c09_validation_v1.json"
SPEC = PROJECT_ROOT / "configs/v3/gate3/gse_endpoint_geometry_c09_validation_v1.json"
PYTHON = "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python"


def _sha(path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    dataset = "results/gate2_representation/gate2_20260824_gse_deduplicated_dataset_export_v1_seed0"
    teacher = "results/gate2_representation/gate2_20260824_gse_teacher_manifest_v1_seed0"
    training = "results/gate2_representation/gate2_20260824_gse_graph_three_seed_training_v1r_seed0"
    action = "results/gate3_semantics/gate3_20260828_gse_action_set_node_training_v1_seed0"
    scalar = "results/gate3_semantics/gate3_20260828_gse_event_center_pair_consistency_training_v1r3_seed0"
    spatial = "results/gate3_semantics/gate3_20260828_gse_spatial_event_center_training_v1_seed0"
    corrective = "results/gate3_semantics/gate3_20260828_gse_spatial_longitudinal_corrective_v1_seed0"
    association = "results/gate3_semantics/gate3_20260827_gse_factorized_association_c09_qualification_v1_seed0"
    consensus = "results/gate3_semantics/gate3_20260828_gse_factorized_consensus_metric_corrective_v1_seed0"
    development = "results/gate3_semantics/gate3_20260828_gse_endpoint_geometry_capacity_v1_seed0"
    parents = "results/gate0_baseline/gate0_20260811_cano_100_topology_parent_recipe_reclassification_v2r_seed0"
    parent_source = "results/gate0_baseline/gate0_20260811_cano_100_topology_parent_candidate_audit_v2_bounded_resampling_seed0"
    card = {
        "schema_version": "v3_data_card_v1",
        "card_id": "gse_endpoint_geometry_c09_validation_v1",
        "status": "APPROVED_FOR_ONE_IMMUTABLE_GSE_ENDPOINT_GEOMETRY_C09_VALIDATION_V1",
        "purpose": "Validate the frozen GSE semantic-event, spatial-center and execution-endpoint graph mechanism on all C09 worlds without training or method adaptation.",
        "approval": {
            "status": "APPROVED", "approved_by": "user-standing-authorization",
            "approved_at": "2026-08-28T00:00:00+08:00", "authorized_gates": [3],
            "authorized_operations": ["audit"],
            "scope": "One immutable C09 validation run. Process 1 freezes the graph without Teacher identity; process 2 performs posthoc objective scoring. No threshold selection, updates, C10 or M-TARE.",
            "confirmation_reference": "User instructed the agent to choose the best in-scope option automatically and not request routine approvals.",
        },
        "source": {
            "dataset_run": dataset, "teacher_run": teacher,
            "frozen_gse_training_run": training, "action_model_run": action,
            "scalar_center_run": scalar, "spatial_center_run": spatial,
            "longitudinal_corrective_run": corrective,
            "c09_association_archive": association, "consensus_correction_run": consensus,
            "development_capacity_run": development, "objective_parent_manifest": parents,
            "license_or_allowed_use": "Local research use of project-generated procedural worlds and locally trained models.",
        },
        "worlds": [
            "S01_flat_tree_small_C09", "S02_3d_tree_small_C09",
            "S03_flat_unicyclic_small_C09", "S04_3d_unicyclic_small_C09",
            "S05_flat_branch_medium_C09", "S06_3d_branch_medium_C09",
            "S07_flat_loop_rich_C09", "S08_3d_loop_rich_C09",
            "S09_flat_complex_C09", "S10_3d_complex_C09",
        ],
        "trajectories": {
            "directed_traversals": 2054, "physical_edges": 1027,
            "rule": "Every directed traversal from all ten C09 procedural topology parents; no deletion or resampling.",
        },
        "sampling": {
            "raw_frame_count": 32678, "effective_sample_count": 24462,
            "independent_units": "Ten C09 procedural topology worlds; nodes and observed executed relations are reporting units.",
            "spatial_interval_m": 1.0, "temporal_context": "Exactly five causal LiDAR frames per observation.",
            "validation_structure_identities": {"junction": 71, "terminal": 59, "total": 130},
            "historical_scope": "C09 has historical perception/association validation exposure, but the endpoint-geometry mechanism and its fixed contracts were frozen on C01-C08 before this run.",
        },
        "split": {
            "model_fit": "C01-C06 only in sealed upstream runs.",
            "method_development": "C07-C08 participated in mechanism confirmation; no numeric grid was selected for endpoint geometry.",
            "validation": "Exactly C09, all ten worlds and all observations.",
            "strict_test": "C10 remains unread and is forbidden in this run.",
            "leakage_audit": "Teacher/event identity and objective TNG coordinates are absent from process 1. Graph files are hashed before process 2 reads Teacher. Runtime sensor pose, executed traversal identity and route geometry are allowed execution state.",
        },
        "teacher": {
            "source": "Sealed C09 TNG identity/event labels and TNG node coordinates, opened only posthoc.",
            "objective_relation": "Consecutive observed junction/terminal identities along each completed directed traversal.",
            "student_forbidden_fields": ["Teacher identity", "Teacher event", "objective node coordinate", "future frame", "C10"],
        },
        "methods": {
            "main": "Frozen three-seed LiDAR action-event model and spatial center decoder; old learned association OR unanimous three-seed 4m metric support; nodes require two executed traversals; post-commit endpoint anchors must agree within 2m; junctions require three incident edges or a two-edge positive outward dot-product branch witness; edges require completed physical traversal.",
            "baseline": "The sealed C07-C08 development endpoint-geometry score is plotted beside C09; no parameter is reselected.",
            "fallback": "On any safety or recall failure, seal FAIL and stop before C10. Do not tune thresholds, delete worlds or substitute a graph.",
        },
        "metrics_and_pre_registered_gates": {
            "node": "Objective 3D TNG center matching within 4m: precision>=0.98 and recall>=0.25.",
            "edge": "Observed executed semantic relations: precision>=0.98 and recall>=0.25.",
            "loop": "False loop merge fraction<=0.01.",
            "population": "10 worlds, 32678 unique frames, 24462 observations, 2054 directed traversals and 130 objective identities.",
            "forbidden": "Zero optimizer/model/checkpoint/threshold selection steps; zero C10 and M-TARE reads.",
        },
        "estimated_cost": {
            "compute": "Three deterministic frozen CUDA passes over 32678 LiDAR frames plus small-head inference and CPU graph scoring.",
            "wall_time_hours": 1.0, "host_ram_gb": 10, "gpu_memory_gb": 12,
            "disk_gb": 3, "gpu": "one RTX 5090 D",
        },
        "retention": "Retain frozen graph nodes/edges/decision trace, spatial/action outputs, endpoint qualification, posthoc metrics, PNG/PDF/SVG/source JSON, logs, environment, commands, RUN_STATE and SHA-256 seal. Temporary spatial feature caches are deleted after each seed.",
        "failure_policy": "Any input/tool/environment drift, Teacher access before graph freeze, count/interface/causality/resource failure, safety failure or recall failure seals FAIL; no retry with changed science.",
        "evidence": {
            "machine_metrics": "Objective node/edge precision, recall, F1, false-loop fraction and all fixed gates.",
            "complete_visual_review": "C07-C08 versus C09 node/edge precision and recall in PNG/PDF/SVG with source JSON.",
            "failure_policy": "Fail closed and preserve diagnostic evidence.",
        },
    }
    write_json(CARD, card)

    inputs = [
        f"{dataset}/RUN_STATE.json", f"{dataset}/metrics/summary.json",
        f"{dataset}/artifacts/evidence_sha256.txt", f"{dataset}/artifacts/frame_manifest.jsonl",
        f"{dataset}/artifacts/sequence_manifest.jsonl",
        f"{teacher}/RUN_STATE.json", f"{teacher}/metrics/summary.json",
        f"{teacher}/artifacts/evidence_sha256.txt", f"{teacher}/artifacts/teacher_observations.jsonl",
        f"{teacher}/artifacts/traversal_manifest.jsonl",
        f"{training}/RUN_STATE.json", f"{training}/metrics/summary.json", f"{training}/artifacts/evidence_sha256.txt",
        f"{action}/RUN_STATE.json", f"{action}/metrics/summary.json", f"{action}/artifacts/evidence_sha256.txt",
        f"{action}/artifacts/normalization_mean.npy", f"{action}/artifacts/normalization_scale.npy",
        f"{scalar}/RUN_STATE.json", f"{scalar}/metrics/summary.json", f"{scalar}/artifacts/evidence_sha256.txt",
        f"{spatial}/RUN_STATE.json", f"{spatial}/metrics/summary.json", f"{spatial}/artifacts/evidence_sha256.txt",
        f"{corrective}/RUN_STATE.json", f"{corrective}/metrics/summary.json", f"{corrective}/artifacts/evidence_sha256.txt",
        f"{association}/RUN_STATE.json", f"{association}/artifacts/evidence_sha256.txt",
        f"{association}/artifacts/c09_runtime_candidate_pairs.npz",
        f"{consensus}/RUN_STATE.json", f"{consensus}/artifacts/evidence_sha256.txt",
        f"{consensus}/artifacts/c09_application/c09_consensus_metric_decisions.npz",
        f"{development}/RUN_STATE.json", f"{development}/artifacts/evidence_sha256.txt",
        f"{development}/artifacts/capacity/summary.json",
        f"{parents}/RUN_STATE.json", f"{parents}/artifacts/evidence_sha256.txt",
        f"{parents}/artifacts/accepted_parent_manifest.json",
        f"{parent_source}/RUN_STATE.json", f"{parent_source}/artifacts/evidence_sha256.txt",
    ]
    for seed in range(3):
        inputs.extend([
            f"{training}/artifacts/models/seed{seed}/best.pt",
            f"{training}/artifacts/models/seed{seed}/validation_outputs.npz",
            f"{action}/artifacts/models/seed{seed}/best.pt",
            f"{scalar}/artifacts/models/seed{seed}/best.pt",
            f"{spatial}/artifacts/models/seed{seed}/best.pt",
            f"{corrective}/artifacts/models/seed{seed}/best.pt",
        ])
    parent_manifest = load_json(PROJECT_ROOT / f"{parents}/artifacts/accepted_parent_manifest.json")
    inputs.extend(
        str(row["source_graph"]) for row in parent_manifest["parents"]
        if str(row.get("parent_id", "")).endswith("_C09")
    )
    tools = {
        "data_card": str(CARD.relative_to(PROJECT_ROOT)),
        "inference": "tools/v3/infer_gse_endpoint_geometry_c09_v1.py",
        "evaluation": "tools/v3/evaluate_gse_endpoint_geometry_c09_v1.py",
        "runner": "tools/v3/run_gse_endpoint_geometry_c09_validation_v1.py",
        "freezer": "tools/v3/freeze_gse_endpoint_geometry_c09_validation_spec_v1.py",
        "post_commit_contract": "src/mtare_topo/topology/gse_post_commit_consolidation.py",
        "trace_replay": "src/mtare_topo/topology/gse_trace_commit_replay.py",
        "objective_score": "src/mtare_topo/evaluation/gse_objective_spatial_graph_score.py",
        "action_model": "src/mtare_topo/representation/gse_action_set_node.py",
        "spatial_model": "src/mtare_topo/representation/gse_spatial_event_center.py",
        "tests_endpoint": "tests/v3/unit/test_gse_post_commit_consolidation.py",
        "governance": "src/mtare_topo/governance.py",
        "preflight": "tools/v3/preflight.py", "create_run": "tools/v3/create_run.py",
    }
    criteria = card["metrics_and_pre_registered_gates"]
    spec = {
        "schema_version": "v3_run_spec_v1", "gate": 3, "execution_phase": 3,
        "date": "20260828", "slug": "gse_endpoint_geometry_c09_validation_v1",
        "seed": 0, "operation": "audit",
        "question": "Does the frozen geometry-semantic execution-endpoint graph retain node/edge safety and recall on C09?",
        "method": card["methods"]["main"], "baseline": card["methods"]["baseline"],
        "fallback": card["methods"]["fallback"],
        "data_card": str(CARD.relative_to(PROJECT_ROOT)), "config_path": str(CARD.relative_to(PROJECT_ROOT)),
        "user_authorization": card["approval"],
        "acceptance_criteria": [criteria[key] for key in ("node", "edge", "loop", "population", "forbidden")],
        "expected_counts": {
            "validation_worlds": 10, "unique_lidar_frames": 32678,
            "causal_observations": 24462, "directed_traversals": 2054,
            "physical_edges": 1027, "objective_nodes": 130,
            "model_inference_frames": 98034, "optimizer_steps": 0,
            "model_updates": 0, "threshold_selection_steps": 0,
            "c10_worlds_read": 0, "mtare_worlds_read": 0,
        },
        "expected_evidence": [
            "Teacher-free graph manifest and frozen node/edge/trace hashes; posthoc objective scores; endpoint qualification; action/spatial outputs; PNG/PDF/SVG/source JSON; environment, commands, raw logs, RUN_STATE and SHA-256 seal."
        ],
        "estimated_cost": card["estimated_cost"],
        "frozen_inputs": {path: _sha(PROJECT_ROOT / path) for path in inputs},
        "frozen_tools": {name: {"path": path, "sha256": _sha(PROJECT_ROOT / path)} for name, path in tools.items()},
        "working_directory": str(PROJECT_ROOT),
        "command": [
            "/usr/bin/timeout", "--signal=INT", "--kill-after=30s", "4300s", PYTHON,
            "tools/v3/run_gse_endpoint_geometry_c09_validation_v1.py",
            "--spec", str(SPEC),
            "--run-dir", str(PROJECT_ROOT / "results/gate3_semantics" / RUN_ID),
        ],
    }
    write_json(SPEC, spec)
    print(CARD.relative_to(PROJECT_ROOT)); print(SPEC.relative_to(PROJECT_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
