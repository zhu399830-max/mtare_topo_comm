#!/usr/bin/env python3
"""Freeze the approved Data Card and formal spatial event-center run spec."""

from __future__ import annotations

import hashlib

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import write_json


RUN_ID = "gate3_20260828_gse_spatial_event_center_training_v1_seed0"
CARD_PATH = PROJECT_ROOT / "configs/v3/gate3/data_cards/gse_spatial_event_center_training_v1.json"
SPEC_PATH = PROJECT_ROOT / "configs/v3/gate3/gse_spatial_event_center_training_v1.json"
PYTHON = "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python"
DATASET = "results/gate2_representation/gate2_20260824_gse_deduplicated_dataset_export_v1_seed0"
SPATIAL_MODELS = "results/gate2_representation/gate2_20260824_gse_graph_three_seed_training_v1r_seed0"
CAUSAL_TEACHER = "results/gate2_representation/gate2_20260827_gse_corrected_causal_teacher_manifest_v1r_seed0"
ACTION = "results/gate3_semantics/gate3_20260828_gse_action_set_node_training_v1_seed0"
CENTER = "results/gate3_semantics/gate3_20260828_gse_event_center_offset_training_v1r3_seed0"
SCALAR = "results/gate3_semantics/gate3_20260828_gse_event_center_pair_consistency_training_v1r3_seed0"
PROJECTION = "results/gate3_semantics/gate3_20260828_gse_projected_trace_commit_capacity_v1_seed0"
VECTOR_V2 = "results/gate3_semantics/gate3_20260828_gse_event_center_vector_training_v2_seed0"


def _sha(path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    approval = {
        "status": "APPROVED",
        "approved_by": "user-standing-authorization",
        "approved_at": "2026-08-28T11:00:00+08:00",
        "authorized_gates": [3],
        "authorized_operations": ["training"],
        "scope": "One immutable C01-C08 three-seed five-frame height-aware spatial event-center training; zero C09/C10/M-TARE.",
        "confirmation_reference": "User instructed automatic best-choice execution without routine approval prompts.",
    }
    card = {
        "schema_version": "v3_data_card_v1",
        "card_id": "gse_spatial_event_center_training_v1",
        "title": "Five-frame spatial event-center learning from circular LiDAR features",
        "status": "APPROVED_FOR_ONE_IMMUTABLE_GSE_SPATIAL_EVENT_CENTER_TRAINING_V1",
        "operation": "training",
        "purpose": "Test whether retaining robot-frame azimuth and elevation layout lets LiDAR localize junction/terminal centers across distinct incident traversals.",
        "approval": approval,
        "source": {
            "dataset_run": DATASET,
            "spatial_encoder_run": SPATIAL_MODELS,
            "causal_teacher_run": CAUSAL_TEACHER,
            "action_set_run": ACTION,
            "event_center_teacher_run": CENTER,
            "scalar_initializer_run": SCALAR,
            "baseline_projection_run": PROJECTION,
            "failed_vector_ablation_run": VECTOR_V2,
            "raw_sources": [
                "Sealed C01-C08 deduplicated organized LiDAR shards.",
                "Three sealed GSE encoders, action-set models and scalar event-center checkpoints.",
                "Sealed objective event-center Teacher and corrected causal five-frame references.",
                "No C09/C10/strict/M-TARE sensor, Teacher, model, graph or planner artifact.",
            ],
            "license_or_allowed_use": "Local research use of project-generated procedural Cano data and locally trained checkpoints with sealed provenance.",
        },
        "worlds": {
            "train": ["S01-S10_C01-C06"],
            "validation": ["S01-S10_C07-C08"],
            "strict_test": ["S01-S10_C09", "S01-S10_C10", "SEALED_MTARE_FINAL_WORLDS"],
            "checkpoint_selection": ["S01-S10_C07-C08"],
            "fit": "S01-S10 C01-C06: 60 worlds",
            "selection_description": "S01-S10 C07-C08: 20 worlds",
            "forbidden": "C09, C10, strict test and all M-TARE worlds",
            "count": 80,
        },
        "trajectories": [
            {"id": "C01-C06_all_directed_traversals", "world": "S01-S10_C01-C06", "split": "train", "world_count": 60, "directed_traversal_count": 12106, "duration_s": 178494, "distance_m": 184552.021499521, "spatial_coverage_m": 92276.010749762, "independent": True},
            {"id": "C07-C08_all_directed_traversals", "world": "S01-S10_C07-C08", "split": "validation", "world_count": 20, "directed_traversal_count": 3972, "duration_s": 57858, "distance_m": 59821.90390958, "spatial_coverage_m": 29910.951954792, "independent": True},
        ],
        "sampling": {
            "raw_frame_count": 252430,
            "effective_sample_count": 34133,
            "effective_structure_event_count": 1066,
            "raw_unique_lidar_frames": 252430,
            "raw_causal_observations": 188126,
            "effective_decision_rows": 34133,
            "fit_rows": 25294,
            "selection_rows": 8839,
            "fit_node_identities": 792,
            "selection_node_identities": 274,
            "fit_multi_traversal_identities": 782,
            "selection_multi_traversal_identities": 272,
            "fit_cross_traversal_positive_pairs": 417435,
            "selection_cross_traversal_positive_pairs": 144533,
            "junction_rows": 26608,
            "terminal_rows": 7525,
            "spatial_interval_m": 1.0,
            "temporal_context": "Exactly the current and four earlier one-metre LiDAR frames from the same directed traversal.",
            "independent_sampling_unit": "Objective TNG node identity across distinct directed traversals, with world-case split before fitting.",
            "rule": "At one-metre arc spacing, each decision sample uses the current and four strictly past frames from one traversal; direct batches are event-balanced and pair batches use distinct traversals of one node identity.",
            "structure_event_counts": {"junction_rows": 26608, "terminal_rows": 7525, "decision_node_identities": 1066},
        },
        "split": {
            "fit": "C01-C06 updates only the new 525698-parameter spatial decoder.",
            "checkpoint_and_threshold_selection": "C07-C08 selects one checkpoint per seed; ensemble gates are frozen before any C09 read.",
            "strict_test": "C09/C10/M-TARE remain unread.",
            "world_disjoint": True,
            "trajectory_disjoint": True,
            "historical_pollution_audit": "No C09/C10/M-TARE sensor, target, feature, checkpoint, threshold or normalization statistic is read. Earlier C09 diagnostics are evidence only and contribute no input to this operation.",
        },
        "teacher": {
            "source": "Objective TNG node center minus sensor position in the executed route basis: forward, horizontal-left and gravity-consistent up.",
            "valid_mask": "Only typed junction and terminal rows enter regression; identity is used only to sample cross-traversal positive pairs and evaluate consistency.",
            "student_input": "Only frozen-encoder pooled [128], azimuth [128,36] and elevation [128,2] features from five causal LiDAR frames, plus the frozen scalar longitudinal prediction.",
            "planner_consistency_plan": "At runtime route tangent and gravity define the same basis. Identity/objective center are never inputs; predicted centers may propose nodes, while edges still require physical traversal.",
        },
        "leakage_audit": {
            "future_frames_excluded": True,
            "gt_identity_excluded_at_inference": True,
            "objective_center_excluded_at_inference": True,
            "pose_excluded_from_spatial_decoder": True,
            "test_excluded_from_supervised_training": True,
            "test_excluded_from_ssl": True,
            "test_excluded_from_normalization": True,
            "test_excluded_from_teacher_calibration": True,
            "test_excluded_from_threshold_calibration": True,
            "test_excluded_from_augmentation_tuning": True,
            "test_excluded_from_checkpoint_selection": True,
        },
        "methods": {
            "main": "For each seed, freeze the original GSE encoder and cache separable 36-bin azimuth plus 2-bin elevation features. Circular convolutions and explicit sine/cosine moments preserve signed bearing; vertical moments preserve height. Freeze the qualified scalar forward offset and train only lateral/up with equal-event direct loss plus cross-traversal global-center consistency.",
            "baseline": "Sealed scalar V1R3 projection and failed pooled-context vector V2. V2 improved relative error 12.3643% but within-4m coverage only 8.9455 points because lateral/up stayed at the zero-output baseline.",
            "fallback": "None inside the run. Scientific failure stops event-center localization and graph replay; no metric, split, radius or planner tuning is allowed.",
        },
        "acceptance": {
            "relative_vector_error": "C07-C08 ensemble improves sealed scalar projection by at least 10%.",
            "association_range_coverage": "Identity-macro within-4m fraction improves by at least 0.10.",
            "longitudinal": "Frozen ensemble forward MAE does not regress.",
            "transverse": "Both lateral and vertical MAE improve their zero-output baselines.",
            "global_center": "3D Euclidean center MAE improves scalar projection.",
            "seed_stability": "All three seeds improve relative error.",
            "system": "Zero encoder/action/scalar optimizer steps, zero C09/C10/M-TARE reads, source unchanged and complete seal.",
        },
        "estimated_cost": {
            "compute": "One RTX 5090 D; three seeds sequentially",
            "wall_time_hours": 3.0,
            "gpu_memory_gb": 12,
            "host_ram_gb": 16,
            "scratch_disk_gb": 3.0,
            "retained_disk_gb": 1.0,
            "disk_gb": 3.0,
        },
        "retention": "Retain three decoder checkpoints, histories, selection outputs, ensemble metrics, per-seed cache manifests/digests, deletion audit, logs and seal. Delete only each deterministic 2.526 GB temporary cache after its seed completes.",
        "failure_policy": "Any source/split/reference drift, forbidden read, nonfinite value, frozen-module update, environment/resource violation, program error or unmet scientific gate seals FAIL. No retry or gate relaxation.",
    }
    write_json(CARD_PATH, card)

    inputs = [
        f"{DATASET}/RUN_STATE.json", f"{DATASET}/metrics/summary.json",
        f"{DATASET}/artifacts/evidence_sha256.txt", f"{DATASET}/artifacts/shard_manifest.json",
        f"{SPATIAL_MODELS}/RUN_STATE.json", f"{SPATIAL_MODELS}/metrics/summary.json",
        f"{SPATIAL_MODELS}/artifacts/evidence_sha256.txt",
        f"{CAUSAL_TEACHER}/RUN_STATE.json", f"{CAUSAL_TEACHER}/metrics/summary.json",
        f"{CAUSAL_TEACHER}/artifacts/evidence_sha256.txt", f"{CAUSAL_TEACHER}/artifacts/teacher_observations.jsonl",
        f"{ACTION}/RUN_STATE.json", f"{ACTION}/metrics/summary.json", f"{ACTION}/artifacts/evidence_sha256.txt",
        f"{ACTION}/scratch/action_set_cache/manifest.json",
        f"{CENTER}/RUN_STATE.json", f"{CENTER}/metrics/summary.json", f"{CENTER}/artifacts/evidence_sha256.txt",
        f"{CENTER}/artifacts/teacher/event_center_teacher.npz",
        f"{SCALAR}/RUN_STATE.json", f"{SCALAR}/metrics/summary.json", f"{SCALAR}/artifacts/evidence_sha256.txt",
        f"{PROJECTION}/RUN_STATE.json", f"{PROJECTION}/metrics/summary.json", f"{PROJECTION}/artifacts/evidence_sha256.txt",
        f"{PROJECTION}/artifacts/projection/event_center_projection.npz",
        f"{VECTOR_V2}/RUN_STATE.json", f"{VECTOR_V2}/metrics/summary.json", f"{VECTOR_V2}/artifacts/evidence_sha256.txt",
    ]
    action_cache = PROJECT_ROOT / ACTION / "scratch/action_set_cache"
    inputs.extend(str(path.relative_to(PROJECT_ROOT)) for path in sorted(action_cache.glob("*.npy")))
    for seed in (0, 1, 2):
        inputs.extend([
            f"{SPATIAL_MODELS}/artifacts/models/seed{seed}/best.pt",
            f"{ACTION}/artifacts/models/seed{seed}/best.pt",
            f"{SCALAR}/artifacts/models/seed{seed}/best.pt",
        ])
    tools = {
        "data_card": str(CARD_PATH.relative_to(PROJECT_ROOT)),
        "spatial_encoder": "src/mtare_topo/representation/gse_graph.py",
        "spatial_feature_projection": "src/mtare_topo/representation/gse_causal_episode_detector.py",
        "spatial_decoder": "src/mtare_topo/representation/gse_spatial_event_center.py",
        "spatial_cache": "src/mtare_topo/data/gse_spatial_event_center_cache.py",
        "action_cache": "src/mtare_topo/data/gse_action_set_cache.py",
        "action_model": "src/mtare_topo/representation/gse_action_set_node.py",
        "scalar_model": "src/mtare_topo/representation/gse_event_center_offset.py",
        "center_teacher": "src/mtare_topo/teacher/gse_event_center_teacher.py",
        "cache_builder": "tools/v3/build_gse_spatial_event_center_cache_v1.py",
        "trainer": "tools/v3/train_gse_spatial_event_center_v1.py",
        "evaluator": "tools/v3/evaluate_gse_spatial_event_center_ensemble_v1.py",
        "runner": "tools/v3/run_gse_spatial_event_center_training_v1.py",
        "freezer": "tools/v3/freeze_gse_spatial_event_center_training_spec_v1.py",
        "preflight": "tools/v3/preflight.py",
        "create_run": "tools/v3/create_run.py",
    }
    spec = {
        "schema_version": "v3_run_spec_v1",
        "gate": 3, "execution_phase": 3, "date": "20260828",
        "slug": "gse_spatial_event_center_training_v1", "seed": 0,
        "operation": "training",
        "question": "Can five-frame circular LiDAR features localize junction/terminal centers laterally and vertically enough to pass the frozen cross-view graph-association gates?",
        "method": card["methods"]["main"],
        "baseline": card["methods"]["baseline"],
        "fallback": card["methods"]["fallback"],
        "data_card": str(CARD_PATH.relative_to(PROJECT_ROOT)),
        "config_path": str(CARD_PATH.relative_to(PROJECT_ROOT)),
        "user_authorization": approval,
        "hyperparameters": {
            "history_frames": 5, "directional_bins": 36, "elevation_bins": 2,
            "epochs": 10, "seeds": [0, 1, 2], "learning_rate": 0.0005,
            "weight_decay": 0.0001, "identities_per_event": 16,
            "direct_to_relative_loss_weight": "1:1", "gradient_clip": 5.0,
            "cache_batch_size": 128, "evaluation_batch_size": 128,
        },
        "expected_counts": {
            "worlds": 80, "unique_frames_per_seed": 252430,
            "causal_observations": 188126, "fit_rows": 25294,
            "selection_rows": 8839, "optimizer_steps_per_seed": 3960,
            "backbone_optimizer_steps": 0, "c09_worlds_read": 0,
            "c10_worlds_read": 0, "mtare_worlds_read": 0,
            "temporary_cache_bytes_per_seed": 2526043720,
        },
        "acceptance_criteria": list(card["acceptance"].values()),
        "estimated_cost": card["estimated_cost"],
        "expected_evidence": [
            "Three cache manifests/deletion records, checkpoints, histories and selection outputs.",
            "Per-seed and ensemble 3D center, cross-view and component metrics.",
            "Environment, command, logs, source integrity, RUN_STATE and SHA-256 seal.",
        ],
        "frozen_inputs": {relative: _sha(PROJECT_ROOT / relative) for relative in inputs},
        "frozen_tools": {
            name: {"path": relative, "sha256": _sha(PROJECT_ROOT / relative)}
            for name, relative in tools.items()
        },
        "command": [
            "/usr/bin/timeout", "--signal=INT", "--kill-after=60s", "14400s",
            PYTHON, "tools/v3/run_gse_spatial_event_center_training_v1.py",
            "--spec", str(SPEC_PATH),
            "--run-dir", str(PROJECT_ROOT / "results/gate3_semantics" / RUN_ID),
        ],
        "working_directory": str(PROJECT_ROOT),
    }
    write_json(SPEC_PATH, spec)
    print(SPEC_PATH.relative_to(PROJECT_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
