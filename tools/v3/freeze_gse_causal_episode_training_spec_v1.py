#!/usr/bin/env python3
"""Freeze the Data Card and run spec for twelve-frame causal episode training."""

from __future__ import annotations

import hashlib
from copy import deepcopy

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, write_json


RUN_ID = "gate3_20260827_gse_causal_episode_training_v1_seed0"
CARD_PATH = PROJECT_ROOT / "configs/v3/gate3/data_cards/gse_causal_episode_training_v1.json"
SPEC_PATH = PROJECT_ROOT / "configs/v3/gate3/gse_causal_episode_training_v1.json"
CARD_ID = "gse_causal_episode_training_v1"
CARD_STATUS = "APPROVED_FOR_ONE_IMMUTABLE_GSE_CAUSAL_EPISODE_TRAINING_V1"
SLUG = "gse_causal_episode_training_v1"
FREEZER_PATH = "tools/v3/freeze_gse_causal_episode_training_spec_v1.py"
SYSTEM_PREDECESSOR: str | None = None
SOURCE_CARD = PROJECT_ROOT / "configs/v3/gate3/data_cards/gse_corrected_causal_event_training_v1.json"
DATASET = "results/gate2_representation/gate2_20260824_gse_deduplicated_dataset_export_v1_seed0"
TRAINING = "results/gate2_representation/gate2_20260824_gse_graph_three_seed_training_v1r_seed0"
TEACHER = "results/gate2_representation/gate2_20260827_gse_corrected_causal_teacher_manifest_v1r_seed0"
VERIFIER = "results/gate3_semantics/gate3_20260826_gse_exit_token_association_corrective_v2_seed0"
SUPERVISION = "results/gate3_semantics/gate3_20260827_gse_causal_event_supervision_audit_v1_seed0"
REFERENCE = "results/gate3_semantics/gate3_20260827_gse_causal_episode_reference_proof_v1r_seed0"
OLD_DIRECTIONAL = "results/gate3_semantics/gate3_20260826_gse_directional_structural_event_training_v1_seed0"
PYTHON = "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python"


def _sha(path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    approval = {
        "status": "APPROVED",
        "approved_by": "user",
        "approved_at": "2026-08-27T23:50:00+08:00",
        "authorized_operations": ["training"],
        "authorized_gates": [3],
        "scope": "One immutable three-seed twelve-frame past-only causal episode detector: C01-C06 fit, C07-C08 checkpoint and threshold selection, six epochs; frozen spatial encoders and zero C09/C10/strict/M-TARE access.",
        "confirmation_reference": "User instructed Codex to stop requesting routine replies and automatically choose the evidence-supported best in-scope option.",
    }
    card = {
        "schema_version": "v3_data_card_v1",
        "card_id": CARD_ID,
        "status": CARD_STATUS,
        "purpose": "Test whether twelve exact past LiDAR scans and episode-level supervision recover stable geometry-semantic node events that the five-frame frame-classification route missed.",
        "approval": approval,
        "worlds": {
            "development": [f"S{family:02d}_*_C{case:02d}" for family in range(1, 11) for case in range(1, 9)],
            "count": 80,
            "fit": "C01-C06: 60 worlds",
            "checkpoint_and_threshold_selection": "C07-C08: 20 worlds",
            "forbidden": "C09/C10, strict-test and all M-TARE worlds",
        },
        "trajectories": {
            "inventory_directed_traversals": 16078,
            "observed_directed_traversals": 16076,
            "zero_observation_short_traversals": 2,
            "sampling": "One observation per 1 m traversal arc after the existing five-frame warm-up; twelve-frame history is left padded only within the same traversal.",
        },
        "sampling": {
            "independent_units": "Programmatic world family/case split; structural loss bags are 5306 same-traversal, same-identity contiguous episodes.",
            "raw_unique_lidar_frames": 252430,
            "effective_causal_observations": 188126,
            "fit_observations": 142184,
            "selection_observations": 45942,
            "valid_twelve_frame_reference_cells": 1818662,
            "full_twelve_frame_observations": 81486,
            "history_length_counts": {"5": 16076, "6": 16074, "7": 15968, "8": 15648, "9": 15106, "10": 14356, "11": 13412, "12": 81486},
            "event_rows": {"corridor": 150964, "junction": 26608, "terminal": 7525, "turn": 1998, "geometry_transition": 1031},
            "event_episodes": {"junction": 3424, "terminal": 994, "turn": 740, "geometry_transition": 148},
            "boundary_regression": "1030/1031 transition rows have a causal historical boundary offset in [0,10.5] m. The single -0.5 m pre-boundary row remains in event MIL and is explicitly masked only from boundary-distance regression.",
        },
        "split_logic": {
            "fit": "C01-C06 only; every fit observation is visited once per epoch by episode-preserving batches.",
            "selection": "C07-C08 only; selects one checkpoint per seed and one ensemble structural rejection threshold.",
            "future_test": "C09/C10 and M-TARE remain unread until model, threshold and node-commit rule freeze.",
        },
        "teacher": {
            "source": "Sealed corrected causal Teacher V1R: TNG node/degree/tunnel/exit identity plus spline/mesh geometry; teacher fields are targets only.",
            "student_input": "Only exact past organized LiDAR range/mask scans encoded by the corresponding frozen seed backbone. No pose, identity, topology, future scan, world ID or Teacher geometry enters the model.",
            "episode_supervision": "Corridor frames are hard negatives; each structural episode must contain at least one jointly correct event trigger. Runtime-style contiguous high-confidence responses collapse to one node proposal.",
        },
        "leakage_audit": {
            "world_isolation": "C01-C06 fit and C07-C08 selection are disjoint; no checkpoint or threshold sees C09/C10/strict/M-TARE.",
            "temporal": "All 1,818,662 references are current/past-only, resolve in the same parent shard, and never cross traversal-local frame reset.",
            "teacher_input_separation": "TNG identity/event/boundary are loss/evaluation fields only and are absent from detector inputs.",
            "normalization": "Frozen MAX_RANGE_M=50 m only; no selection or test statistics.",
        },
        "methods": {
            "main": "Per seed, execute the frozen GSE spatial encoder once over 252430 unique scans; retain pooled 128D plus 36-bin circular features. Train a 379014-parameter causal GRU/directional-change residual detector with episode MIL and normalized boundary backprojection loss. Original encoder receives zero optimizer steps.",
            "baseline": "Sealed strongest five-frame directional corrected-label ensemble, macro-F1 0.6879041032 and geometry-transition identity coverage 0/17.",
            "fallback": "None inside this run. Any scientific failure stops detector expansion and triggers a new method review; graph/planner tuning is forbidden.",
        },
        "metrics_and_pre_registered_gates": {
            "frame": "C07-C08 ensemble macro-F1 >=0.7379041032, gain >=0.05; at least 2/3 seeds gain >=0.05 and no seed regresses.",
            "runtime_trigger": "Validation-only nonvacuous contiguous-trigger threshold; structural precision >=0.98, false-trigger fraction <=0.01 and structural episode recall >=0.40.",
            "identity": "Junction/terminal/turn identity coverage >=0.90/0.90/0.40 and geometry-transition correct identities >=7/17.",
            "reporting": "Also report episode macro-F1, boundary MAE, uncertainty and the same runtime-shaped metrics for the old directional baseline; these do not replace the unchanged gates.",
        },
        "estimated_cost": {
            "compute": "One RTX 5090 D; three seeds sequentially",
            "wall_time_hours": 12.0,
            "host_ram_gb": 16,
            "gpu_memory_gb": 16,
            "scratch_disk_gb": 3.0,
            "retained_disk_gb": 2.0,
        },
        "retention": "Retain three detector checkpoints, epoch histories, selection outputs, ensemble metrics, per-seed cache manifests/digests, deletion audit, logs, config, RUN_STATE and seal. Delete only the three regenerable 2.41 GB spatial caches after each seed.",
        "failure_policy": "Any source/split/reference drift, forbidden read, nonfinite value, backbone update, environment/resource violation, program error or unmet scientific requirement seals FAIL. No retry, threshold relaxation, dataset change or graph tuning.",
    }
    template = deepcopy(load_json(SOURCE_CARD))
    card["worlds"] = template["worlds"]
    card["trajectories"] = template["trajectories"]
    card["source"] = {
        "dataset_run": DATASET,
        "frozen_gse_training_run": TRAINING,
        "corrected_teacher_run": TEACHER,
        "calibration_source_v2_run": VERIFIER,
        "causal_supervision_audit_run": SUPERVISION,
        "causal_reference_proof_run": REFERENCE,
        "old_directional_baseline_run": OLD_DIRECTIONAL,
        "raw_sources": [
            "Sealed C01-C08 deduplicated Zarr: 252430 unique organized LiDAR frames and 188126 causal observations.",
            "Sealed corrected causal Teacher V1R and episode audit: 5306 structural episodes and 1031 transition rows.",
            "Three sealed original GSE checkpoints provide read-only spatial encoders; three frozen feature arrays provide five-frame residual baselines.",
            "Sealed old directional outputs provide the strongest unchanged corrected-label baseline.",
            "No C09/C10/strict/M-TARE sensor, Teacher, model, graph or planner artifact is read.",
        ],
        "license_or_allowed_use": "Local research use of project procedural Cano worlds, LiDAR derivatives, objective TNG/spline/mesh teachers and locally trained checkpoints with full provenance.",
        "partial_reuse": "READ_ONLY_C01_C08_ZARR_THREE_FROZEN_SPATIAL_ENCODERS_CORRECTED_TEACHER_REFERENCE_PROOF_BASELINE_FEATURES_AND_OLD_DIRECTIONAL_OUTPUTS",
    }
    if SYSTEM_PREDECESSOR is not None:
        card["source"].update({
            "system_predecessor_run": SYSTEM_PREDECESSOR,
            "system_predecessor_status": "FAIL_GSE_CAUSAL_EPISODE_TRAINING_V1",
            "system_predecessor_reuse": "FAILURE_EVIDENCE_ONLY; NO CHECKPOINT OR PARTIAL TRAINING REUSE",
        })
        card["source"]["raw_sources"].append(
            "The V1 system-failed run is retained only to prove the NumPy/PyTorch dtype entrance bug and zero optimizer steps; its scratch cache and partial model directory are not reused."
        )
    card["sampling"].update({
        "raw_frame_count": 252430,
        "effective_sample_count": 188126,
        "effective_structure_event_count": 37162,
        "spatial_interval_m": 1.0,
        "rule": "All 188126 C01-C08 observations remain the effective population. Each 12-frame sample contains only the current and up to eleven earlier one-metre frames from the same traversal; C01-C06 trains and C07-C08 selects.",
        "structure_event_counts": {
            "junction_rows": 26608, "terminal_rows": 7525, "turn_rows": 1998,
            "geometry_transition_rows": 1031, "structural_episodes": 5306,
        },
        "temporal_context": "Current plus up to eleven strictly earlier one-metre traversal anchors; left padded to 12 without crossing a traversal reset.",
    })
    card["teacher"].update({
        "valid_mask": "Corridor rows use episode_id=-1. Every structural row belongs to exactly one compact episode. Only the 1030 transition rows with a non-negative boundary inside the 12-frame history supervise boundary regression; the one pre-boundary row remains an event sample.",
        "planner_consistency_plan": "A validation-calibrated stable structural trigger may propose one topology node per contiguous response; predicted boundary offset backprojects its metric location. Edges still require physical traversal.",
    })
    card["split"] = {
        "fit": "C01-C06: 60 worlds and 142184 observations; only the new detector updates.",
        "checkpoint_and_threshold_selection": "C07-C08: 20 worlds and 45942 observations; checkpoint and one structural threshold only.",
        "strict_test": "C10 and M-TARE remain unread until the complete representation and graph policy freeze.",
        "future_validation": "C09/C10 and M-TARE are excluded from this operation.",
        "world_disjoint": True,
        "trajectory_disjoint": True,
        "historical_pollution_audit": "The frozen backbone saw only its predeclared C01-C08 development split. This detector fits C01-C06 and selects on disjoint C07-C08; historical C09 diagnostics contribute no sensor, feature, label, checkpoint or threshold input.",
    }
    card["leakage_audit"].update({
        "future_frames_excluded": True,
        "gt_identity_excluded_at_inference": True,
        "test_excluded_from_supervised_training": True,
        "test_excluded_from_ssl": True,
        "test_excluded_from_normalization": True,
        "test_excluded_from_teacher_calibration": True,
        "test_excluded_from_threshold_calibration": True,
        "test_excluded_from_augmentation_tuning": True,
        "test_excluded_from_checkpoint_selection": True,
    })
    card["estimated_cost"]["disk_gb"] = 3.0
    write_json(CARD_PATH, card)

    inputs = [
        f"{DATASET}/RUN_STATE.json", f"{DATASET}/metrics/summary.json",
        f"{DATASET}/artifacts/evidence_sha256.txt", f"{DATASET}/artifacts/shard_manifest.json",
        f"{TRAINING}/RUN_STATE.json", f"{TRAINING}/metrics/summary.json", f"{TRAINING}/artifacts/evidence_sha256.txt",
        f"{TEACHER}/RUN_STATE.json", f"{TEACHER}/metrics/summary.json", f"{TEACHER}/artifacts/evidence_sha256.txt",
        f"{TEACHER}/artifacts/teacher_observations.jsonl",
        f"{VERIFIER}/RUN_STATE.json", f"{VERIFIER}/metrics/summary.json", f"{VERIFIER}/artifacts/evidence_sha256.txt",
        f"{VERIFIER}/artifacts/pair_cache/pairs.npz",
        f"{SUPERVISION}/RUN_STATE.json", f"{SUPERVISION}/metrics/summary.json", f"{SUPERVISION}/artifacts/evidence_sha256.txt",
        f"{SUPERVISION}/artifacts/transition_timing.csv",
        f"{REFERENCE}/RUN_STATE.json", f"{REFERENCE}/metrics/summary.json", f"{REFERENCE}/metrics/reference_proof.json", f"{REFERENCE}/artifacts/evidence_sha256.txt",
        f"{OLD_DIRECTIONAL}/RUN_STATE.json", f"{OLD_DIRECTIONAL}/metrics/summary.json", f"{OLD_DIRECTIONAL}/artifacts/evidence_sha256.txt",
        f"{OLD_DIRECTIONAL}/artifacts/training/ensemble_selection_outputs.npz",
    ]
    for seed in (0, 1, 2):
        inputs.extend([
            f"{TRAINING}/artifacts/models/seed{seed}/best.pt",
            f"{VERIFIER}/artifacts/models/seed{seed}/frozen_observation_features.npy",
            f"{OLD_DIRECTIONAL}/artifacts/training/seed{seed}/selection_outputs.npz",
        ])
    if SYSTEM_PREDECESSOR is not None:
        inputs.extend([
            f"{SYSTEM_PREDECESSOR}/RUN_STATE.json",
            f"{SYSTEM_PREDECESSOR}/metrics/summary.json",
            f"{SYSTEM_PREDECESSOR}/artifacts/evidence_sha256.txt",
        ])
    tools = {
        "data_card": str(CARD_PATH.relative_to(PROJECT_ROOT)),
        "spatial_backbone": "src/mtare_topo/representation/gse_graph.py",
        "episode_detector": "src/mtare_topo/representation/gse_causal_episode_detector.py",
        "episode_cache": "src/mtare_topo/data/gse_causal_episode_cache.py",
        "episode_sampler": "src/mtare_topo/data/gse_causal_episode_sampler.py",
        "episode_metrics": "src/mtare_topo/evaluation/gse_causal_episode_metrics.py",
        "cache_builder": "tools/v3/build_gse_causal_episode_spatial_cache_v1.py",
        "trainer": "tools/v3/train_gse_causal_episode_detector_v1.py",
        "ensemble_evaluator": "tools/v3/evaluate_gse_causal_episode_ensemble_v1.py",
        "runner": "tools/v3/run_gse_causal_episode_training_v1.py",
        "freezer": FREEZER_PATH,
        "governance": "src/mtare_topo/governance.py",
        "preflight": "tools/v3/preflight.py",
        "create_run": "tools/v3/create_run.py",
    }
    spec = {
        "schema_version": "v3_run_spec_v1", "gate": 3, "execution_phase": 3,
        "date": "20260827", "slug": SLUG, "seed": 0,
        "operation": "training",
        "question": "Can exact twelve-scan causal geometry evidence plus episode-level MIL produce stable, precise structure events that directly justify topology nodes?",
        "method": card["methods"]["main"], "baseline": card["methods"]["baseline"],
        "fallback": card["methods"]["fallback"],
        "data_card": str(CARD_PATH.relative_to(PROJECT_ROOT)),
        "config_path": str(CARD_PATH.relative_to(PROJECT_ROOT)),
        "user_authorization": approval,
        "acceptance_criteria": list(card["metrics_and_pre_registered_gates"].values()) + [
            "Exactly 39996 detector optimizer steps, zero backbone steps and zero C09/C10/strict/M-TARE reads; complete immutable evidence and seal."
        ],
        "expected_counts": {
            "worlds": 80, "inventory_directed_traversals": 16078,
            "observed_directed_traversals": 16076, "unique_frames_per_seed": 252430,
            "model_inference_frames": 757290, "causal_observations": 188126,
            "fit_observations": 142184, "selection_observations": 45942,
            "structural_episodes": 5306, "valid_boundary_rows": 1030,
            "pre_boundary_event_rows": 1, "seeds": 3, "epochs": 6,
            "optimizer_steps": 39996, "backbone_optimizer_steps": 0,
            "trainable_parameters_per_seed": 379014,
            "c09_worlds_read": 0, "strict_test_worlds_read": 0, "mtare_worlds_read": 0,
        },
        "hyperparameters": {
            "history_frames": 12, "directional_bins": 36, "cache_batch_size": 128,
            "epochs": 6, "training_batch_size": 64, "evaluation_batch_size": 256,
            "learning_rate": 0.0003, "weight_decay": 0.0001,
            "gradient_clip": 5.0, "optimizer": "AdamW", "seeds": [0, 1, 2],
            "threshold_grid": "0.000..1.000 inclusive at 0.001",
        },
        "expected_evidence": [
            "Three frozen-encoder cache manifests/digests, three checkpoints and histories, selection outputs, runtime-shaped old/new trigger metrics, identity coverage, environment, logs, deletion audit, RUN_STATE and SHA-256 seal."
        ],
        "estimated_cost": {"compute": "One RTX 5090 D, serial three-seed cache and detector training", "disk_gb": 3.0, "wall_time_hours": 12.0},
        "frozen_inputs": {path: _sha(PROJECT_ROOT / path) for path in inputs},
        "frozen_tools": {name: {"path": path, "sha256": _sha(PROJECT_ROOT / path)} for name, path in tools.items()},
        "working_directory": str(PROJECT_ROOT),
        "command": [
            "/usr/bin/systemd-inhibit", "--what=sleep:shutdown", "--why=GSE twelve-frame causal episode training", "--mode=block",
            "/usr/bin/timeout", "--signal=INT", "--kill-after=300s", "43800s", PYTHON,
            "tools/v3/run_gse_causal_episode_training_v1.py", "--spec", str(SPEC_PATH),
            "--run-dir", str(PROJECT_ROOT / "results/gate3_semantics" / RUN_ID),
        ],
    }
    write_json(SPEC_PATH, spec)
    print(CARD_PATH.relative_to(PROJECT_ROOT))
    print(SPEC_PATH.relative_to(PROJECT_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
