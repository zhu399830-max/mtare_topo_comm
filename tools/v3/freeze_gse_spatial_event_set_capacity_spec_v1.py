#!/usr/bin/env python3
"""Freeze the Data Card and one immutable spatial event-set capacity spec."""

from __future__ import annotations

import hashlib

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import write_json


RUN_ID = "gate3_20260828_gse_spatial_event_set_capacity_v1_seed0"
CARD_PATH = PROJECT_ROOT / "configs/v3/gate3/data_cards/gse_spatial_event_set_capacity_v1.json"
SPEC_PATH = PROJECT_ROOT / "configs/v3/gate3/gse_spatial_event_set_capacity_v1.json"
PYTHON = "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python"
DATASET = "results/gate2_representation/gate2_20260824_gse_deduplicated_dataset_export_v1_seed0"
TEACHER_RUN = "results/gate2_representation/gate2_20260828_gse_spatial_multi_event_teacher_export_v1_seed0"
ENCODERS = "results/gate2_representation/gate2_20260824_gse_graph_three_seed_training_v1r_seed0"
PROJECTION_RUN = "results/gate3_semantics/gate3_20260828_gse_spatial_trace_commit_requalification_v1r2_seed0"
READINESS = "results/gate3_semantics/gate3_20260828_gse_spatial_event_set_readiness_v1r_seed0"


def _sha256(path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    approval = {
        "status": "APPROVED",
        "approved_by": "user-standing-authorization",
        "approved_at": "2026-08-28T18:10:00+08:00",
        "authorized_gates": [3],
        "authorized_operations": ["training"],
        "scope": "One immutable C01-C08 three-seed frozen-encoder spatial event-set capacity proof; zero C09/C10/M-TARE inputs.",
        "confirmation_reference": "User instructed automatic best-choice execution and no routine approval prompts.",
    }
    card = {
        "schema_version": "v3_data_card_v1",
        "card_id": "gse_spatial_event_set_capacity_v1",
        "title": "Frozen-encoder multi-event structure detection and robot-relative localization capacity",
        "status": "APPROVED_FOR_ONE_IMMUTABLE_GSE_SPATIAL_EVENT_SET_CAPACITY_V1",
        "operation": "training",
        "purpose": "Test whether a learned 16-query set decoder can replace the invalid mutually-exclusive scene label with simultaneous visible terminal/junction events and <=4 m robot-relative positions.",
        "approval": approval,
        "source": {
            "dataset_run": DATASET,
            "teacher_run": TEACHER_RUN,
            "encoder_run": ENCODERS,
            "readiness_run": READINESS,
            "exclusive_center_projection_run": PROJECTION_RUN,
            "raw_sources": [
                "Sealed C01-C08 five-frame organized LiDAR sequences at one-metre arc spacing.",
                "Sealed fixed-16 native-mesh-LOS terminal/junction Teacher.",
                "Three corresponding frozen GSE five-frame encoder checkpoints.",
                "C01-C08-only historical single-center projections for the exclusive baseline.",
            ],
            "license_or_allowed_use": "Local research use of project-generated procedural Cano worlds and locally trained checkpoints.",
        },
        "worlds": {
            "train": ["S01-S10_C01-C06"],
            "validation": ["S01-S10_C07-C08"],
            "strict_test": ["S01-S10_C10", "SEALED_MTARE_FINAL_WORLDS"],
            "historical_development_only": ["S01-S10_C09"],
            "fit": "C01-C06: 60 worlds",
            "selection_description": "C07-C08: 20 disjoint worlds",
            "forbidden": "C09, C10 and all M-TARE worlds",
            "count": 80,
        },
        "trajectories": [
            {"id": "C01-C06_all_directed_traversals", "world": "S01-S10_C01-C06", "split": "train", "world_count": 60, "directed_traversal_count": 12106, "duration_s": 178494, "distance_m": 184552.021499521, "spatial_coverage_m": 92276.010749762, "independent": True},
            {"id": "C07-C08_all_directed_traversals", "world": "S01-S10_C07-C08", "split": "validation", "world_count": 20, "directed_traversal_count": 3972, "duration_s": 57858, "distance_m": 59821.90390958, "spatial_coverage_m": 29910.951954792, "independent": True},
        ],
        "sampling": {
            "raw_frame_count": 252430,
            "effective_sample_count": 188126,
            "effective_structure_event_count": 1076,
            "fit_observations": 142184,
            "selection_observations": 45942,
            "visible_event_tokens": 133055,
            "fit_visible_tokens": 99492,
            "selection_visible_tokens": 33563,
            "zero_event_rows": 81069,
            "single_event_rows": 83093,
            "multi_event_rows": 23964,
            "maximum_set_cardinality": 5,
            "fixed_slots": 16,
            "spatial_interval_m": 1.0,
            "temporal_context": "Exactly current plus four earlier scans from the same directed traversal.",
            "independent_sampling_unit": "Procedural world-case and objective terminal/junction identity; split before fitting.",
            "rule": "Visit every C01-C06 observation once per epoch, including all 62,247 fit empty-set rows; no deletion, resampling or test access.",
            "structure_event_counts": {"terminal": 30789, "junction": 102266, "identities": 1076},
        },
        "split": {
            "fit": "C01-C06 supplies all gradients, positive-slot weight and terminal/junction class weights.",
            "checkpoint_and_threshold_selection": "C07-C08 selects minimum set loss checkpoint and one confidence from the predeclared 0.05:0.05:0.95 grid.",
            "strict_test": "C10 and M-TARE remain unopened; C09 is historical development only and is not read.",
            "world_disjoint": True,
            "trajectory_disjoint": True,
            "historical_pollution_audit": "A preparatory file-search process enumerated one historical C09 archive's field names/shapes but no values; that process is quarantined and contributes no input, statistic, code choice, threshold or evidence. The formal command binds only C01-C08 hashes and records zero C09/C10/M-TARE reads.",
        },
        "teacher": {
            "source": "Every degree-1 terminal and degree>=3 junction within 50 m whose objective node target has native-mesh line of sight from the fifth causal pose.",
            "labels": "Fixed 16 slots: presence, terminal/junction type, forward-left-up relative xyz, Teacher-only identity and mask.",
            "valid_mask": "event_mask marks only unique in-range native-LOS events; padding uses type/identity -1 and zero xyz. Zero-event observations remain valid empty sets.",
            "student_input": "Only frozen encoder context [128] and circular directional feature [128,180] from five causal LiDAR scans.",
            "forbidden_student_inputs": ["Teacher identity", "absolute pose", "world", "parent", "TNG", "mesh", "future frame", "C09", "C10", "M-TARE"],
            "planner_consistency_plan": "At deployment the decoder emits robot-relative event proposals only. Graph nodes remain provisional until causal stability checks, and graph edges still require physical traversal; no Teacher identity or objective position is available to the planner.",
        },
        "leakage_audit": {
            "future_frames_excluded": True,
            "gt_identity_excluded_at_inference": True,
            "objective_position_excluded_at_inference": True,
            "pose_excluded_from_decoder": True,
            "test_excluded_from_supervised_training": True,
            "test_excluded_from_ssl": True,
            "test_excluded_from_normalization": True,
            "test_excluded_from_teacher_calibration": True,
            "test_excluded_from_threshold_calibration": True,
            "test_excluded_from_augmentation_tuning": True,
            "test_excluded_from_checkpoint_selection": True,
        },
        "methods": {
            "main": "For each seed, freeze the corresponding five-frame GSE encoder and train only the 93,638-parameter 16-query SpatialEventSetDecoder. Content-canonical assignment supervises presence, terminal/junction type, relative xyz and uncertainty. Fit-only inverse-frequency weights correct 4.37% positive-slot and terminal/junction imbalance. Circular feature roll and equal xyz rotation are train-only.",
            "baseline": "Primary baseline is the same frozen encoder's mutually-exclusive terminal/junction logits plus the sealed C01-C08 single-center projection. Secondary baseline is the fixed five-scan analytic range-geometry/open-sector event rule with its deepest observed sector as a single center.",
            "fallback": "None inside this run. Descriptor output remains typed but its loss is fixed to zero so this proof isolates event detection/localization; association learning is a later, independently gated task.",
        },
        "acceptance": {
            "matching": "A true positive requires same event type and one-to-one 3D error <=4.0 m.",
            "precision": "Every seed precision >=0.90.",
            "recall": "Every seed overall recall >=0.25 and terminal/junction recall each >=0.20.",
            "baseline_gain": "Every seed F1 exceeds the better baseline by >=0.05 and multi-event recall exceeds the better baseline by >=0.10.",
            "localization": "Every seed matched localization MAE <=4.0 m.",
            "system": "Exactly 3x4,448 decoder optimizer steps, zero encoder steps, source unchanged, zero C09/C10/M-TARE reads and complete seal.",
        },
        "estimated_cost": {
            "compute": "One RTX 5090 D; three seed-matched cache/train stages serially plus CPU analytic baseline",
            "wall_time_hours": 10.0,
            "gpu_memory_gb": 8,
            "host_ram_gb": 16,
            "scratch_disk_gb": 10,
            "retained_disk_gb": 1.0,
            "disk_gb": 10,
        },
        "retention": "Retain three decoder checkpoints, histories, selection outputs, cache manifests/digests, compact legacy logits, baseline outputs, threshold curves, paper PNG/PDF/SVG, logs and seal. Delete only each exact deterministic ~8.1 GiB temporary feature cache after its seed completes.",
        "failure_policy": "Any source/split/Teacher/environment drift, forbidden input/read, nonfinite value, encoder update, resource violation, program error or unmet science gate seals FAIL. Do not retry with changed queries, radius, match cap, thresholds, backbone or planner.",
    }
    write_json(CARD_PATH, card)

    inputs = [
        f"{DATASET}/RUN_STATE.json", f"{DATASET}/metrics/summary.json",
        f"{DATASET}/artifacts/evidence_sha256.txt", f"{DATASET}/artifacts/shard_manifest.json",
        f"{TEACHER_RUN}/RUN_STATE.json", f"{TEACHER_RUN}/metrics/summary.json",
        f"{TEACHER_RUN}/artifacts/evidence_sha256.txt",
        f"{TEACHER_RUN}/artifacts/export/teacher_shard_manifest.jsonl",
        f"{TEACHER_RUN}/artifacts/export/event_identity_map.json",
        f"{ENCODERS}/RUN_STATE.json", f"{ENCODERS}/metrics/summary.json",
        f"{ENCODERS}/artifacts/evidence_sha256.txt",
        f"{PROJECTION_RUN}/RUN_STATE.json", f"{PROJECTION_RUN}/metrics/summary.json",
        f"{PROJECTION_RUN}/artifacts/evidence_sha256.txt",
        f"{READINESS}/RUN_STATE.json", f"{READINESS}/metrics/summary.json",
        f"{READINESS}/artifacts/evidence_sha256.txt",
    ]
    for seed in (0, 1, 2):
        inputs.extend((
            f"{ENCODERS}/artifacts/models/seed{seed}/best.pt",
            f"{PROJECTION_RUN}/artifacts/projection/seed{seed}_all_rows.npz",
        ))
    tools = {
        "data_card": str(CARD_PATH.relative_to(PROJECT_ROOT)),
        "encoder": "src/mtare_topo/representation/gse_graph.py",
        "decoder": "src/mtare_topo/representation/gse_spatial_event_set.py",
        "cache": "src/mtare_topo/data/gse_spatial_event_set_cache.py",
        "metrics": "src/mtare_topo/evaluation/gse_spatial_event_set_metrics.py",
        "nonlearning_observation": "src/mtare_topo/semantics/nonlearning_geometry_observation.py",
        "range_geometry": "src/mtare_topo/semantics/range_geometry_baseline.py",
        "range_exit": "src/mtare_topo/semantics/range_exit_baseline.py",
        "cache_builder": "tools/v3/build_gse_spatial_event_set_cache_v1.py",
        "trainer": "tools/v3/train_gse_spatial_event_set_v1.py",
        "evaluator": "tools/v3/evaluate_gse_spatial_event_set_capacity_v1.py",
        "runner": "tools/v3/run_gse_spatial_event_set_capacity_v1.py",
        "freezer": "tools/v3/freeze_gse_spatial_event_set_capacity_spec_v1.py",
        "preflight": "tools/v3/preflight.py",
        "create_run": "tools/v3/create_run.py",
    }
    spec = {
        "schema_version": "v3_run_spec_v1",
        "gate": 3, "execution_phase": 3, "date": "20260828",
        "slug": "gse_spatial_event_set_capacity_v1", "seed": 0,
        "operation": "training",
        "question": "Can three frozen-encoder 16-query decoders safely detect simultaneous visible terminal/junction sets and localize them within 4 m on C07-C08 better than exclusive and analytic baselines?",
        "method": card["methods"]["main"],
        "baseline": card["methods"]["baseline"],
        "fallback": card["methods"]["fallback"],
        "data_card": str(CARD_PATH.relative_to(PROJECT_ROOT)),
        "config_path": str(CARD_PATH.relative_to(PROJECT_ROOT)),
        "user_authorization": approval,
        "hyperparameters": {
            "history_frames": 5, "query_count": 16, "directional_bins": 180,
            "seeds": [0, 1, 2], "epochs": 8, "batch_size": 256,
            "learning_rate": 0.0003, "weight_decay": 0.0001, "gradient_clip": 5.0,
            "loss_weights": {"presence": 1.0, "event_type": 1.0, "position": 5.0, "descriptor": 0.0, "uncertainty": 0.01},
            "fit_presence_positive_weight": 21.86559723394846,
            "fit_event_type_class_weights": [2.1448712973742077, 0.651987575197578],
            "threshold_grid": [round(value * 0.05, 2) for value in range(1, 20)],
            "matching_maximum_3d_error_m": 4.0,
        },
        "expected_counts": {
            "worlds": 80, "unique_frames_per_seed": 252430,
            "observations": 188126, "fit_rows": 142184, "selection_rows": 45942,
            "visible_tokens": 133055, "fit_visible_tokens": 99492,
            "optimizer_steps_per_seed": 4448, "encoder_optimizer_steps": 0,
            "c09_worlds_read": 0, "c10_worlds_read": 0, "mtare_worlds_read": 0,
        },
        "acceptance_criteria": list(card["acceptance"].values()),
        "estimated_cost": card["estimated_cost"],
        "expected_evidence": [
            "Three feature-cache manifests/deletion records, compact legacy logits, decoder checkpoints, histories and selection outputs.",
            "Same-type <=4 m precision/recall/F1, per-type and multi-event metrics for all seeds and both baselines.",
            "Paper PNG/PDF/SVG plus source, environment, logs, source integrity, RUN_STATE and SHA-256 seal.",
        ],
        "frozen_inputs": {relative: _sha256(PROJECT_ROOT / relative) for relative in inputs},
        "frozen_tools": {
            name: {"path": relative, "sha256": _sha256(PROJECT_ROOT / relative)}
            for name, relative in tools.items()
        },
        "command": [
            "/usr/bin/timeout", "--signal=INT", "--kill-after=60s", "43200s",
            PYTHON, "tools/v3/run_gse_spatial_event_set_capacity_v1.py",
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
