#!/usr/bin/env python3
"""Freeze the approved spatial-context longitudinal corrective run."""

from __future__ import annotations

import hashlib

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, write_json


RUN_ID = "gate3_20260828_gse_spatial_longitudinal_corrective_v1_seed0"
CARD_PATH = PROJECT_ROOT / "configs/v3/gate3/data_cards/gse_spatial_longitudinal_corrective_v1.json"
SPEC_PATH = PROJECT_ROOT / "configs/v3/gate3/gse_spatial_longitudinal_corrective_v1.json"
PYTHON = "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python"
DATASET = "results/gate2_representation/gate2_20260824_gse_deduplicated_dataset_export_v1_seed0"
GSE = "results/gate2_representation/gate2_20260824_gse_graph_three_seed_training_v1r_seed0"
CAUSAL_TEACHER = "results/gate2_representation/gate2_20260827_gse_corrected_causal_teacher_manifest_v1r_seed0"
ACTION = "results/gate3_semantics/gate3_20260828_gse_action_set_node_training_v1_seed0"
CENTER = "results/gate3_semantics/gate3_20260828_gse_event_center_offset_training_v1r3_seed0"
SCALAR = "results/gate3_semantics/gate3_20260828_gse_event_center_pair_consistency_training_v1r3_seed0"
PROJECTION = "results/gate3_semantics/gate3_20260828_gse_projected_trace_commit_capacity_v1_seed0"
SPATIAL = "results/gate3_semantics/gate3_20260828_gse_spatial_event_center_training_v1_seed0"
QUALIFICATION = "results/gate3_semantics/gate3_20260828_gse_spatial_event_center_qualification_v1r_seed0"
AUDIT = "results/gate3_semantics/gate3_20260828_gse_spatial_center_residual_audit_v1r3_seed0"


def _sha(path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    approval = {
        "status": "APPROVED", "approved_by": "user-standing-authorization",
        "approved_at": "2026-08-28T11:50:00+08:00",
        "authorized_gates": [3], "authorized_operations": ["training"],
        "scope": "One immutable C01-C08 three-seed spatial-context longitudinal corrective; frozen V1 transverse decoder and zero C09/C10/M-TARE.",
        "confirmation_reference": "User instructed automatic best-choice execution without routine approval prompts.",
    }
    card = load_json(PROJECT_ROOT / "configs/v3/gate3/data_cards/gse_spatial_event_center_training_v1.json")
    card.update({
        "card_id": "gse_spatial_longitudinal_corrective_v1",
        "title": "Spatial-context longitudinal event-center corrective",
        "status": "APPROVED_FOR_ONE_IMMUTABLE_GSE_SPATIAL_LONGITUDINAL_CORRECTIVE_V1",
        "operation": "training",
        "purpose": "Correct the junction-dominated forward event-center error isolated by the sealed residual audit while preserving learned lateral/up coordinates exactly.",
        "approval": approval,
        "methods": {
            "main": "For each seed, reproduce the exact V1 spatial cache and checkpoint. Freeze all 525698 V1 spatial-decoder parameters and its lateral/up output; add a zero-initialized 73985-parameter residual over the same 576D spatial context. Add the bounded residual only to the sealed scalar forward distance and train with unchanged equal-event direct plus cross-traversal global-center loss.",
            "baseline": "Sealed V1 arithmetic spatial ensemble: relative improvement 13.96597%, within-4m gain 0.095971, lateral/up MAE 0.209026/0.136350 m. Residual audit shows oracle-long plus predicted-transverse reaches 0.999928 within-4m.",
            "fallback": "None. Failure stops event-center regression; no loss-weight, epoch, radius, threshold, aggregation or planner tuning.",
        },
        "estimated_cost": {"compute": "One RTX 5090 D; three seeds sequentially", "wall_time_hours": 3.0, "host_ram_gb": 16, "gpu_memory_gb": 12, "scratch_disk_gb": 3.0, "retained_disk_gb": 1.0, "disk_gb": 3.0},
        "retention": "Retain three longitudinal residual checkpoints, histories, selection outputs, ensemble metrics, exact cache reproduction manifests/deletion audit, logs and seal. Delete each 2.526 GB deterministic cache after its seed.",
        "failure_policy": "Any cache mismatch, transverse drift, source/split drift, forbidden read, frozen-module update, resource/program error or unmet unchanged gate seals FAIL. No retry or gate relaxation.",
    })
    card["source"].update({"spatial_predecessor_run": SPATIAL, "spatial_qualification_run": QUALIFICATION, "residual_audit_run": AUDIT})
    card["split"]["fit"] = (
        "C01-C06 updates only the new 73985-parameter longitudinal residual; "
        "all 525698 V1 spatial-decoder parameters remain frozen."
    )
    card["acceptance"].update({"transverse_frozen": "Every seed and ensemble lateral/up output has exact zero drift from V1."})
    write_json(CARD_PATH, card)

    inputs = [
        f"{DATASET}/RUN_STATE.json", f"{DATASET}/metrics/summary.json", f"{DATASET}/artifacts/evidence_sha256.txt", f"{DATASET}/artifacts/shard_manifest.json",
        f"{GSE}/RUN_STATE.json", f"{GSE}/metrics/summary.json", f"{GSE}/artifacts/evidence_sha256.txt",
        f"{CAUSAL_TEACHER}/RUN_STATE.json", f"{CAUSAL_TEACHER}/metrics/summary.json", f"{CAUSAL_TEACHER}/artifacts/evidence_sha256.txt", f"{CAUSAL_TEACHER}/artifacts/teacher_observations.jsonl",
        f"{ACTION}/RUN_STATE.json", f"{ACTION}/metrics/summary.json", f"{ACTION}/artifacts/evidence_sha256.txt", f"{ACTION}/scratch/action_set_cache/manifest.json",
        f"{CENTER}/RUN_STATE.json", f"{CENTER}/metrics/summary.json", f"{CENTER}/artifacts/evidence_sha256.txt", f"{CENTER}/artifacts/teacher/event_center_teacher.npz",
        f"{SCALAR}/RUN_STATE.json", f"{SCALAR}/metrics/summary.json", f"{SCALAR}/artifacts/evidence_sha256.txt",
        f"{PROJECTION}/RUN_STATE.json", f"{PROJECTION}/metrics/summary.json", f"{PROJECTION}/artifacts/evidence_sha256.txt", f"{PROJECTION}/artifacts/projection/event_center_projection.npz",
        f"{SPATIAL}/RUN_STATE.json", f"{SPATIAL}/metrics/summary.json", f"{SPATIAL}/artifacts/evidence_sha256.txt",
        f"{QUALIFICATION}/RUN_STATE.json", f"{QUALIFICATION}/metrics/summary.json", f"{QUALIFICATION}/artifacts/evidence_sha256.txt",
        f"{AUDIT}/RUN_STATE.json", f"{AUDIT}/metrics/summary.json", f"{AUDIT}/artifacts/evidence_sha256.txt", f"{AUDIT}/artifacts/audit/summary.json",
    ]
    action_cache = PROJECT_ROOT / ACTION / "scratch/action_set_cache"
    inputs.extend(str(path.relative_to(PROJECT_ROOT)) for path in sorted(action_cache.glob("*.npy")))
    for seed in (0, 1, 2):
        inputs.extend([
            f"{GSE}/artifacts/models/seed{seed}/best.pt", f"{ACTION}/artifacts/models/seed{seed}/best.pt",
            f"{SCALAR}/artifacts/models/seed{seed}/best.pt", f"{SPATIAL}/artifacts/models/seed{seed}/best.pt",
            f"{SPATIAL}/artifacts/models/seed{seed}/selection_outputs.npz", f"{SPATIAL}/artifacts/cache_manifests/seed{seed}_manifest.json",
        ])
    tools = {
        "data_card": str(CARD_PATH.relative_to(PROJECT_ROOT)),
        "spatial_encoder": "src/mtare_topo/representation/gse_graph.py",
        "spatial_projection": "src/mtare_topo/representation/gse_causal_episode_detector.py",
        "spatial_decoder": "src/mtare_topo/representation/gse_spatial_event_center.py",
        "spatial_cache": "src/mtare_topo/data/gse_spatial_event_center_cache.py",
        "action_cache": "src/mtare_topo/data/gse_action_set_cache.py",
        "action_model": "src/mtare_topo/representation/gse_action_set_node.py",
        "scalar_model": "src/mtare_topo/representation/gse_event_center_offset.py",
        "center_teacher": "src/mtare_topo/teacher/gse_event_center_teacher.py",
        "cache_builder": "tools/v3/build_gse_spatial_event_center_cache_v1.py",
        "trainer": "tools/v3/train_gse_spatial_longitudinal_corrective_v1.py",
        "trainer_helper": "tools/v3/train_gse_spatial_event_center_v1.py",
        "evaluator": "tools/v3/evaluate_gse_spatial_event_center_ensemble_v1.py",
        "runner": "tools/v3/run_gse_spatial_longitudinal_corrective_v1.py",
        "freezer": "tools/v3/freeze_gse_spatial_longitudinal_corrective_spec_v1.py",
        "preflight": "tools/v3/preflight.py", "create_run": "tools/v3/create_run.py",
    }
    spec = {
        "schema_version": "v3_run_spec_v1", "gate": 3, "execution_phase": 3,
        "date": "20260828", "slug": "gse_spatial_longitudinal_corrective_v1", "seed": 0,
        "operation": "training",
        "question": "Can spatial context correct junction longitudinal center error while preserving the solved transverse decoder and pass all unchanged association gates?",
        "method": card["methods"]["main"], "baseline": card["methods"]["baseline"], "fallback": card["methods"]["fallback"],
        "data_card": str(CARD_PATH.relative_to(PROJECT_ROOT)), "config_path": str(CARD_PATH.relative_to(PROJECT_ROOT)), "user_authorization": approval,
        "hyperparameters": {"history_frames": 5, "directional_bins": 36, "elevation_bins": 2, "epochs": 10, "seeds": [0, 1, 2], "learning_rate": 0.0005, "weight_decay": 0.0001, "identities_per_event": 16, "direct_to_relative_loss_weight": "1:1", "gradient_clip": 5.0, "residual_bound_m": 12.0, "cache_batch_size": 128, "evaluation_batch_size": 128},
        "expected_counts": {"worlds": 80, "unique_frames_per_seed": 252430, "causal_observations": 188126, "fit_rows": 25294, "selection_rows": 8839, "optimizer_steps_per_seed": 3960, "spatial_decoder_optimizer_steps": 0, "backbone_optimizer_steps": 0, "c09_worlds_read": 0, "c10_worlds_read": 0, "mtare_worlds_read": 0, "temporary_cache_bytes_per_seed": 2526043720},
        "acceptance_criteria": list(card["acceptance"].values()), "estimated_cost": card["estimated_cost"],
        "expected_evidence": ["Three exact cache reproduction manifests and deletion records.", "Three residual checkpoints/histories/selection outputs and unchanged transverse audit.", "Ensemble unchanged seven-gate metrics, environment, logs, source integrity, RUN_STATE and seal."],
        "frozen_inputs": {relative: _sha(PROJECT_ROOT / relative) for relative in inputs},
        "frozen_tools": {name: {"path": relative, "sha256": _sha(PROJECT_ROOT / relative)} for name, relative in tools.items()},
        "command": [
            "/usr/bin/timeout", "--signal=INT", "--kill-after=60s", "14400s", PYTHON,
            "tools/v3/run_gse_spatial_longitudinal_corrective_v1.py", "--spec", str(SPEC_PATH),
            "--run-dir", str(PROJECT_ROOT / "results/gate3_semantics" / RUN_ID),
        ],
        "working_directory": str(PROJECT_ROOT),
    }
    write_json(SPEC_PATH, spec); print(SPEC_PATH.relative_to(PROJECT_ROOT)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
