#!/usr/bin/env python3
"""Freeze one read-only spatial event-set failure attribution Data Card/spec."""

from __future__ import annotations

import hashlib

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import validate_data_card, write_json


RUN_ID = "gate3_20260828_gse_spatial_event_set_failure_attribution_v1_seed0"
CARD = PROJECT_ROOT / "configs/v3/gate3/data_cards/gse_spatial_event_set_failure_attribution_v1.json"
SPEC = PROJECT_ROOT / "configs/v3/gate3/gse_spatial_event_set_failure_attribution_v1.json"
CARD_ID = "gse_spatial_event_set_failure_attribution_v1"
CARD_STATUS = "APPROVED_FOR_ONE_IMMUTABLE_GSE_SPATIAL_EVENT_SET_FAILURE_ATTRIBUTION_V1"
SLUG = "gse_spatial_event_set_failure_attribution_v1"
FREEZER_TOOL = "tools/v3/freeze_gse_spatial_event_set_failure_attribution_spec_v1.py"
RUNNER_TOOL = "tools/v3/run_gse_spatial_event_set_failure_attribution_v1.py"
TEACHER = "results/gate2_representation/gate2_20260828_gse_spatial_multi_event_teacher_export_v1_seed0"
CAPACITY = "results/gate3_semantics/gate3_20260828_gse_spatial_event_set_capacity_v1_seed0"
PYTHON = "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python"


def _sha(path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    approval = {
        "status": "APPROVED", "approved_by": "user-standing-authorization",
        "approved_at": "2026-08-28T19:00:00+08:00", "authorized_gates": [3],
        "authorized_operations": ["training"],
        "scope": "One immutable read-only C07-C08 attribution of the sealed spatial event-set capacity failure; zero training/C09/C10/M-TARE.",
        "confirmation_reference": "User instructed automatic best-choice execution and no routine approval prompts.",
    }
    card = {
        "schema_version": "v3_data_card_v1", "card_id": CARD_ID,
        "title": "Spatial event-set duplicate, confidence and encoder-capacity attribution",
        "status": CARD_STATUS,
        "operation": "audit", "purpose": "Decide whether the sealed set failure is recoverable by fixed 4 m duplicate suppression/confidence calibration or requires a new spatial encoder objective.",
        "approval": approval,
        "source": {"teacher_run": TEACHER, "capacity_run": CAPACITY, "raw_sources": ["Sealed C07-C08 fixed-16 Teacher targets", "Three sealed decoder selection-output archives", "Sealed exclusive baseline metrics"], "license_or_allowed_use": "Local project-generated research evidence."},
        "worlds": {"train": ["S01-S10_C01-C06 (declared development source; not read by this audit)"], "validation": ["S01-S10_C07-C08"], "strict_test": ["S01-S10_C10", "M-TARE"], "fit": "None", "selection_description": "C07-C08 existing sealed outputs only", "forbidden": "C09, C10 and M-TARE", "count": 20},
        "trajectories": [{"id": "C07-C08_all_directed_traversals", "world": "S01-S10_C07-C08", "split": "validation", "world_count": 20, "directed_traversal_count": 3972, "duration_s": 57858, "distance_m": 59821.90390958, "spatial_coverage_m": 29910.951954792, "independent": True}],
        "sampling": {"raw_frame_count": 45942, "effective_sample_count": 45942, "effective_structure_event_count": 274, "visible_event_tokens": 33563, "multi_event_rows": 6153, "spatial_interval_m": 1.0, "temporal_context": "No new sensor read; audit sealed five-frame outputs.", "independent_sampling_unit": "C07-C08 world and objective event identity.", "rule": "Use every sealed selection row and token exactly once; no resampling or new threshold selection.", "structure_event_counts": {"terminal_tokens": 7596, "junction_tokens": 25967}},
        "split": {"fit": "None", "checkpoint_and_threshold_selection": "None; reuse each seed's sealed threshold 0.95.", "strict_test": "C09/C10/M-TARE unread.", "world_disjoint": True, "trajectory_disjoint": True, "historical_pollution_audit": "The formal audit binds only C07-C08 Teacher and capacity-run hashes."},
        "teacher": {"source": "Sealed native-mesh-LOS fixed-16 event Teacher.", "labels": "terminal/junction type and robot-relative xyz.", "valid_mask": "Use the sealed event_mask; padding remains inactive.", "student_input": "No student or inference; evaluator reads sealed predictions.", "planner_consistency_plan": "Attribution only; no graph/planner output or parameter change."},
        "leakage_audit": {"future_frames_excluded": True, "gt_identity_excluded_at_inference": True, "test_excluded_from_supervised_training": True, "test_excluded_from_ssl": True, "test_excluded_from_normalization": True, "test_excluded_from_teacher_calibration": True, "test_excluded_from_threshold_calibration": True, "test_excluded_from_augmentation_tuning": True, "test_excluded_from_checkpoint_selection": True},
        "methods": {
            "main": "At each sealed 0.95 threshold, compute same-type <=4 m duplicates and deterministic confidence-first 4 m NMS; independently compute all-16-query typed and position-only <=4 m target-coverage oracles. Stratify by target distance, cardinality, type and topology family.",
            "baseline": "Sealed exclusive single-center F1=0.391306 and recall=0.262849.",
            "fallback": "If all-seed NMS F1 is not baseline+0.05 and all-seed typed-query oracle recall is not baseline recall+0.10, require a new spatial encoder objective; no corrective training inside this audit.",
        },
        "acceptance": {"population": "Exactly 20 worlds, 45942 observations, 33563 targets and 6153 multi-event rows.", "decision": "Emit exactly one of duplicate-dominant, confidence/cardinality-dominant or encoder-insufficient by the frozen gates.", "system": "Zero training/inference/threshold selection/C09/C10/M-TARE, source unchanged and complete plot/seal."},
        "estimated_cost": {"compute": "CPU read-only array audit", "wall_time_hours": 0.1, "host_ram_gb": 4, "gpu_memory_gb": 0, "disk_gb": 0.1},
        "retention": "Retain summary, distance/cardinality/type/family strata, NMS/oracle evidence, PNG/PDF/SVG/source, logs and seal.",
        "failure_policy": "Any source/count/join/metric/decision ambiguity or forbidden read fails closed; no threshold or prediction mutation beyond the evaluator-only fixed NMS counterfactual.",
    }
    report = validate_data_card(card)
    if not report.passed:
        raise RuntimeError(f"generated failure attribution Data Card invalid: {report.errors}")
    write_json(CARD, card)
    inputs = [
        f"{TEACHER}/RUN_STATE.json", f"{TEACHER}/metrics/summary.json", f"{TEACHER}/artifacts/evidence_sha256.txt",
        f"{CAPACITY}/RUN_STATE.json", f"{CAPACITY}/metrics/summary.json", f"{CAPACITY}/metrics/capacity/summary.json", f"{CAPACITY}/artifacts/evidence_sha256.txt",
    ]
    for seed in (0, 1, 2):
        inputs.extend((f"{CAPACITY}/artifacts/models/seed{seed}/summary.json", f"{CAPACITY}/artifacts/models/seed{seed}/selection_outputs.npz"))
    tools = {
        "data_card": str(CARD.relative_to(PROJECT_ROOT)),
        "teacher_join": "src/mtare_topo/data/gse_spatial_event_set_cache.py",
        "metrics": "src/mtare_topo/evaluation/gse_spatial_event_set_metrics.py",
        "executor": "tools/v3/execute_gse_spatial_event_set_failure_attribution_v1.py",
        "runner": RUNNER_TOOL,
        "freezer": FREEZER_TOOL,
        "preflight": "tools/v3/preflight.py", "create_run": "tools/v3/create_run.py",
    }
    spec = {
        "schema_version": "v3_run_spec_v1", "gate": 3, "execution_phase": 3, "date": "20260828",
        "slug": SLUG, "seed": 0, "operation": "audit",
        "question": "Are duplicate/confidence failures sufficient to explain the sealed set-decoder gap, or is a new spatial encoder objective required?",
        "method": card["methods"]["main"], "baseline": card["methods"]["baseline"], "fallback": card["methods"]["fallback"],
        "data_card": str(CARD.relative_to(PROJECT_ROOT)), "config_path": str(CARD.relative_to(PROJECT_ROOT)), "user_authorization": approval,
        "hyperparameters": {"sealed_thresholds": [0.95, 0.95, 0.95], "match_and_nms_radius_m": 4.0, "distance_bins_m": [0,4,8,12,16,24,32,40,50], "nms_f1_gain_gate": 0.05, "typed_oracle_recall_gain_gate": 0.10},
        "expected_counts": {"worlds": 20, "observations": 45942, "target_tokens": 33563, "multi_event_rows": 6153, "optimizer_steps": 0, "model_inference_frames": 0, "c09_worlds_read": 0, "c10_worlds_read": 0, "mtare_worlds_read": 0},
        "acceptance_criteria": list(card["acceptance"].values()), "estimated_cost": card["estimated_cost"],
        "expected_evidence": ["Three sealed/NMS/oracle comparisons and distance/cardinality/type/family strata.", "One deterministic route decision and paper PNG/PDF/SVG/source.", "Logs, source integrity, RUN_STATE and SHA-256 seal."],
        "frozen_inputs": {relative: _sha(PROJECT_ROOT / relative) for relative in inputs},
        "frozen_tools": {name: {"path": relative, "sha256": _sha(PROJECT_ROOT / relative)} for name, relative in tools.items()},
        "command": ["/usr/bin/timeout", "--signal=INT", "--kill-after=60s", "1200s", PYTHON, RUNNER_TOOL, "--spec", str(SPEC), "--run-dir", str(PROJECT_ROOT / "results/gate3_semantics" / RUN_ID)],
        "working_directory": str(PROJECT_ROOT),
    }
    write_json(SPEC, spec)
    print(SPEC.relative_to(PROJECT_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
