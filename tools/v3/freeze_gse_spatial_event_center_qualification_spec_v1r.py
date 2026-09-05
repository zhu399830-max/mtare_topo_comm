#!/usr/bin/env python3
"""Freeze the evaluation-only corrective for the JSON serialization failure."""

from __future__ import annotations

import hashlib

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, write_json


RUN_ID = "gate3_20260828_gse_spatial_event_center_qualification_v1r_seed0"
CARD_PATH = PROJECT_ROOT / "configs/v3/gate3/data_cards/gse_spatial_event_center_qualification_v1r.json"
SPEC_PATH = PROJECT_ROOT / "configs/v3/gate3/gse_spatial_event_center_qualification_v1r.json"
PYTHON = "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python"
TRAINING = "results/gate3_semantics/gate3_20260828_gse_spatial_event_center_training_v1_seed0"
ACTION = "results/gate3_semantics/gate3_20260828_gse_action_set_node_training_v1_seed0"
CENTER = "results/gate3_semantics/gate3_20260828_gse_event_center_offset_training_v1r3_seed0"
PROJECTION = "results/gate3_semantics/gate3_20260828_gse_projected_trace_commit_capacity_v1_seed0"


def _sha(path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    approval = {
        "status": "APPROVED", "approved_by": "user-standing-authorization",
        "approved_at": "2026-08-28T11:10:00+08:00",
        "authorized_gates": [3], "authorized_operations": ["checkpoint_selection"],
        "scope": "One immutable evaluation-only qualification of the three sealed V1 spatial decoders; no training and zero C09/C10/M-TARE.",
        "confirmation_reference": "User instructed automatic best-choice execution without routine approval prompts.",
    }
    card = load_json(PROJECT_ROOT / "configs/v3/gate3/data_cards/gse_spatial_event_center_training_v1.json")
    card.update({
        "card_id": "gse_spatial_event_center_qualification_v1r",
        "title": "Evaluation-only qualification of sealed spatial event-center models",
        "status": "APPROVED_FOR_ONE_IMMUTABLE_GSE_SPATIAL_EVENT_CENTER_QUALIFICATION_V1R",
        "operation": "checkpoint_selection",
        "purpose": "Recover only the ensemble metric serialization that failed after all three spatial models completed successfully.",
        "approval": approval,
        "methods": {
            "main": "Read the three sealed V1 C07-C08 selection outputs, average predicted local vectors/centers, compare once against the unchanged sealed scalar projection, cast NumPy gate results to native JSON booleans and seal the metrics.",
            "baseline": "The same sealed scalar V1R3 projection and the same pre-registered V1 spatial qualification gates.",
            "fallback": "None. Any program or scientific failure is sealed; models are not retrained and thresholds are not changed.",
        },
        "estimated_cost": {
            "compute": "CPU-only evaluation of three sealed selection outputs",
            "wall_time_hours": 0.05, "host_ram_gb": 2, "gpu_memory_gb": 0,
            "scratch_disk_gb": 0, "retained_disk_gb": 0.1, "disk_gb": 0.1,
        },
        "retention": "Retain ensemble metrics/output, source integrity, log, RUN_STATE and SHA-256 seal; V1 model artifacts remain in their original immutable run.",
        "failure_policy": "Any input drift, identity mismatch, forbidden read, evaluator error or unmet unchanged gate seals FAIL. No retraining, threshold change or metric change.",
    })
    card["source"].update({
        "system_predecessor_run": TRAINING,
        "system_predecessor_status": "FAIL_GSE_SPATIAL_EVENT_CENTER_TRAINING_V1",
        "system_predecessor_reuse": "READ_ONLY_COMPLETED_THREE_SEED_MODELS_AND_SELECTION_OUTPUTS; NO CACHE_OR_PARTIAL_STATE",
    })
    write_json(CARD_PATH, card)

    inputs = [
        f"{TRAINING}/RUN_STATE.json", f"{TRAINING}/metrics/summary.json", f"{TRAINING}/artifacts/evidence_sha256.txt",
        f"{ACTION}/RUN_STATE.json", f"{ACTION}/artifacts/evidence_sha256.txt", f"{ACTION}/scratch/action_set_cache/traversal_id.npy",
        f"{CENTER}/RUN_STATE.json", f"{CENTER}/artifacts/evidence_sha256.txt", f"{CENTER}/artifacts/teacher/event_center_teacher.npz",
        f"{PROJECTION}/RUN_STATE.json", f"{PROJECTION}/artifacts/evidence_sha256.txt", f"{PROJECTION}/artifacts/projection/event_center_projection.npz",
    ]
    for seed in (0, 1, 2):
        inputs.extend([
            f"{TRAINING}/artifacts/models/seed{seed}/best.pt",
            f"{TRAINING}/artifacts/models/seed{seed}/summary.json",
            f"{TRAINING}/artifacts/models/seed{seed}/selection_outputs.npz",
        ])
    tools = {
        "data_card": str(CARD_PATH.relative_to(PROJECT_ROOT)),
        "evaluator": "tools/v3/evaluate_gse_spatial_event_center_ensemble_v1.py",
        "trainer_metric_helper": "tools/v3/train_gse_spatial_event_center_v1.py",
        "runner": "tools/v3/run_gse_spatial_event_center_qualification_v1r.py",
        "freezer": "tools/v3/freeze_gse_spatial_event_center_qualification_spec_v1r.py",
        "preflight": "tools/v3/preflight.py", "create_run": "tools/v3/create_run.py",
    }
    spec = {
        "schema_version": "v3_run_spec_v1", "gate": 3, "execution_phase": 3,
        "date": "20260828", "slug": "gse_spatial_event_center_qualification_v1r", "seed": 0,
        "operation": "checkpoint_selection",
        "question": "Do the already sealed three spatial decoders pass every unchanged C07-C08 event-center association gate?",
        "method": card["methods"]["main"], "baseline": card["methods"]["baseline"],
        "fallback": card["methods"]["fallback"],
        "data_card": str(CARD_PATH.relative_to(PROJECT_ROOT)),
        "config_path": str(CARD_PATH.relative_to(PROJECT_ROOT)),
        "user_authorization": approval,
        "hyperparameters": {"seeds": [0, 1, 2], "ensemble": "arithmetic mean", "new_training_steps": 0, "gates_changed": False},
        "expected_counts": {"models": 3, "selection_rows": 8839, "new_optimizer_steps": 0, "c09_worlds_read": 0, "c10_worlds_read": 0, "mtare_worlds_read": 0},
        "acceptance_criteria": list(card["acceptance"].values()),
        "estimated_cost": card["estimated_cost"],
        "expected_evidence": ["Ensemble selection output and unchanged gate metrics.", "Source integrity, raw log, RUN_STATE and SHA-256 seal."],
        "frozen_inputs": {relative: _sha(PROJECT_ROOT / relative) for relative in inputs},
        "frozen_tools": {name: {"path": relative, "sha256": _sha(PROJECT_ROOT / relative)} for name, relative in tools.items()},
        "command": [
            "/usr/bin/timeout", "--signal=INT", "--kill-after=10s", "300s", PYTHON,
            "tools/v3/run_gse_spatial_event_center_qualification_v1r.py",
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
