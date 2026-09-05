#!/usr/bin/env python3
"""Freeze the C07-C08 spatial event-center residual audit."""

from __future__ import annotations

import hashlib

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import write_json


RUN_ID = "gate3_20260828_gse_spatial_center_residual_audit_v1_seed0"
SPEC_PATH = PROJECT_ROOT / "configs/v3/gate3/gse_spatial_center_residual_audit_v1.json"
PYTHON = "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python"
TRAINING = "results/gate3_semantics/gate3_20260828_gse_spatial_event_center_training_v1_seed0"
QUALIFICATION = "results/gate3_semantics/gate3_20260828_gse_spatial_event_center_qualification_v1r_seed0"
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
    inputs = [
        f"{TRAINING}/RUN_STATE.json", f"{TRAINING}/metrics/summary.json", f"{TRAINING}/artifacts/evidence_sha256.txt",
        f"{QUALIFICATION}/RUN_STATE.json", f"{QUALIFICATION}/metrics/summary.json", f"{QUALIFICATION}/artifacts/evidence_sha256.txt",
        f"{QUALIFICATION}/metrics/ensemble/ensemble_selection_outputs.npz", f"{QUALIFICATION}/metrics/ensemble/summary.json",
        f"{ACTION}/RUN_STATE.json", f"{ACTION}/artifacts/evidence_sha256.txt", f"{ACTION}/scratch/action_set_cache/traversal_id.npy",
        f"{CENTER}/RUN_STATE.json", f"{CENTER}/artifacts/evidence_sha256.txt", f"{CENTER}/artifacts/teacher/event_center_teacher.npz",
        f"{PROJECTION}/RUN_STATE.json", f"{PROJECTION}/artifacts/evidence_sha256.txt", f"{PROJECTION}/artifacts/projection/event_center_projection.npz",
    ]
    for seed in (0, 1, 2):
        inputs.append(f"{TRAINING}/artifacts/models/seed{seed}/selection_outputs.npz")
    tools = {
        "residual_metrics": "src/mtare_topo/evaluation/gse_spatial_center_residual.py",
        "center_teacher": "src/mtare_topo/teacher/gse_event_center_teacher.py",
        "executor": "tools/v3/execute_gse_spatial_center_residual_audit_v1.py",
        "runner": "tools/v3/run_gse_spatial_center_residual_audit_v1.py",
        "freezer": "tools/v3/freeze_gse_spatial_center_residual_audit_spec_v1.py",
        "preflight": "tools/v3/preflight.py", "create_run": "tools/v3/create_run.py",
    }
    spec = {
        "schema_version": "v3_run_spec_v1", "gate": 3, "execution_phase": 3,
        "date": "20260828", "slug": "gse_spatial_center_residual_audit_v1", "seed": 0,
        "operation": "audit",
        "user_authorization": {
            "status": "APPROVED", "approved_by": "user-standing-authorization",
            "approved_at": "2026-08-28T11:25:00+08:00",
            "scope": "One immutable read-only C07-C08 spatial-center residual audit; zero training and zero C09/C10/M-TARE.",
            "confirmation_reference": "User instructed automatic best-choice execution without routine approval prompts.",
        },
        "question": "Is the remaining 0.004029 within-4m gain deficit caused by frozen longitudinal error, learned transverse error, specific event/identity tails or seed disagreement?",
        "method": "Enumerate all 144533 cross-traversal pairs for 272 C07-C08 node identities and compare scalar, full spatial, predicted-long/oracle-transverse, oracle-long/predicted-transverse and oracle centers. Stratify by event and identity; measure seed disagreement. Coordinate-median and row-medoid aggregations are diagnostic only.",
        "baseline": "Sealed scalar zero-transverse projection and sealed arithmetic-mean spatial ensemble.",
        "fallback": "None. This audit changes no model, checkpoint, threshold, association radius or graph policy.",
        "hyperparameters": {"association_radius_m": 4.0, "selection_rows": 8839, "identities": 272, "pairs": 144533, "new_optimizer_steps": 0},
        "expected_counts": {"selection_rows": 8839, "identities": 272, "cross_traversal_pairs": 144533, "new_optimizer_steps": 0, "c09_worlds_read": 0, "c10_worlds_read": 0, "mtare_worlds_read": 0},
        "acceptance_criteria": [
            "Exact 8839 rows, 272 multi-traversal identities and 144533 pairs.",
            "Exact scalar/full metrics reproduce sealed qualification.",
            "All four component counterfactuals, event strata, identity tail and seed-disagreement metrics are finite.",
            "One complete paper-grade residual figure, raw identity JSONL, logs, source integrity and seal; zero model updates or forbidden reads.",
        ],
        "estimated_cost": {"compute": "CPU-only sealed-output audit", "wall_time_hours": 0.1, "disk_gb": 0.1, "host_ram_gb": 4, "gpu_memory_gb": 0},
        "expected_evidence": ["Component/event/identity residual metrics and top-20 failure tail.", "Paper-grade four-panel residual figure and manifest.", "Command, raw log, source integrity, RUN_STATE and SHA-256 seal."],
        "frozen_inputs": {relative: _sha(PROJECT_ROOT / relative) for relative in inputs},
        "frozen_tools": {name: {"path": relative, "sha256": _sha(PROJECT_ROOT / relative)} for name, relative in tools.items()},
        "command": [
            "/usr/bin/timeout", "--signal=INT", "--kill-after=10s", "300s", PYTHON,
            "tools/v3/run_gse_spatial_center_residual_audit_v1.py",
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
