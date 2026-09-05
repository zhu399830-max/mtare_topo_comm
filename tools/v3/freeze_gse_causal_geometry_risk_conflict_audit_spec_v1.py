#!/usr/bin/env python3
"""Freeze the C01-C08 risk-conflict audit card and one-shot run spec."""

from __future__ import annotations

import hashlib
from pathlib import Path

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, write_json


RUN_ID = "gate3_20260827_gse_causal_geometry_risk_conflict_audit_v1_seed0"
CARD_PATH = PROJECT_ROOT / "configs/v3/gate3/data_cards/gse_causal_geometry_risk_conflict_audit_v1.json"
SPEC_PATH = PROJECT_ROOT / "configs/v3/gate3/gse_causal_geometry_risk_conflict_audit_v1.json"
SOURCE_CARD = PROJECT_ROOT / "configs/v3/gate3/data_cards/gse_corrected_causal_event_training_v1.json"
DATASET = "results/gate2_representation/gate2_20260824_gse_deduplicated_dataset_export_v1_seed0"
TEACHER = "results/gate2_representation/gate2_20260827_gse_corrected_causal_teacher_manifest_v1r_seed0"
VERIFIER = "results/gate3_semantics/gate3_20260826_gse_exit_token_association_corrective_v2_seed0"
OLD_DIRECTIONAL = "results/gate3_semantics/gate3_20260826_gse_directional_structural_event_training_v1_seed0"
FROZEN_MULTITASK = "results/gate3_semantics/gate3_20260827_gse_causal_geometry_delta_multitask_training_v1_seed0"
LAST_BLOCK = "results/gate3_semantics/gate3_20260827_gse_causal_geometry_delta_last_block_training_v1_seed0"
PYTHON = "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    # Do not publish even the card/spec pair until the conditional predecessor
    # is fully sealed.  This makes an accidental early freezer invocation a
    # read-only failure rather than a partially frozen experiment.
    predecessor_required = (
        PROJECT_ROOT / LAST_BLOCK / "RUN_STATE.json",
        PROJECT_ROOT / LAST_BLOCK / "metrics/summary.json",
        PROJECT_ROOT / LAST_BLOCK / "artifacts/evidence_sha256.txt",
        PROJECT_ROOT / LAST_BLOCK / "artifacts/training/ensemble_selection_outputs.npz",
    )
    missing = [str(path) for path in predecessor_required if not path.is_file()]
    if missing:
        raise RuntimeError(
            "risk-conflict audit cannot freeze before the bounded last-block "
            f"predecessor is sealed; missing={missing}"
        )
    source_card = load_json(SOURCE_CARD)
    card = {
        "schema_version": "gse_read_only_audit_card_v1",
        "card_id": "gse_causal_geometry_risk_conflict_audit_v1",
        "status": "APPROVED_FOR_ONE_IMMUTABLE_GSE_CAUSAL_GEOMETRY_RISK_CONFLICT_AUDIT_V1",
        "purpose": "Determine whether zero accepted change identities are caused by causal-history length, the global structural-confidence gate, or scalar geometry risk insufficiency.",
        "approval": {
            "status": "APPROVED",
            "approved_by": "user",
            "approved_at": "2026-08-27T23:30:00+08:00",
            "scope": "One immutable read-only C01-C08 mechanism audit after the bounded last-block failure; no model update, checkpoint selection, C09/C10 or M-TARE access.",
            "confirmation_reference": "User instructed Codex to automatically select the evidence-supported optimal route and continue without requesting routine approvals.",
        },
        "worlds": source_card["worlds"],
        "trajectories": source_card["trajectories"],
        "sampling": {
            "raw_unique_lidar_frames": 252430,
            "causal_observations": 188126,
            "fit_worlds": 60,
            "fit_directed_traversals": 12106,
            "fit_unique_lidar_frames": 190600,
            "fit_observations": 142184,
            "selection_worlds": 20,
            "selection_directed_traversals": 3972,
            "selection_unique_lidar_frames": 61830,
            "selection_observations": 45942,
            "selection_change_frames": 240,
            "selection_change_identities": 17,
            "causal_lags_steps": list(range(2, 13)),
            "spatial_interval_m": 1.0,
            "independent_sampling_unit": "persistent bidirectional structural identity; frame scores are diagnostic only.",
        },
        "teacher": {
            "source": "Sealed corrected causal Teacher V1R with persistent structural identities and objective spline/mesh width, height, slope and curvature.",
            "student_observation": "Three sealed causal frozen feature arrays plus sealed ensemble event outputs from the original, directional, frozen-delta and last-block representations.",
            "invalid_geometry_policy": "A lag pair is excluded if either endpoint is invalid; no imputation, future frame or cross-traversal pair is permitted.",
        },
        "split": {
            "fit": "C01-C06 only; fixes the 99th-percentile corridor risk threshold independently for every declared lag.",
            "selection": "C07-C08 only; measures transfer AUC, false positives and 17-identity coverage without selecting a winning lag.",
            "strict_test": "C09/C10 and every M-TARE world remain unread.",
            "leakage_audit": "World, pose, identity, TNG and future observations are never scoring inputs; identity is used only after scoring for coverage.",
        },
        "leakage_audit": source_card["leakage_audit"],
        "method": {
            "main": "Decompose structural-versus-conditional change confidence; audit fixed causal geometry lags 2-12; transfer fit-only one-percent corridor thresholds; stratify hard-negative tails.",
            "baseline": "Objective Teacher geometry provides an observability ceiling and the original/directional/frozen-delta outputs localize which representation stage suppresses change events.",
            "fallback": "If longer history is informative but scalar risk remains unsafe, implement one low-capacity multivariate geometry-conditioned identity-level risk proof; otherwise revise the causal input/Teacher contract.",
        },
        "acceptance": {
            "audit_completion": "All declared lags and four confidence sources are reproduced with exact population counts and immutable source evidence.",
            "next_method_rule": "Use the predeclared mechanism conclusion; do not tune a winning lag or lower node precision/false-acceptance gates.",
        },
        "cost": {"compute": "CPU-only array audit; zero inference/training", "wall_time_hours": 0.25, "disk_gb": 0.25, "host_ram_gb": 6},
        "failure_policy": "Any source/hash/count/index/split drift, last-block source not sealed FAIL, nonfinite input, model update, or C09/C10/M-TARE access seals FAIL.",
        "retention": "Keep CSV/JSON, paper-ready PNG/PDF/SVG, environment, command, raw log, RUN_STATE and SHA-256 seal.",
    }
    write_json(CARD_PATH, card)

    inputs = []
    for run in (DATASET, TEACHER, VERIFIER, OLD_DIRECTIONAL, FROZEN_MULTITASK, LAST_BLOCK):
        inputs.extend([f"{run}/RUN_STATE.json", f"{run}/metrics/summary.json", f"{run}/artifacts/evidence_sha256.txt"])
    inputs.extend(
        [
            f"{TEACHER}/artifacts/teacher_observations.jsonl",
            f"{VERIFIER}/artifacts/pair_cache/pairs.npz",
            f"{OLD_DIRECTIONAL}/artifacts/training/ensemble_selection_outputs.npz",
            f"{FROZEN_MULTITASK}/artifacts/training/ensemble_selection_outputs.npz",
            f"{LAST_BLOCK}/artifacts/training/ensemble_selection_outputs.npz",
        ]
    )
    for seed in (0, 1, 2):
        inputs.append(f"{VERIFIER}/artifacts/models/seed{seed}/frozen_observation_features.npy")
    tool_paths = {
        "audit_card": str(CARD_PATH.relative_to(PROJECT_ROOT)),
        "audit_contract": "src/mtare_topo/evaluation/gse_causal_geometry_risk_conflict.py",
        "executor": "tools/v3/execute_gse_causal_geometry_risk_conflict_audit_v1.py",
        "runner": "tools/v3/run_gse_causal_geometry_risk_conflict_audit_v1.py",
        "freezer": "tools/v3/freeze_gse_causal_geometry_risk_conflict_audit_spec_v1.py",
        "evidence_integrity": "src/mtare_topo/evaluation/gse_evidence_integrity.py",
        "governance": "src/mtare_topo/governance.py",
        "preflight": "tools/v3/preflight.py",
        "create_run": "tools/v3/create_run.py",
    }
    spec = {
        "schema_version": "v3_run_spec_v1",
        "gate": 3,
        "execution_phase": 3,
        "date": "20260827",
        "slug": "gse_causal_geometry_risk_conflict_audit_v1",
        "seed": 0,
        "operation": "audit",
        "question": "Why do transferable geometry deltas coexist with zero safely accepted change identities?",
        "method": card["method"]["main"],
        "baseline": card["method"]["baseline"],
        "fallback": card["method"]["fallback"],
        "audit_card": str(CARD_PATH.relative_to(PROJECT_ROOT)),
        "config_path": str(CARD_PATH.relative_to(PROJECT_ROOT)),
        "user_authorization": {
            "status": "APPROVED",
            "approved_by": "user",
            "approved_at": card["approval"]["approved_at"],
            "scope": card["approval"]["scope"],
            "confirmation_reference": card["approval"]["confirmation_reference"],
        },
        "acceptance_criteria": [
            "Exact C01-C06 60/142184 and C07-C08 20/45942 populations with exactly 17 corrected change identities.",
            "Reproduce confidence decomposition for four sealed representations and geometry lags 2-12 without selecting a winning lag.",
            "Freeze each fit-only one-percent corridor threshold before selection evaluation and report hard-negative provenance.",
            "Zero optimizer/inference steps and zero C09/C10/M-TARE access; complete PNG/PDF/SVG plus SHA-256 evidence seal.",
        ],
        "expected_counts": {
            "fit_worlds": 60,
            "fit_observations": 142184,
            "selection_worlds": 20,
            "selection_observations": 45942,
            "selection_change_identities": 17,
            "lag_rows": 22,
            "hard_negative_rows": 22,
            "optimizer_steps": 0,
            "model_inference_frames": 0,
            "c09_worlds_read": 0,
            "strict_test_worlds_read": 0,
            "mtare_worlds_read": 0,
        },
        "expected_evidence": [
            "Confidence decomposition, lag risk table, hard-negative provenance, mechanism conclusion, paper-ready figure, environment, raw log, RUN_STATE and complete SHA-256 seal."
        ],
        "estimated_cost": {"compute": "CPU-only; zero training/inference", "disk_gb": 0.25, "wall_time_hours": 0.25},
        "frozen_inputs": {path: _sha(PROJECT_ROOT / path) for path in inputs},
        "frozen_tools": {
            name: {"path": path, "sha256": _sha(PROJECT_ROOT / path)}
            for name, path in tool_paths.items()
        },
        "working_directory": str(PROJECT_ROOT),
        "command": [
            "/usr/bin/systemd-inhibit",
            "--what=sleep:shutdown",
            "--why=GSE causal geometry risk conflict audit",
            "--mode=block",
            "/usr/bin/timeout",
            "--signal=INT",
            "--kill-after=30s",
            "960s",
            PYTHON,
            "tools/v3/run_gse_causal_geometry_risk_conflict_audit_v1.py",
            "--spec",
            str(SPEC_PATH),
            "--run-dir",
            str(PROJECT_ROOT / "results/gate3_semantics" / RUN_ID),
        ],
    }
    write_json(SPEC_PATH, spec)
    print(CARD_PATH.relative_to(PROJECT_ROOT))
    print(SPEC_PATH.relative_to(PROJECT_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
