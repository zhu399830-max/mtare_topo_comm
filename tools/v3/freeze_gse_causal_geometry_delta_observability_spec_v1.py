#!/usr/bin/env python3
"""Freeze the C01-C08 causal geometry-delta audit card and one-shot run spec."""

from __future__ import annotations

import hashlib
from pathlib import Path

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, write_json


RUN_ID = "gate3_20260827_gse_causal_geometry_delta_observability_v1_seed0"
CARD_PATH = PROJECT_ROOT / "configs/v3/gate3/data_cards/gse_causal_geometry_delta_observability_v1.json"
SPEC_PATH = PROJECT_ROOT / "configs/v3/gate3/gse_causal_geometry_delta_observability_v1.json"
SOURCE_CARD = PROJECT_ROOT / "configs/v3/gate3/data_cards/gse_corrected_causal_event_training_v1.json"
TEACHER = "results/gate2_representation/gate2_20260827_gse_corrected_causal_teacher_manifest_v1r_seed0"
VERIFIER = "results/gate3_semantics/gate3_20260826_gse_exit_token_association_corrective_v2_seed0"
PREDECESSOR = "results/gate3_semantics/gate3_20260827_gse_corrected_causal_event_training_v1_seed0"
PYTHON = "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    source_card = load_json(SOURCE_CARD)
    card = {
        "schema_version": "gse_read_only_audit_card_v1",
        "card_id": "gse_causal_geometry_delta_observability_v1",
        "status": "APPROVED_FOR_ONE_IMMUTABLE_GSE_CAUSAL_GEOMETRY_DELTA_OBSERVABILITY_V1",
        "purpose": "Determine whether five-frame LiDAR features expose cross-world causal width, height, slope and curvature changes strongly enough to justify an explicit geometry-delta multitask representation.",
        "approval": {
            "status": "APPROVED",
            "approved_by": "user",
            "approved_at": "2026-08-27T20:00:00+08:00",
            "scope": "One immutable read-only C01-C08 observability audit; no training, checkpoint selection, threshold calibration, C09/C10 or M-TARE access.",
            "confirmation_reference": "User instructed Codex to keep working and automatically choose the evidence-supported optimal in-scope option without requesting routine replies or approvals.",
        },
        "worlds": source_card["worlds"],
        "sampling": {
            "raw_unique_lidar_frames": 252430,
            "causal_observations": 188126,
            "fit_observations": 142184,
            "selection_observations": 45942,
            "causal_lag_steps": 4,
            "geometry_valid_lag_pairs_expected": 122765,
            "fit_geometry_valid_lag_pairs_expected": 92845,
            "selection_geometry_valid_lag_pairs_expected": 29920,
            "fit_transition_observations": 791,
            "selection_transition_observations": 240,
            "fit_transition_identities": 59,
            "selection_transition_identities": 17,
            "spatial_interval_m": 1.0,
            "independent_sampling_unit": "persistent bidirectional causal change-point identity; frame-level AUC is development observability evidence and must be paired with identity coverage.",
        },
        "teacher": {
            "source": "Sealed corrected causal Teacher V1R: TNG event identity plus spline/mesh width, height, slope and curvature; both current and four-step-past rows must have geometry_valid=true.",
            "student_observation": "Three sealed 146-D frozen observation arrays produced from current plus four strictly past LiDAR frames. Only columns 8:12 are decoded to metric geometry using fixed scales [30,30,45,0.1].",
            "invalid_geometry_policy": "Exclude a lag pair if either endpoint has geometry_valid=false or a nonfinite geometry field; never impute zero.",
        },
        "split": {
            "fit": "C01-C06, 60 worlds, used only to set the corridor 99th-percentile diagnostic threshold.",
            "selection": "C07-C08, 20 disjoint worlds, used for pre-registered observability and transfer gates.",
            "strict_test": "C09/C10 and all M-TARE worlds remain unread.",
            "leakage_audit": "No future frame, GT identity, geometry Teacher or world ID is a model input. Identity is used only to count selection coverage after scoring.",
        },
        "method": {
            "main": "Signed four-step delta of frozen predicted width, height, slope and curvature; transition-vs-corridor observability is summarized by ROC-AUC. Three-seed metric predictions are averaged before computing the main score.",
            "baseline": "The same delta computed from objective Teacher geometry provides a data/Teacher ceiling; the failed corrected-label pure classification head is preserved as the predecessor failure.",
            "fallback": "If frozen features fail AUC or transfer-gap gates, do not train a frozen decoder; perform one bounded last-encoder-block fine-tune design instead.",
        },
        "acceptance": {
            "teacher_selection_auc_min": 0.70,
            "predicted_mean_selection_auc_min": 0.65,
            "predicted_mean_fit_selection_auc_gap_max": 0.05,
            "low_fpr_note": "Fit-only 1% corridor FPR transfer, transition recall and identity coverage are mandatory diagnostics but not pass criteria; their expected failure motivates joint learning rather than scalar thresholding.",
        },
        "cost": {"compute": "CPU-only array audit; zero inference/training", "wall_time_hours": 0.1, "disk_gb": 0.25, "host_ram_gb": 4},
        "failure_policy": "Any source/hash/count/index/split drift, nonfinite accepted input, C09/C10/M-TARE access, model update or unmet observability gate seals FAIL. Do not change the Teacher or acceptance thresholds.",
        "retention": "Keep summary CSV/JSON, PNG/PDF/SVG motivation figure, environment, command, log, RUN_STATE and SHA-256 seal for the paper development record.",
    }
    write_json(CARD_PATH, card)

    inputs = [
        f"{TEACHER}/RUN_STATE.json",
        f"{TEACHER}/metrics/summary.json",
        f"{TEACHER}/artifacts/evidence_sha256.txt",
        f"{TEACHER}/artifacts/teacher_observations.jsonl",
        f"{VERIFIER}/RUN_STATE.json",
        f"{VERIFIER}/metrics/summary.json",
        f"{VERIFIER}/artifacts/evidence_sha256.txt",
        f"{VERIFIER}/artifacts/pair_cache/pairs.npz",
        f"{PREDECESSOR}/RUN_STATE.json",
        f"{PREDECESSOR}/metrics/summary.json",
        f"{PREDECESSOR}/artifacts/evidence_sha256.txt",
    ]
    for seed in (0, 1, 2):
        inputs.append(f"{VERIFIER}/artifacts/models/seed{seed}/frozen_observation_features.npy")
    tool_paths = {
        "audit_card": str(CARD_PATH.relative_to(PROJECT_ROOT)),
        "audit_contract": "src/mtare_topo/evaluation/gse_causal_geometry_delta.py",
        "executor": "tools/v3/execute_gse_causal_geometry_delta_observability_v1.py",
        "runner": "tools/v3/run_gse_causal_geometry_delta_observability_v1.py",
        "freezer": "tools/v3/freeze_gse_causal_geometry_delta_observability_spec_v1.py",
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
        "slug": "gse_causal_geometry_delta_observability_v1",
        "seed": 0,
        "operation": "audit",
        "question": "Do frozen five-frame GSE features expose transferable causal geometry change strongly enough to justify explicit geometry-delta multitask learning?",
        "method": card["method"]["main"],
        "baseline": card["method"]["baseline"],
        "fallback": card["method"]["fallback"],
        "audit_card": str(CARD_PATH.relative_to(PROJECT_ROOT)),
        "config_path": str(CARD_PATH.relative_to(PROJECT_ROOT)),
        "user_authorization": {
            "status": "APPROVED",
            "approved_by": "user",
            "approved_at": "2026-08-27T20:00:00+08:00",
            "scope": card["approval"]["scope"],
            "confirmation_reference": card["approval"]["confirmation_reference"],
        },
        "acceptance_criteria": [
            "Exact 80 C01-C08 worlds, 188126 observations, 122765 geometry-valid four-step lag pairs, and 17 selection transition identities.",
            "Teacher selection transition-vs-corridor AUC>=0.70.",
            "Three-seed mean frozen prediction selection AUC>=0.65 and absolute fit-selection AUC gap<=0.05.",
            "Zero optimizer/inference step and zero C09/C10/M-TARE access; preserve the 1% FPR transfer diagnostic without converting it into a pass gate.",
        ],
        "expected_counts": {
            "worlds": 80,
            "observations": 188126,
            "fit_observations": 142184,
            "selection_observations": 45942,
            "valid_lag_pairs": 122765,
            "fit_valid_lag_pairs": 92845,
            "selection_valid_lag_pairs": 29920,
            "selection_transition_identities": 17,
            "optimizer_steps": 0,
            "model_inference_frames": 0,
            "c09_worlds_read": 0,
            "strict_test_worlds_read": 0,
            "mtare_worlds_read": 0,
        },
        "expected_evidence": [
            "AUC, component delta MAE, fit-only 1% FPR threshold transfer, identity coverage, paper motivation figure in PNG/PDF/SVG, source integrity, environment, log, RUN_STATE and complete SHA-256 seal."
        ],
        "estimated_cost": {"compute": "CPU-only; zero training/inference", "disk_gb": 0.25, "wall_time_hours": 0.1},
        "frozen_inputs": {path: _sha(PROJECT_ROOT / path) for path in inputs},
        "frozen_tools": {
            name: {"path": path, "sha256": _sha(PROJECT_ROOT / path)}
            for name, path in tool_paths.items()
        },
        "working_directory": str(PROJECT_ROOT),
        "command": [
            "/usr/bin/systemd-inhibit",
            "--what=sleep:shutdown",
            "--why=GSE causal geometry-delta observability audit",
            "--mode=block",
            "/usr/bin/timeout",
            "--signal=INT",
            "--kill-after=30s",
            "660s",
            PYTHON,
            "tools/v3/run_gse_causal_geometry_delta_observability_v1.py",
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
