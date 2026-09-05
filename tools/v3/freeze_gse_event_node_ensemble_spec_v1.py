#!/usr/bin/env python3
"""Freeze the one-time C01-C08 GSE event-node ensemble calibration spec."""

from __future__ import annotations

import hashlib
import json

from _bootstrap import PROJECT_ROOT


RUN_ID = "gate4_20260826_gse_event_node_ensemble_calibration_v1_seed0"
SPEC = PROJECT_ROOT / "configs/v3/gate4/gse_event_node_ensemble_calibration_v1.json"
CARD = "configs/v3/gate4/data_cards/gse_event_node_ensemble_calibration_v1.json"
VERIFIER = "results/gate3_semantics/gate3_20260826_gse_exit_token_association_corrective_v2_seed0"
NODE = "results/gate4_topology/gate4_20260826_gse_node_matchability_ensemble_calibration_v1_seed0"
TOOLS = {
    "data_card": CARD,
    "evaluator": "tools/v3/evaluate_gse_event_node_ensemble_v1.py",
    "runner": "tools/v3/run_gse_event_node_ensemble_calibration_v1.py",
    "selector": "src/mtare_topo/representation/gse_open_set_association.py",
    "evidence_integrity": "src/mtare_topo/evaluation/gse_evidence_integrity.py",
    "governance": "src/mtare_topo/governance.py",
    "preflight": "tools/v3/preflight.py",
    "create_run": "tools/v3/create_run.py"
}


def _sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    if SPEC.exists():
        raise RuntimeError("event-node ensemble spec already exists")
    inputs = [
        f"{VERIFIER}/RUN_STATE.json",
        f"{VERIFIER}/metrics/summary.json",
        f"{VERIFIER}/artifacts/evidence_sha256.txt",
        f"{NODE}/RUN_STATE.json",
        f"{NODE}/metrics/summary.json",
        f"{NODE}/artifacts/evidence_sha256.txt",
        f"{NODE}/artifacts/calibration/node_matchability_selection_outputs.npz"
    ]
    for seed in (0, 1, 2):
        inputs.extend([
            f"{VERIFIER}/artifacts/models/seed{seed}/frozen_observation_features.npy",
            f"{VERIFIER}/artifacts/models/seed{seed}/frozen_exit_token_outputs.npz"
        ])
    run_dir = PROJECT_ROOT / "results/gate4_topology" / RUN_ID
    command = [
        "/usr/bin/systemd-inhibit", "--what=sleep:shutdown",
        "--why=GSE event-node ensemble calibration", "--mode=block",
        "/usr/bin/timeout", "--signal=INT", "--kill-after=60s", "240s",
        "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python",
        "tools/v3/run_gse_event_node_ensemble_calibration_v1.py",
        "--spec", str(SPEC), "--run-dir", str(run_dir)
    ]
    spec = {
        "schema_version": "v3_run_spec_v1",
        "slug": "gse_event_node_ensemble_calibration_v1",
        "date": "20260826",
        "gate": 4,
        "execution_phase": 4,
        "operation": "threshold_calibration",
        "question": "Can frozen three-seed structural probability improve open-set node recall while retaining safe precision before any new C09 replay?",
        "method": "Float64 equal-weight mean of three frozen structural probabilities multiplied by the frozen equal-weight node-matchability score; one C07-C08 threshold.",
        "baseline": "Frozen node-matchability ensemble alone at 0.982292910416921.",
        "data_card": CARD,
        "config_path": CARD,
        "seed": 0,
        "working_directory": str(PROJECT_ROOT),
        "estimated_cost": {"wall_time_hours": 0.03, "disk_gb": 0.1, "compute": "CPU read-only score composition over 45942 observations; zero inference or optimizer."},
        "expected_counts": {
            "selection_worlds": 20,
            "selection_observations": 45942,
            "selection_positive": 13693,
            "selection_negative": 32249,
            "seeds": 3,
            "optimizer_steps": 0,
            "model_inference_frames": 0,
            "c09_worlds_read": 0,
            "strict_test_worlds_read": 0,
            "mtare_worlds_read": 0
        },
        "acceptance_criteria": [
            "Exact C07-C08 population 45942 = 13693 positive + 32249 negative across ten families.",
            "Only the predeclared equal-weight structural probability times frozen node score; no learned weights, seed selection or model updates.",
            "Precision>=0.98, false accept<=0.01, recall>=0.40, plus inherited non-vacuous ten-family precision/recall gates.",
            "Zero C09/C10/M-TARE read, zero inference/optimizer, source unchanged and full SHA-256 seal."
        ],
        "expected_evidence": [
            "Selection score archive, unique threshold, aggregate/per-family metrics, hashes, environment, raw log, RUN_STATE and seal."
        ],
        "frozen_inputs": {relative: _sha(PROJECT_ROOT / relative) for relative in inputs},
        "frozen_tools": {
            name: {"path": relative, "sha256": _sha(PROJECT_ROOT / relative)}
            for name, relative in TOOLS.items()
        },
        "command": command,
        "user_authorization": {
            "status": "APPROVED",
            "approved_by": "user",
            "approved_at": "2026-08-26T19:15:00+08:00",
            "scope": "One immutable C01-C08-only event-node ensemble calibration.",
            "confirmation_reference": "Standing authorization for autonomous GSE-Graph paper execution without repeated approval prompts."
        }
    }
    SPEC.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(SPEC.relative_to(PROJECT_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
