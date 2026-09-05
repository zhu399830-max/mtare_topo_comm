#!/usr/bin/env python3
"""Freeze the one-time open-set node-matchability calibration spec."""

from __future__ import annotations

import hashlib
import json

from _bootstrap import PROJECT_ROOT


RUN_ID = "gate4_20260826_gse_node_matchability_ensemble_calibration_v1_seed0"
SPEC = PROJECT_ROOT / "configs/v3/gate4/gse_node_matchability_ensemble_calibration_v1.json"
CARD = "configs/v3/gate4/data_cards/gse_node_matchability_ensemble_calibration_v1.json"
SOURCE = "results/gate3_semantics/gate3_20260826_gse_exit_token_association_corrective_v2_seed0"
TOOLS = {
    "data_card": CARD,
    "evaluator": "tools/v3/evaluate_gse_node_matchability_ensemble_v1.py",
    "runner": "tools/v3/run_gse_node_matchability_ensemble_calibration_v1.py",
    "verifier": "src/mtare_topo/representation/gse_exit_token_association.py",
    "selector": "src/mtare_topo/representation/gse_open_set_association.py",
    "evidence_integrity": "src/mtare_topo/evaluation/gse_evidence_integrity.py",
    "governance": "src/mtare_topo/governance.py",
    "preflight": "tools/v3/preflight.py",
    "create_run": "tools/v3/create_run.py",
}


def _sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    if SPEC.exists():
        raise RuntimeError("node-matchability spec already exists")
    relative_inputs = [
        f"{SOURCE}/RUN_STATE.json",
        f"{SOURCE}/metrics/summary.json",
        f"{SOURCE}/artifacts/evidence_sha256.txt",
        f"{SOURCE}/artifacts/pair_cache/pairs.npz",
    ]
    for seed in (0, 1, 2):
        relative_inputs.extend([
            f"{SOURCE}/artifacts/models/seed{seed}/frozen_observation_features.npy",
            f"{SOURCE}/artifacts/models/seed{seed}/normalization.npz",
            f"{SOURCE}/artifacts/models/seed{seed}/best.pt",
        ])
    run_dir = PROJECT_ROOT / "results/gate4_topology" / RUN_ID
    command = [
        "/usr/bin/systemd-inhibit", "--what=sleep:shutdown",
        "--why=GSE node matchability calibration", "--mode=block",
        "/usr/bin/timeout", "--signal=INT", "--kill-after=60s", "660s",
        "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python",
        "tools/v3/run_gse_node_matchability_ensemble_calibration_v1.py",
        "--spec", str(SPEC), "--run-dir", str(run_dir),
    ]
    spec = {
        "schema_version": "v3_run_spec_v1",
        "slug": "gse_node_matchability_ensemble_calibration_v1",
        "date": "20260826", "gate": 4, "execution_phase": 4,
        "operation": "threshold_calibration",
        "question": "Can the three frozen matchability heads provide a non-vacuous open-set structural-node gate before C09 topology replay?",
        "method": "Float64 equal-weight mean of three frozen V2 matchability sigmoid scores; select one threshold on all 45942 C07-C08 observations under unchanged aggregate and per-family open-set gates.",
        "baseline": "Three individual frozen matchability scores are diagnostic only; no seed or learned ensemble weight is selected.",
        "data_card": CARD, "config_path": CARD, "seed": 0,
        "working_directory": str(PROJECT_ROOT),
        "estimated_cost": {"wall_time_hours": 0.17, "disk_gb": 0.1, "compute": "CPU matchability-head inference over 188126 observations x 3; zero backbone/optimizer/C09."},
        "expected_counts": {
            "fit_observations": 142184, "fit_positive": 39310, "fit_negative": 102874,
            "selection_observations": 45942, "selection_positive": 13693,
            "selection_negative": 32249, "seeds": 3,
            "optimizer_steps": 0, "backbone_optimizer_steps": 0,
            "c09_worlds_read": 0, "strict_test_worlds_read": 0, "mtare_worlds_read": 0,
        },
        "acceptance_criteria": [
            "Exact 188126 observation identity and 142184/45942 partition with 39310/13693 positives.",
            "Only equal-weight float64 mean of all three frozen seed matchability scores; no updates or seed selection.",
            "Precision>=0.98, false accept<=0.01 and recall>=0.25; all ten families nonempty with precision>=0.95 and recall>=0.10.",
            "Zero C09/C10/M-TARE reads and zero optimizer/backbone steps; source unchanged and full SHA-256 seal.",
        ],
        "expected_evidence": [
            "Selection score archive, exact operating point, aggregate/per-family metrics, hashes, environment, log, RUN_STATE and seal."
        ],
        "frozen_inputs": {relative: _sha(PROJECT_ROOT / relative) for relative in relative_inputs},
        "frozen_tools": {
            name: {"path": relative, "sha256": _sha(PROJECT_ROOT / relative)}
            for name, relative in TOOLS.items()
        },
        "command": command,
        "user_authorization": {
            "status": "APPROVED", "approved_by": "user",
            "approved_at": "2026-08-26T17:15:00+08:00",
            "scope": "One immutable C01-C08-only node-matchability threshold calibration.",
            "confirmation_reference": "Standing authorization for autonomous GSE-Graph paper corrective work without repeated approval prompts.",
        },
    }
    SPEC.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(SPEC.relative_to(PROJECT_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
