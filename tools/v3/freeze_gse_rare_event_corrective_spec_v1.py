#!/usr/bin/env python3
"""Freeze the one-time three-seed rare-event corrective training spec."""

from __future__ import annotations

import hashlib
import json

from _bootstrap import PROJECT_ROOT


RUN_ID = "gate3_20260826_gse_rare_event_corrective_training_v1_seed0"
SPEC = PROJECT_ROOT / "configs/v3/gate3/gse_rare_event_corrective_training_v1.json"
CARD = "configs/v3/gate3/data_cards/gse_rare_event_corrective_training_v1.json"
VERIFIER = "results/gate3_semantics/gate3_20260826_gse_exit_token_association_corrective_v2_seed0"
DATASET = "results/gate2_representation/gate2_20260824_gse_deduplicated_dataset_export_v1_seed0"
TOOLS = {
    "data_card": CARD,
    "model": "src/mtare_topo/representation/gse_rare_event_corrective.py",
    "trainer": "tools/v3/train_gse_rare_event_corrective_v1.py",
    "runner": "tools/v3/run_gse_rare_event_corrective_training_v1.py",
    "dataset_reader": "src/mtare_topo/data/gse_training_dataset.py",
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
        raise RuntimeError("rare-event corrective spec already exists")
    inputs = [
        f"{VERIFIER}/RUN_STATE.json",
        f"{VERIFIER}/metrics/summary.json",
        f"{VERIFIER}/artifacts/evidence_sha256.txt",
        f"{VERIFIER}/artifacts/pair_cache/pairs.npz",
        f"{DATASET}/RUN_STATE.json",
        f"{DATASET}/metrics/summary.json",
        f"{DATASET}/artifacts/evidence_sha256.txt",
        f"{DATASET}/artifacts/sequence_manifest.jsonl"
    ]
    for seed in (0, 1, 2):
        inputs.extend([
            f"{VERIFIER}/artifacts/models/seed{seed}/frozen_observation_features.npy",
            f"{VERIFIER}/artifacts/models/seed{seed}/frozen_exit_token_outputs.npz"
        ])
    run_dir = PROJECT_ROOT / "results/gate3_semantics" / RUN_ID
    command = [
        "/usr/bin/systemd-inhibit", "--what=sleep:shutdown",
        "--why=GSE rare-event corrective training", "--mode=block",
        "/usr/bin/timeout", "--signal=INT", "--kill-after=300s", "7500s",
        "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python",
        "tools/v3/run_gse_rare_event_corrective_training_v1.py",
        "--spec", str(SPEC), "--run-dir", str(run_dir)
    ]
    spec = {
        "schema_version": "v3_run_spec_v1",
        "slug": "gse_rare_event_corrective_training_v1",
        "date": "20260826",
        "gate": 3,
        "execution_phase": 3,
        "operation": "training",
        "question": "Can an identity-balanced residual decoder recover turn and geometry-transition identities from frozen causal geometry-semantic features without sacrificing open-set safety?",
        "method": "Three deterministic 292-128-64-5 residual heads over mean/std of three frozen 146D features; event-class then structural-identity balanced sampling; zero backbone update.",
        "baseline": "Frozen equal-weight event probabilities: C07-C08 macro-F1 0.681994; sealed event-node product precision/false/recall 0.990045/0.009955/0.493902 with zero correct-class rare identity coverage.",
        "data_card": CARD,
        "config_path": CARD,
        "seed": 0,
        "working_directory": str(PROJECT_ROOT),
        "estimated_cost": {"wall_time_hours": 1.0, "disk_gb": 1.0, "compute": "Three small CUDA heads, 40 epochs each, about 33300 optimizer steps; zero backbone/LiDAR/C09."},
        "expected_counts": {
            "fit_worlds": 60,
            "fit_observations": 142184,
            "selection_worlds": 20,
            "selection_observations": 45942,
            "seeds": 3,
            "backbone_optimizer_steps": 0,
            "c09_worlds_read": 0,
            "strict_test_worlds_read": 0,
            "mtare_worlds_read": 0
        },
        "hyperparameters": {
            "epochs": 40,
            "batch_size": 512,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "optimizer": "AdamW",
            "seeds": [0, 1, 2]
        },
        "acceptance_criteria": [
            "Exact 60/20 world and 142184/45942 observation partition with all five frozen event counts and identity counts.",
            "C07-C08 ensemble event macro-F1>=0.7319938212.",
            "At one selected structural threshold: precision>=0.98, false accept<=0.01, recall>=0.40 and all ten families nonempty with precision>=0.95/recall>=0.10.",
            "Correct-class identity coverage junction/terminal>=0.90 and turn/geometry-transition>=0.40.",
            "Zero backbone update and zero C09/C10/M-TARE read; three checkpoints, curves, outputs, environment, logs, RUN_STATE and full seal."
        ],
        "expected_evidence": [
            "Three best checkpoints and epoch curves, ensemble selection outputs, class confusion/F1, identity coverage, structural operating point, hashes, environment, raw log, RUN_STATE and seal."
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
            "approved_at": "2026-08-26T20:20:00+08:00",
            "scope": "One immutable C01-C06 fit/C07-C08 selection three-seed rare-event corrective training.",
            "confirmation_reference": "Standing authorization for autonomous GSE-Graph paper execution without repeated approval prompts."
        }
    }
    SPEC.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(SPEC.relative_to(PROJECT_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
