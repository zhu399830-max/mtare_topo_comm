#!/usr/bin/env python3
"""Freeze the one-time directional structural-event training spec."""

from __future__ import annotations

import hashlib
import json

from _bootstrap import PROJECT_ROOT


RUN_ID = "gate3_20260826_gse_directional_structural_event_training_v1_seed0"
SPEC = PROJECT_ROOT / "configs/v3/gate3/gse_directional_structural_event_training_v1.json"
CARD = "configs/v3/gate3/data_cards/gse_directional_structural_event_training_v1.json"
VERIFIER = "results/gate3_semantics/gate3_20260826_gse_exit_token_association_corrective_v2_seed0"
DATASET = "results/gate2_representation/gate2_20260824_gse_deduplicated_dataset_export_v1_seed0"
TRAINING = "results/gate2_representation/gate2_20260824_gse_graph_three_seed_training_v1r_seed0"
TOOLS = {
    "data_card": CARD,
    "backbone_model": "src/mtare_topo/representation/gse_graph.py",
    "directional_head": "src/mtare_topo/representation/gse_directional_structural_event.py",
    "trainer": "tools/v3/train_gse_directional_structural_event_v1.py",
    "runner": "tools/v3/run_gse_directional_structural_event_training_v1.py",
    "dataset_reader": "src/mtare_topo/data/gse_training_dataset.py",
    "selector": "src/mtare_topo/representation/gse_open_set_association.py",
    "rare_event_metrics": "src/mtare_topo/representation/gse_rare_event_corrective.py",
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
        raise RuntimeError("directional structural-event spec already exists")
    inputs = [
        f"{VERIFIER}/RUN_STATE.json",
        f"{VERIFIER}/metrics/summary.json",
        f"{VERIFIER}/artifacts/evidence_sha256.txt",
        f"{VERIFIER}/artifacts/pair_cache/pairs.npz",
        f"{DATASET}/RUN_STATE.json",
        f"{DATASET}/metrics/summary.json",
        f"{DATASET}/artifacts/evidence_sha256.txt",
        f"{DATASET}/artifacts/sequence_manifest.jsonl",
        f"{TRAINING}/RUN_STATE.json",
        f"{TRAINING}/metrics/summary.json",
        f"{TRAINING}/artifacts/evidence_sha256.txt",
    ]
    for seed in (0, 1, 2):
        inputs.extend([
            f"{VERIFIER}/artifacts/models/seed{seed}/frozen_observation_features.npy",
            f"{TRAINING}/artifacts/models/seed{seed}/best.pt",
        ])
    run_dir = PROJECT_ROOT / "results/gate3_semantics" / RUN_ID
    command = [
        "/usr/bin/systemd-inhibit", "--what=sleep:shutdown",
        "--why=GSE directional structural event training", "--mode=block",
        "/usr/bin/timeout", "--signal=INT", "--kill-after=300s", "33000s",
        "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python",
        "tools/v3/run_gse_directional_structural_event_training_v1.py",
        "--spec", str(SPEC), "--run-dir", str(run_dir),
    ]
    spec = {
        "schema_version": "v3_run_spec_v1",
        "slug": "gse_directional_structural_event_training_v1",
        "date": "20260826",
        "gate": 3,
        "execution_phase": 3,
        "operation": "training",
        "question": "Can azimuth-preserving five-frame geometry evidence recover turn and geometry-transition identities while retaining the fixed one-percent node-generation false-accept contract?",
        "method": "Three frozen-backbone directional heads. Circular change encoder over current/mean-past/delta azimuth features; separate binary structural evidence and conditional four-class event outputs; zero-initialized residuals.",
        "baseline": "Frozen three-seed event mean and sealed event-node product; failed 292D compressed-output corrective retained as an ablation.",
        "data_card": CARD,
        "config_path": CARD,
        "seed": 0,
        "working_directory": str(PROJECT_ROOT),
        "estimated_cost": {"wall_time_hours": 8.0, "disk_gb": 2.0, "compute": "One RTX 5090 D, three heads serially, 36000 head optimizer steps, zero backbone/LiDAR/C09."},
        "expected_counts": {
            "fit_worlds": 60, "fit_observations": 142184,
            "selection_worlds": 20, "selection_observations": 45942,
            "seeds": 3, "optimizer_steps": 36000,
            "backbone_optimizer_steps": 0, "c09_worlds_read": 0,
            "strict_test_worlds_read": 0, "mtare_worlds_read": 0,
        },
        "hyperparameters": {
            "epochs": 12, "draws_per_epoch": 24000, "batch_size": 24,
            "evaluation_batch_size": 48, "evaluate_every": 3,
            "learning_rate": 0.001, "weight_decay": 0.0001,
            "optimizer": "AdamW", "gradient_clip": 5.0, "seeds": [0, 1, 2],
        },
        "acceptance_criteria": [
            "Exact 60/20 world and 142184/45942 observation split with fixed class and identity counts.",
            "C07-C08 ensemble event macro-F1>=0.7319938212.",
            "One selected binary threshold has precision>=0.98, false accept<=0.01, recall>=0.40; all ten families nonempty with precision>=0.95/recall>=0.10.",
            "Correct-class identity coverage junction/terminal>=0.90 and turn/geometry-transition>=0.40.",
            "Exactly 36000 head optimizer steps, zero backbone update and zero C09/C10/M-TARE read; complete immutable evidence and seal.",
        ],
        "expected_evidence": [
            "Three checkpoints, 36 epoch records with 12 selection evaluations, sampling contract, ensemble outputs, event/identity/open-set metrics, environment, raw log, RUN_STATE and SHA-256 seal."
        ],
        "frozen_inputs": {relative: _sha(PROJECT_ROOT / relative) for relative in inputs},
        "frozen_tools": {
            name: {"path": relative, "sha256": _sha(PROJECT_ROOT / relative)}
            for name, relative in TOOLS.items()
        },
        "command": command,
        "user_authorization": {
            "status": "APPROVED", "approved_by": "user",
            "approved_at": "2026-08-26T21:15:00+08:00",
            "scope": "One immutable C01-C06 fit/C07-C08 selection directional structural-event training with frozen GSE backbones.",
            "confirmation_reference": "Standing user authorization for autonomous in-scope GSE-Graph corrective work without repeated approval prompts.",
        },
    }
    SPEC.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(SPEC.relative_to(PROJECT_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

