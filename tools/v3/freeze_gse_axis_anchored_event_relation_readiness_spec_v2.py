#!/usr/bin/env python3
"""Freeze the full-population multi-label method-readiness spec."""
from __future__ import annotations

import hashlib

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, validate_data_card, write_json


RUN_ID = "gate3_20260829_gse_axis_anchored_event_relation_readiness_v2_seed0"
SPEC = PROJECT_ROOT / "configs/v3/gate3/gse_axis_anchored_event_relation_readiness_v2.json"
DATA_CARD = "configs/v3/gate3/data_cards/gse_axis_anchored_event_relation_readiness_v2.json"
PYTHON = "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    if SPEC.exists():
        raise RuntimeError("axis-anchored readiness V2 spec already exists")
    card = load_json(PROJECT_ROOT / DATA_CARD)
    validation = validate_data_card(card)
    if not validation.passed:
        raise RuntimeError(f"axis-anchored readiness V2 card invalid: {validation.errors}")
    dataset = "results/gate2_representation/gate2_20260824_gse_deduplicated_dataset_export_v1_seed0"
    v1 = "results/gate3_semantics/gate3_20260829_gse_axis_anchored_event_relation_readiness_v1_seed0"
    v1r = "results/gate3_semantics/gate3_20260829_gse_axis_anchored_event_relation_readiness_v1r_seed0"
    inputs = [
        DATA_CARD,
        f"{dataset}/RUN_STATE.json",
        f"{dataset}/metrics/summary.json",
        f"{dataset}/artifacts/sequence_manifest.jsonl",
        f"{dataset}/artifacts/shard_manifest.json",
        f"{dataset}/artifacts/evidence_sha256.txt",
        f"{v1}/RUN_STATE.json", f"{v1}/metrics/summary.json", f"{v1}/artifacts/evidence_sha256.txt",
        f"{v1r}/RUN_STATE.json", f"{v1r}/metrics/summary.json", f"{v1r}/artifacts/evidence_sha256.txt",
        "configs/v3/gate2/environments/phase3_torch290_cu129_zarr2187_v1r2.json",
    ]
    tools = {
        "model": "src/mtare_topo/representation/gse_axis_anchored_event_relation.py",
        "encoder": "src/mtare_topo/representation/phase3_structural_semantics.py",
        "teacher_materializer": "src/mtare_topo/data/gse_axis_anchored_event_relation_training.py",
        "executor": "tools/v3/execute_gse_axis_anchored_event_relation_readiness_v1.py",
        "runner": "tools/v3/run_gse_axis_anchored_event_relation_readiness_v1.py",
        "freezer": "tools/v3/freeze_gse_axis_anchored_event_relation_readiness_spec_v2.py",
        "model_tests": "tests/v3/unit/test_gse_axis_anchored_event_relation.py",
        "teacher_tests": "tests/v3/unit/test_gse_axis_anchored_event_relation_training.py",
        "governance": "src/mtare_topo/governance.py",
        "preflight": "tools/v3/preflight.py",
        "create_run": "tools/v3/create_run.py",
    }
    spec = {
        "schema_version": "v3_run_spec_v1",
        "gate": 3,
        "execution_phase": 3,
        "date": "20260829",
        "slug": "gse_axis_anchored_event_relation_readiness_v2",
        "seed": 0,
        "operation": "audit",
        "data_card": DATA_CARD,
        "question": "Can independent persistent/reveal/withdraw fields exactly represent all sealed C01-C08 causal relations, including same-bearing reveal-plus-withdraw events, while preserving finite causal model contracts?",
        "method": "Stream all 80 development worlds through the typed multi-label Teacher rasterizer, then run deterministic finite forward/backward and invariance checks on eight fixed real observations containing all five events and the discovered collision case.",
        "baseline": "Sealed categorical V1/V1R interface retained as a failed representation ablation.",
        "fallback": "Any count drift, same-channel collision, mask failure, nonfinite gradient or invariance regression stops before training; do not alter data, bins or thresholds.",
        "user_authorization": card["approval"],
        "acceptance_criteria": [
            "Fifteen unit tests pass for the model, multi-label loss, Teacher rasterizer and descriptor schedule.",
            "Exactly 60/10/10 C01-C06/C07/C08 worlds and 142184/21548/24394 observations are rasterized without same-channel collision.",
            "Exactly 202/20/33 same-bin reveal-plus-withdraw cases remain representable in fit/C07/C08.",
            "Eight fixed real observations cover five event classes, all three relation channels, positive/negative descriptor pairs and the real collision case.",
            "819067 parameters; finite masked/split backward; relation/event rotation <=3e-5, batch permutation <=3e-6, reverse/repeat exact.",
            "Full 33083-entry dataset seal verifies; RAM <=8 GiB; zero optimizer/checkpoint/C09/C10/M-TARE/graph/planner; sources unchanged."
        ],
        "expected_counts": {
            "worlds": 80,
            "fit_observations": 142184,
            "c07_observations": 21548,
            "c08_observations": 24394,
            "fixed_observations": 8,
            "unique_raw_frames": 34,
            "event_classes": 5,
            "relation_channels": 3,
            "unit_tests": 15,
            "optimizer_steps": 0,
            "checkpoints_created": 0,
            "new_model_inference_observations": 0,
            "c09_worlds_read": 0,
            "c10_worlds_read": 0,
            "mtare_worlds_read": 0,
            "graph_replays": 0
        },
        "expected_evidence": [
            "Exact per-partition full-population multi-label audit.",
            "Real collision manifest, component losses and paper-ready figure/source.",
            "Unit-test/raw logs, environment, source integrity, RUN_STATE and seal."
        ],
        "estimated_cost": card["estimated_cost"],
        "frozen_inputs": {path: sha256(PROJECT_ROOT / path) for path in sorted(inputs)},
        "frozen_tools": {name: {"path": path, "sha256": sha256(PROJECT_ROOT / path)} for name, path in tools.items()},
        "working_directory": str(PROJECT_ROOT),
        "command": [
            "/usr/bin/timeout", "3600s", PYTHON,
            "tools/v3/run_gse_axis_anchored_event_relation_readiness_v1.py",
            "--spec", str(SPEC),
            "--run-dir", str(PROJECT_ROOT / f"results/gate3_semantics/{RUN_ID}")
        ],
    }
    write_json(SPEC, spec)
    print(SPEC.relative_to(PROJECT_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
