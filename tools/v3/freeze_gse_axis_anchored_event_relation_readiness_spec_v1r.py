#!/usr/bin/env python3
"""Freeze the masked-training corrective method-readiness spec."""
from __future__ import annotations

import hashlib

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, validate_data_card, write_json


RUN_ID = "gate3_20260829_gse_axis_anchored_event_relation_readiness_v1r_seed0"
SPEC = PROJECT_ROOT / "configs/v3/gate3/gse_axis_anchored_event_relation_readiness_v1r.json"
DATA_CARD = "configs/v3/gate3/data_cards/gse_axis_anchored_event_relation_readiness_v1r.json"
PYTHON = "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()


def main() -> int:
    if SPEC.exists(): raise RuntimeError("axis-anchored readiness V1R spec already exists")
    card = load_json(PROJECT_ROOT / DATA_CARD); validation = validate_data_card(card)
    if not validation.passed: raise RuntimeError(f"axis-anchored readiness V1R card invalid: {validation.errors}")
    dataset = "results/gate2_representation/gate2_20260824_gse_deduplicated_dataset_export_v1_seed0"
    v1 = "results/gate3_semantics/gate3_20260829_gse_axis_anchored_event_relation_readiness_v1_seed0"
    inputs = [DATA_CARD, f"{dataset}/RUN_STATE.json", f"{dataset}/metrics/summary.json", f"{dataset}/artifacts/sequence_manifest.jsonl", f"{dataset}/artifacts/shard_manifest.json", f"{dataset}/artifacts/evidence_sha256.txt", f"{v1}/RUN_STATE.json", f"{v1}/metrics/summary.json", f"{v1}/metrics/readiness/summary.json", f"{v1}/artifacts/evidence_sha256.txt", "configs/v3/gate2/environments/phase3_torch290_cu129_zarr2187_v1r2.json"]
    tools = {
        "model": "src/mtare_topo/representation/gse_axis_anchored_event_relation.py",
        "encoder": "src/mtare_topo/representation/phase3_structural_semantics.py",
        "executor": "tools/v3/execute_gse_axis_anchored_event_relation_readiness_v1.py",
        "runner": "tools/v3/run_gse_axis_anchored_event_relation_readiness_v1.py",
        "freezer": "tools/v3/freeze_gse_axis_anchored_event_relation_readiness_spec_v1r.py",
        "tests": "tests/v3/unit/test_gse_axis_anchored_event_relation.py",
        "governance": "src/mtare_topo/governance.py",
        "preflight": "tools/v3/preflight.py", "create_run": "tools/v3/create_run.py",
    }
    spec = {
        "schema_version": "v3_run_spec_v1", "gate": 3, "execution_phase": 3, "date": "20260829", "slug": "gse_axis_anchored_event_relation_readiness_v1r", "seed": 0, "operation": "audit", "data_card": DATA_CARD,
        "question": "Can missing early relation labels be masked without deleting frames or fabricating targets, while fit-only class weights and split core/identity objectives remain finite and preserve every V1 model invariant?",
        "method": "Repeat the unchanged V1 real batch and invariants; additionally mask 14 early pair positions, provide independent current-branch masks, apply frozen C01-C06 inverse-square-root event/relation weights, and separately backpropagate core and identity-balanced objectives.",
        "baseline": "Sealed V1 readiness on the identical seven observations.",
        "fallback": "Any corrective or prior invariant failure stops training; do not change rows, model, weights, masks or thresholds.",
        "user_authorization": card["approval"],
        "acceptance_criteria": [
            "Eight unit tests pass including masked relation and split-loss regression.",
            "Masked early pairs, independent branch presence, fixed event/relation weights and split core/descriptor losses are all finite with finite gradients.",
            "Unchanged 819196 parameters; same seven observations cover five events/four relations and identity pairs.",
            "Relation/event rotation <=3e-5, batch permutation <=3e-6, reverse/repeat exact.",
            "Full 33083 source seal verifies; RAM <=8 GiB; zero optimizer/checkpoint/new inference/C07-C10/M-TARE/graph/planner; sources unchanged."
        ],
        "expected_counts": {"fixed_observations": 7, "unique_raw_frames": 29, "masked_relation_pair_positions": 14, "event_classes": 5, "relation_classes": 4, "unit_tests": 8, "optimizer_steps": 0, "checkpoints_created": 0, "new_model_inference_observations": 0, "c09_worlds_read": 0, "c10_worlds_read": 0, "mtare_worlds_read": 0, "graph_replays": 0},
        "expected_evidence": ["All V1 invariants plus masked/split training contract.", "Real-batch manifest, component losses and paper-ready figure/source.", "Logs, environment, source integrity, RUN_STATE and seal."],
        "estimated_cost": {"compute": "Deterministic RTX 5090 D corrective readiness", "wall_time_hours": 0.05, "host_ram_gb": 8, "gpu_memory_gb": 4, "disk_gb": 0.1},
        "frozen_inputs": {path: sha256(PROJECT_ROOT / path) for path in sorted(inputs)},
        "frozen_tools": {name: {"path": path, "sha256": sha256(PROJECT_ROOT / path)} for name, path in tools.items()},
        "working_directory": str(PROJECT_ROOT),
        "command": ["/usr/bin/timeout", "3600s", PYTHON, "tools/v3/run_gse_axis_anchored_event_relation_readiness_v1.py", "--spec", str(SPEC), "--run-dir", str(PROJECT_ROOT / f"results/gate3_semantics/{RUN_ID}")],
    }
    write_json(SPEC, spec); print(SPEC.relative_to(PROJECT_ROOT)); return 0


if __name__ == "__main__": raise SystemExit(main())
