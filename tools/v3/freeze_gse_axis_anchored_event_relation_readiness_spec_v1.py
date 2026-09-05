#!/usr/bin/env python3
"""Freeze the axis-anchored event-relation method-readiness run spec."""
from __future__ import annotations

import hashlib

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, validate_data_card, write_json


RUN_ID = "gate3_20260829_gse_axis_anchored_event_relation_readiness_v1_seed0"
SPEC = PROJECT_ROOT / "configs/v3/gate3/gse_axis_anchored_event_relation_readiness_v1.json"
DATA_CARD = "configs/v3/gate3/data_cards/gse_axis_anchored_event_relation_readiness_v1.json"
PYTHON = "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()


def main() -> int:
    if SPEC.exists(): raise RuntimeError("axis-anchored readiness spec already exists")
    card = load_json(PROJECT_ROOT / DATA_CARD); validation = validate_data_card(card)
    if not validation.passed: raise RuntimeError(f"axis-anchored readiness card invalid: {validation.errors}")
    dataset = "results/gate2_representation/gate2_20260824_gse_deduplicated_dataset_export_v1_seed0"
    teacher = "results/gate3_semantics/gate3_20260829_gse_axis_anchored_event_relation_teacher_feasibility_v1_seed0"
    inputs = [DATA_CARD, f"{dataset}/RUN_STATE.json", f"{dataset}/metrics/summary.json", f"{dataset}/artifacts/sequence_manifest.jsonl", f"{dataset}/artifacts/shard_manifest.json", f"{dataset}/artifacts/evidence_sha256.txt", f"{teacher}/RUN_STATE.json", f"{teacher}/metrics/summary.json", f"{teacher}/metrics/feasibility/summary.json", f"{teacher}/artifacts/evidence_sha256.txt", "configs/v3/gate2/environments/phase3_torch290_cu129_zarr2187_v1r2.json"]
    tools = {
        "model": "src/mtare_topo/representation/gse_axis_anchored_event_relation.py",
        "circular_encoder": "src/mtare_topo/representation/phase3_structural_semantics.py",
        "typed_observation": "src/mtare_topo/semantics/geometric_semantics.py",
        "executor": "tools/v3/execute_gse_axis_anchored_event_relation_readiness_v1.py",
        "runner": "tools/v3/run_gse_axis_anchored_event_relation_readiness_v1.py",
        "freezer": "tools/v3/freeze_gse_axis_anchored_event_relation_readiness_spec_v1.py",
        "tests": "tests/v3/unit/test_gse_axis_anchored_event_relation.py",
        "governance": "src/mtare_topo/governance.py",
        "preflight": "tools/v3/preflight.py", "create_run": "tools/v3/create_run.py",
    }
    spec = {
        "schema_version": "v3_run_spec_v1", "gate": 3, "execution_phase": 3, "date": "20260829", "slug": "gse_axis_anchored_event_relation_readiness_v1", "seed": 0, "operation": "audit", "data_card": DATA_CARD,
        "question": "Can the typed axis-anchored event-relation model satisfy causal, geometric, equivariant, reverse-axis and finite-backward contracts on real data without Teacher identity in forward?",
        "method": "Randomly initialize the 819196-parameter five-frame circular model once; run seven frozen C01 observations covering five events and four branch relations; evaluate typed forward isolation, masked event/relation/geometry/association loss, CUDA finite backward, aligned rotation, batch permutation, reverse-axis closure, deterministic repeat and past-only branch union.",
        "baseline": "Old three-class relational token model and complete-set models are interface baselines only; no weights are loaded.",
        "fallback": "Any method or system contract failure stops before training. Keep the 3e-5 rotation threshold and fix precision/operator behavior rather than relaxing it.",
        "user_authorization": card["approval"],
        "acceptance_criteria": [
            "Forward accepts only scans [B,5,2,16,720]; identity, pose, world, traversal, future and graph state are forbidden.",
            "At most 1.5M parameters and typed five-event/four-relation/branch-geometry/descriptor/uncertainty outputs.",
            "Fixed seven-observation real batch covers every event/relation class and positive/negative place and branch identity pairs.",
            "All real outputs, seven normalized loss components and all gradients are finite.",
            "Aligned relation/event rotation error <=3e-5; batch permutation <=3e-6; reverse-axis round trip and deterministic repeat exact.",
            "Seven unit tests pass; full dataset seal verifies; RAM <=8 GiB; zero optimizer/checkpoint/new inference/C07-C10/M-TARE/graph/planner; source unchanged."
        ],
        "expected_counts": {"fixed_observations": 7, "unique_raw_frames": 29, "event_classes": 5, "relation_classes": 4, "unit_tests": 7, "optimizer_steps": 0, "checkpoints_created": 0, "new_model_inference_observations": 0, "c09_worlds_read": 0, "c10_worlds_read": 0, "mtare_worlds_read": 0, "graph_replays": 0},
        "expected_evidence": ["Typed method contract and all readiness checks.", "Fixed real-batch manifest and component losses.", "Paper-ready PNG/PDF/SVG and figure source.", "Unit/raw logs, environment, source integrity, RUN_STATE and SHA-256 seal."],
        "estimated_cost": {"compute": "One deterministic RTX 5090 D forward/backward readiness proof", "wall_time_hours": 0.05, "host_ram_gb": 8, "gpu_memory_gb": 4, "disk_gb": 0.1},
        "frozen_inputs": {path: sha256(PROJECT_ROOT / path) for path in sorted(inputs)},
        "frozen_tools": {name: {"path": path, "sha256": sha256(PROJECT_ROOT / path)} for name, path in tools.items()},
        "working_directory": str(PROJECT_ROOT),
        "command": ["/usr/bin/timeout", "3600s", PYTHON, "tools/v3/run_gse_axis_anchored_event_relation_readiness_v1.py", "--spec", str(SPEC), "--run-dir", str(PROJECT_ROOT / f"results/gate3_semantics/{RUN_ID}")],
    }
    write_json(SPEC, spec); print(SPEC.relative_to(PROJECT_ROOT)); return 0


if __name__ == "__main__": raise SystemExit(main())
