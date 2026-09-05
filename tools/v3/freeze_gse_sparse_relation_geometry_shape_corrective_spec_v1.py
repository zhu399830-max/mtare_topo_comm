#!/usr/bin/env python3
"""Freeze the one sparse token-geometry shape corrective spec."""
from __future__ import annotations

import hashlib

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, validate_data_card, write_json


RUN_ID = "gate3_20260829_gse_sparse_relation_geometry_shape_corrective_v1_seed0"
SPEC = PROJECT_ROOT / "configs/v3/gate3/gse_sparse_relation_geometry_shape_corrective_v1.json"
DATA_CARD = "configs/v3/gate3/data_cards/gse_sparse_relation_geometry_shape_corrective_v1.json"
PYTHON = "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    if SPEC.exists():
        raise RuntimeError("geometry shape corrective spec already exists")
    card = load_json(PROJECT_ROOT / DATA_CARD)
    validation = validate_data_card(card)
    if not validation.passed:
        raise RuntimeError(f"geometry shape card invalid: {validation.errors}")
    dataset = "results/gate2_representation/gate2_20260824_gse_deduplicated_dataset_export_v1_seed0"
    readiness = "results/gate3_semantics/gate3_20260829_gse_axis_anchored_event_relation_readiness_v2_seed0"
    cardinality = "results/gate3_semantics/gate3_20260829_gse_sparse_relation_cardinality_corrective_v1_seed0"
    inputs = [
        DATA_CARD, f"{dataset}/RUN_STATE.json", f"{dataset}/metrics/summary.json", f"{dataset}/artifacts/evidence_sha256.txt", f"{dataset}/artifacts/sequence_manifest.jsonl",
        f"{readiness}/RUN_STATE.json", f"{readiness}/metrics/summary.json", f"{readiness}/artifacts/evidence_sha256.txt",
        f"{cardinality}/RUN_STATE.json", f"{cardinality}/metrics/summary.json", f"{cardinality}/artifacts/evidence_sha256.txt",
        "configs/v3/gate2/environments/phase3_torch290_cu129_zarr2187_v1r2.json",
    ]
    tools = {
        "model": "src/mtare_topo/representation/gse_sparse_circular_relation_transport.py",
        "trainer": "tools/v3/train_gse_sparse_circular_relation_transport_v2.py",
        "base_loader": "tools/v3/train_gse_axis_anchored_event_relation_v1.py",
        "executor": "tools/v3/execute_gse_sparse_relation_geometry_shape_corrective_v1.py",
        "runner": "tools/v3/run_gse_sparse_relation_geometry_shape_corrective_v1.py",
        "freezer": "tools/v3/freeze_gse_sparse_relation_geometry_shape_corrective_spec_v1.py",
        "model_tests": "tests/v3/unit/test_gse_sparse_circular_relation_transport.py",
        "count_tests": "tests/v3/unit/test_gse_sparse_relation_cardinality_corrective.py",
        "training_tests": "tests/v3/unit/test_train_gse_sparse_circular_relation_transport_v2.py",
        "governance": "src/mtare_topo/governance.py", "preflight": "tools/v3/preflight.py", "create_run": "tools/v3/create_run.py",
    }
    spec = {
        "schema_version": "v3_run_spec_v1", "gate": 3, "execution_phase": 3,
        "date": "20260829", "slug": "gse_sparse_relation_geometry_shape_corrective_v1", "seed": 0,
        "operation": "audit", "data_card": DATA_CARD,
        "question": "Does the corrected five-dimensional token-geometry uncertainty interface support the complete sparse relation training objective?",
        "method": "Remove exactly one unmatched uncertainty output (65 parameters), then run the same eight fit observations through count, proposal, transport, event, token/global geometry, uncertainty and descriptor losses with complete deterministic backward and symmetry checks.",
        "baseline": "Sealed 784578-parameter cardinality candidate with six uncertainty outputs for five geometry targets.",
        "fallback": "Any mismatch or missing gradient stops training; no padding, slicing, threshold change or graph compensation.",
        "user_authorization": card["approval"],
        "acceptance_criteria": [
            "Corrected parameter count is exactly 784513, a 65-parameter reduction only in token_geometry_head.",
            "Four profile values plus one width have exactly five uncertainty values.",
            "All core and descriptor losses, outputs and every trainable gradient are finite and nonzero on eight real observations.",
            "Count/geometry rotation <=3e-5, batch permutation <=3e-6, causal past and repeat exact.",
            "Forward remains scans-only and zero optimizer/checkpoint/C07-C10/M-TARE/graph/planner."
        ],
        "expected_counts": {"parameters": 784513, "removed_parameters": 65, "fit_worlds_read": 3, "real_observations": 8, "optimizer_steps": 0, "checkpoints_created": 0, "c07_worlds_read": 0, "c08_worlds_read": 0, "c09_worlds_read": 0, "c10_worlds_read": 0, "mtare_worlds_read": 0, "graph_replays": 0},
        "expected_evidence": ["Exact parameter/shape report.", "Full real core-plus-descriptor loss and gradient report.", "PNG/PDF/SVG/source, CSV, environment, logs, RUN_STATE and seal."],
        "estimated_cost": {"compute": "Deterministic CUDA corrective only", "wall_time_hours": 0.05, "host_ram_gb": 4, "gpu_memory_gb": 12, "disk_gb": 0.1},
        "frozen_inputs": {path: sha256(PROJECT_ROOT / path) for path in sorted(inputs)},
        "frozen_tools": {name: {"path": path, "sha256": sha256(PROJECT_ROOT / path)} for name, path in tools.items()},
        "working_directory": str(PROJECT_ROOT),
        "command": ["/usr/bin/timeout", "1800s", PYTHON, "tools/v3/run_gse_sparse_relation_geometry_shape_corrective_v1.py", "--spec", str(SPEC), "--run-dir", str(PROJECT_ROOT / f"results/gate3_semantics/{RUN_ID}")],
    }
    write_json(SPEC, spec)
    print(SPEC.relative_to(PROJECT_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
