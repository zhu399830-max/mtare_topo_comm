#!/usr/bin/env python3
"""Freeze the one sparse-relation cardinality corrective run spec."""
from __future__ import annotations

import hashlib

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, validate_data_card, write_json


RUN_ID = "gate3_20260829_gse_sparse_relation_cardinality_corrective_v1_seed0"
SPEC = PROJECT_ROOT / "configs/v3/gate3/gse_sparse_relation_cardinality_corrective_v1.json"
DATA_CARD = "configs/v3/gate3/data_cards/gse_sparse_circular_relation_transport_cardinality_corrective_v1.json"
PYTHON = "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    if SPEC.exists():
        raise RuntimeError("cardinality corrective spec already exists")
    card = load_json(PROJECT_ROOT / DATA_CARD)
    validation = validate_data_card(card)
    if not validation.passed:
        raise RuntimeError(f"cardinality corrective card invalid: {validation.errors}")
    dataset = "results/gate2_representation/gate2_20260824_gse_deduplicated_dataset_export_v1_seed0"
    predecessor = "results/gate3_semantics/gate3_20260829_gse_axis_anchored_event_relation_readiness_v2_seed0"
    sparse = "results/gate3_semantics/gate3_20260829_gse_sparse_circular_relation_transport_readiness_v1_seed0"
    slot = "results/gate3_semantics/gate3_20260829_gse_cardinality_conditioned_circular_slot_transport_three_seed_training_v1_seed0"
    inputs = [
        DATA_CARD,
        f"{dataset}/RUN_STATE.json", f"{dataset}/metrics/summary.json",
        f"{dataset}/artifacts/evidence_sha256.txt", f"{dataset}/artifacts/sequence_manifest.jsonl",
        f"{predecessor}/RUN_STATE.json", f"{predecessor}/metrics/summary.json", f"{predecessor}/artifacts/evidence_sha256.txt",
        f"{sparse}/RUN_STATE.json", f"{sparse}/metrics/summary.json", f"{sparse}/metrics/readiness/summary.json", f"{sparse}/artifacts/evidence_sha256.txt",
        f"{slot}/RUN_STATE.json", f"{slot}/metrics/summary.json", f"{slot}/artifacts/evidence_sha256.txt", f"{slot}/artifacts/models/seed0/best.pt",
        "configs/v3/gate2/environments/phase3_torch290_cu129_zarr2187_v1r2.json",
    ]
    tools = {
        "model": "src/mtare_topo/representation/gse_sparse_circular_relation_transport.py",
        "predecessor_model": "src/mtare_topo/representation/gse_circular_peak_geometry_model.py",
        "training_loader": "tools/v3/train_gse_axis_anchored_event_relation_v1.py",
        "readiness_helpers": "tools/v3/execute_gse_sparse_circular_relation_transport_readiness_v1.py",
        "executor": "tools/v3/execute_gse_sparse_relation_cardinality_corrective_v1.py",
        "runner": "tools/v3/run_gse_sparse_relation_cardinality_corrective_v1.py",
        "freezer": "tools/v3/freeze_gse_sparse_relation_cardinality_corrective_spec_v1.py",
        "model_tests": "tests/v3/unit/test_gse_sparse_circular_relation_transport.py",
        "corrective_tests": "tests/v3/unit/test_gse_sparse_relation_cardinality_corrective.py",
        "governance": "src/mtare_topo/governance.py",
        "preflight": "tools/v3/preflight.py",
        "create_run": "tools/v3/create_run.py",
    }
    spec = {
        "schema_version": "v3_run_spec_v1", "gate": 3, "execution_phase": 3,
        "date": "20260829", "slug": "gse_sparse_relation_cardinality_corrective_v1", "seed": 0,
        "operation": "audit", "data_card": DATA_CARD,
        "question": "Does the sparse relation candidate have a minimal explicit causal 0--6 cardinality interface suitable for three-seed training?",
        "method": "Add only Linear(128,7) to the sealed sparse relation candidate and use objective per-frame branch presence as count targets. On the same eight fixed fit observations, verify exact 903-parameter delta, count CE/gradient, causal future exclusion, rotation, permutation, repeat and predecessor backbone compatibility.",
        "baseline": "Sealed 783675-parameter sparse relation readiness without a deployable count decision.",
        "fallback": "Any failure stops training; repair only the cardinality or valid-mask interface, never tune an existence threshold or compensate in graph logic.",
        "user_authorization": card["approval"],
        "acceptance_criteria": [
            "Parameter total is 784578 and the sole count head contributes exactly 903 parameters.",
            "Forward remains scans-only and count logits/probabilities are finite [8,5,7] with explicit classes 0--6.",
            "All forty real frame targets equal both presence count and unique objective identity count; count CE and count-head gradients are finite and nonzero.",
            "Future perturbation changes no earlier count output; 10-degree rotation error <=3e-5, batch permutation <=3e-6 and repeat is exact.",
            "All 53 predecessor backbone keys and values load exactly.",
            "Zero optimizer/checkpoint/C07-C10 shard/M-TARE/graph/planner and all frozen inputs unchanged."
        ],
        "expected_counts": {
            "parameters": 784578, "parameter_delta": 903, "count_classes": 7,
            "fit_worlds_read": 3, "real_observations": 8, "causal_frame_positions": 40,
            "optimizer_steps": 0, "checkpoints_created": 0, "c07_worlds_read": 0, "c08_worlds_read": 0,
            "c09_worlds_read": 0, "c10_worlds_read": 0, "mtare_worlds_read": 0, "graph_replays": 0,
        },
        "expected_evidence": [
            "Exact parameter/count-target/backbone report.",
            "Causal, rotation, permutation, repeat and finite-gradient checks.",
            "PNG/PDF/SVG/source, CSV, environment, logs, RUN_STATE and SHA-256 seal."
        ],
        "estimated_cost": {"compute": "Deterministic CUDA readiness only", "wall_time_hours": 0.05, "host_ram_gb": 4, "gpu_memory_gb": 12, "disk_gb": 0.1},
        "frozen_inputs": {path: sha256(PROJECT_ROOT / path) for path in sorted(inputs)},
        "frozen_tools": {name: {"path": path, "sha256": sha256(PROJECT_ROOT / path)} for name, path in tools.items()},
        "working_directory": str(PROJECT_ROOT),
        "command": [
            "/usr/bin/timeout", "1800s", PYTHON,
            "tools/v3/run_gse_sparse_relation_cardinality_corrective_v1.py",
            "--spec", str(SPEC),
            "--run-dir", str(PROJECT_ROOT / f"results/gate3_semantics/{RUN_ID}"),
        ],
    }
    write_json(SPEC, spec)
    print(SPEC.relative_to(PROJECT_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
