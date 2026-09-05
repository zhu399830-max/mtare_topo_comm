#!/usr/bin/env python3
"""Freeze the one sparse circular relation-transport readiness spec."""
from __future__ import annotations

import hashlib

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, validate_data_card, write_json


RUN_ID = "gate3_20260829_gse_sparse_circular_relation_transport_readiness_v1_seed0"
SPEC = PROJECT_ROOT / "configs/v3/gate3/gse_sparse_circular_relation_transport_readiness_v1.json"
DATA_CARD = "configs/v3/gate3/data_cards/gse_sparse_circular_relation_transport_readiness_v1.json"
PYTHON = "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    if SPEC.exists():
        raise RuntimeError("sparse relation readiness spec already exists")
    card = load_json(PROJECT_ROOT / DATA_CARD); validation = validate_data_card(card)
    if not validation.passed:
        raise RuntimeError(f"sparse relation readiness card invalid: {validation.errors}")
    dataset = "results/gate2_representation/gate2_20260824_gse_deduplicated_dataset_export_v1_seed0"
    predecessor = "results/gate3_semantics/gate3_20260829_gse_axis_anchored_event_relation_readiness_v2_seed0"
    slot = "results/gate3_semantics/gate3_20260829_gse_cardinality_conditioned_circular_slot_transport_three_seed_training_v1_seed0"
    attribution = "results/gate3_semantics/gate3_20260829_gse_axis_anchored_v2_failure_attribution_v1_seed0"
    inputs = [DATA_CARD, f"{dataset}/RUN_STATE.json", f"{dataset}/metrics/summary.json", f"{dataset}/artifacts/evidence_sha256.txt", f"{dataset}/artifacts/sequence_manifest.jsonl", f"{predecessor}/RUN_STATE.json", f"{predecessor}/metrics/summary.json", f"{predecessor}/artifacts/evidence_sha256.txt", f"{slot}/RUN_STATE.json", f"{slot}/metrics/summary.json", f"{slot}/artifacts/evidence_sha256.txt", f"{slot}/artifacts/models/seed0/best.pt", f"{attribution}/RUN_STATE.json", f"{attribution}/metrics/summary.json", f"{attribution}/artifacts/evidence_sha256.txt", "configs/v3/gate2/environments/phase3_torch290_cu129_zarr2187_v1r2.json"]
    tools = {
        "model": "src/mtare_topo/representation/gse_sparse_circular_relation_transport.py",
        "predecessor_model": "src/mtare_topo/representation/gse_circular_peak_geometry_model.py",
        "training_loader": "tools/v3/train_gse_axis_anchored_event_relation_v1.py",
        "executor": "tools/v3/execute_gse_sparse_circular_relation_transport_readiness_v1.py",
        "runner": "tools/v3/run_gse_sparse_circular_relation_transport_readiness_v1.py",
        "freezer": "tools/v3/freeze_gse_sparse_circular_relation_transport_readiness_spec_v1.py",
        "tests": "tests/v3/unit/test_gse_sparse_circular_relation_transport.py",
        "attribution_tests": "tests/v3/unit/test_gse_axis_anchored_v2_failure_attribution.py",
        "governance": "src/mtare_topo/governance.py", "preflight": "tools/v3/preflight.py", "create_run": "tools/v3/create_run.py",
    }
    spec = {
        "schema_version": "v3_run_spec_v1", "gate": 3, "execution_phase": 3, "date": "20260829", "slug": "gse_sparse_circular_relation_transport_readiness_v1", "seed": 0, "operation": "audit", "data_card": DATA_CARD,
        "question": "Can sparse circular exit tokens and explicit current-token/dustbin transport satisfy all causal, semantic, equivariant, compatibility and finite-gradient contracts before V2 training?",
        "method": "Load the exact predecessor backbone into a 783675-parameter LiDAR-only V2, run fixed synthetic relation cases and eight fixed complete-history C01-C03 observations, and verify token/dustbin semantics, causal future exclusion, rotation, permutation, reverse time, determinism and complete finite gradients.",
        "baseline": "Failed dense Axis-Anchored V1 and the compatible five-frame Slot backbone with C07/C08 axis 5.4982/5.7835 degrees.",
        "fallback": "Any interface or invariant failure stops before training and permits only a local readiness correction; no dense relation restoration, checkpoint selection, threshold tuning or graph compensation.",
        "user_authorization": card["approval"],
        "acceptance_criteria": [
            "Typed forward has only scans and exactly six circular tokens with persistent/reveal/withdraw dustbin semantics.",
            "Full 80-world population provenance and the fixed eight-row/34-frame real batch match sealed evidence exactly.",
            "All required predecessor backbone keys and values load exactly; no shape mismatch.",
            "Simultaneous reveal+withdraw, real-to-real persistence, wrap, reverse-time involution and causal future exclusion pass.",
            "Proposal/transport/axis rotation <=3e-5 (bearing <=3e-4 degrees), semantic batch permutation <=3e-6, repeat exact and every trainable parameter has finite real-batch gradient.",
            "Zero optimizer/checkpoint/C07-C10 shard read/M-TARE/graph/planner and all frozen inputs unchanged."
        ],
        "expected_counts": {"parameters": 783675, "full_worlds_provenance": 80, "fit_worlds_read": 3, "real_observations": 8, "unique_history_frames": 34, "maximum_tokens": 6, "optimizer_steps": 0, "checkpoints_created": 0, "c09_worlds_read": 0, "c10_worlds_read": 0, "mtare_worlds_read": 0, "graph_replays": 0},
        "expected_evidence": ["Typed method and checkpoint compatibility report.", "Synthetic relation and real-batch finite-gradient evidence.", "PNG/PDF/SVG/source, checks CSV, environment, logs, RUN_STATE and SHA-256 seal."],
        "estimated_cost": {"compute": "Deterministic CUDA readiness only", "wall_time_hours": 0.05, "host_ram_gb": 4, "gpu_memory_gb": 12, "disk_gb": 0.1},
        "frozen_inputs": {path: sha256(PROJECT_ROOT / path) for path in sorted(inputs)}, "frozen_tools": {name: {"path": path, "sha256": sha256(PROJECT_ROOT / path)} for name, path in tools.items()}, "working_directory": str(PROJECT_ROOT),
        "command": ["/usr/bin/timeout", "1800s", PYTHON, "tools/v3/run_gse_sparse_circular_relation_transport_readiness_v1.py", "--spec", str(SPEC), "--run-dir", str(PROJECT_ROOT / f"results/gate3_semantics/{RUN_ID}")],
    }
    write_json(SPEC, spec); print(SPEC.relative_to(PROJECT_ROOT)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
