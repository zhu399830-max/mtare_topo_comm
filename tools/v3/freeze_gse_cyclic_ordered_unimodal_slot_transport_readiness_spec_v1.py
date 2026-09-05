#!/usr/bin/env python3
"""Freeze the one immutable COUST zero-training readiness run."""
from __future__ import annotations

import hashlib

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, validate_data_card, write_json


RUN_ID = "gate3_20260829_gse_cyclic_ordered_unimodal_slot_transport_readiness_v1_seed0"
SPEC = PROJECT_ROOT / "configs/v3/gate3/gse_cyclic_ordered_unimodal_slot_transport_readiness_v1.json"
CARD = "configs/v3/gate3/data_cards/gse_cyclic_ordered_unimodal_slot_transport_readiness_v1.json"
PYTHON = "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    if SPEC.exists():
        raise RuntimeError("COUST readiness spec already exists")
    card = load_json(PROJECT_ROOT / CARD)
    validation = validate_data_card(card)
    if not validation.passed:
        raise RuntimeError(f"invalid Data Card: {validation.errors}")

    dataset = "results/gate2_representation/gate2_20260824_gse_deduplicated_dataset_export_v1_seed0"
    teacher = "results/gate2_representation/gate2_20260829_gse_circular_exit_geometry_field_teacher_export_v1_seed0"
    training = "results/gate3_semantics/gate3_20260829_gse_cardinality_conditioned_circular_slot_transport_three_seed_training_v1_seed0"
    attribution = "results/gate3_semantics/gate3_20260829_gse_cardinality_conditioned_circular_slot_transport_failure_attribution_v1_seed0"
    inputs = [
        CARD,
        f"{dataset}/RUN_STATE.json",
        f"{dataset}/artifacts/evidence_sha256.txt",
        f"{dataset}/artifacts/shard_manifest.json",
        f"{teacher}/RUN_STATE.json",
        f"{teacher}/artifacts/evidence_sha256.txt",
        f"{teacher}/artifacts/export/teacher_shard_manifest.jsonl",
        f"{training}/RUN_STATE.json",
        f"{training}/metrics/summary.json",
        f"{training}/artifacts/evidence_sha256.txt",
        f"{attribution}/RUN_STATE.json",
        f"{attribution}/metrics/summary.json",
        f"{attribution}/artifacts/evidence_sha256.txt",
    ]
    tools = {
        "model": "src/mtare_topo/representation/gse_cyclic_ordered_unimodal_slot_transport.py",
        "parent_model": "src/mtare_topo/representation/gse_cardinality_conditioned_circular_slot_transport.py",
        "executor": "tools/v3/execute_gse_cyclic_ordered_unimodal_slot_transport_readiness_v1.py",
        "runner": "tools/v3/run_gse_cyclic_ordered_unimodal_slot_transport_readiness_v1.py",
        "freezer": "tools/v3/freeze_gse_cyclic_ordered_unimodal_slot_transport_readiness_spec_v1.py",
        "real_batch": "tools/v3/execute_gse_circular_peak_geometry_model_readiness_v1.py",
        "model_tests": "tests/v3/unit/test_gse_cyclic_ordered_unimodal_slot_transport.py",
        "governance": "src/mtare_topo/governance.py",
        "preflight": "tools/v3/preflight.py",
        "create_run": "tools/v3/create_run.py",
    }
    spec = {
        "schema_version": "v3_run_spec_v1",
        "gate": 3,
        "execution_phase": 3,
        "date": "20260829",
        "slug": "gse_cyclic_ordered_unimodal_slot_transport_readiness_v1",
        "seed": 0,
        "operation": "audit",
        "data_card": CARD,
        "question": "Can orientation-preserving cyclic assignment plus proper unimodal slot distributions remove the identified K>=3 label-switching failure without violating causal circular symmetry?",
        "method": "Zero-training full-population, synthetic and real-batch readiness for cyclic-ordered unimodal slot transport.",
        "baseline": "Sealed cardinality-conditioned slot transport training and its frozen failure attribution.",
        "fallback": "Any population, order, gradient, unimodality, symmetry, causal or finite-backward failure stops COUST before training.",
        "user_authorization": {
            "status": "APPROVED",
            "approved_by": "user-standing-authorization",
            "approved_at": "2026-08-29T00:40:00+08:00",
            "authorized_gates": [3],
            "authorized_operations": ["audit"],
            "scope": "One immutable zero-training COUST readiness run.",
            "confirmation_reference": "User requested autonomous best-method execution without repeated approval prompts.",
        },
        "acceptance_criteria": [
            "Exact 80-world train population and causal source join with no forbidden Teacher arrays.",
            "Cyclic assignment is invariant to cyclic slot relabeling, penalizes reversed order, and duplicate slots receive missing-mode gradient.",
            "Projected distributions are normalized, finite, unimodal, and have valid concentration.",
            "Real 1-4 exit batch has finite forward/loss/backward and five-frame history connectivity.",
            "Circular rotation, cyclic-slot, batch-permutation and repeat errors satisfy the frozen 3e-5 contract; zero training or test/graph use.",
        ],
        "expected_counts": {
            "worlds": 80,
            "observations": 188126,
            "peaks": 396913,
            "cardinality_1": 7525,
            "cardinality_2": 154279,
            "cardinality_3": 24458,
            "cardinality_4": 1864,
            "real_rows": 8,
            "parameters": 788618,
            "optimizer_steps": 0,
            "c09_worlds_read": 0,
            "c10_worlds_read": 0,
            "mtare_worlds_read": 0,
            "graph_replays": 0,
        },
        "expected_evidence": [
            "Full population and causal audit.",
            "Synthetic cyclic-order, duplicate-gradient and unimodality metrics.",
            "Real finite backward and symmetry metrics.",
            "PNG/PDF/SVG/source figure, environment, log, RUN_STATE and SHA-256 seal.",
        ],
        "estimated_cost": {
            "compute": "CPU metadata audit and real eight-row Torch readiness",
            "wall_time_hours": 0.1,
            "host_ram_gb": 6,
            "gpu_memory_gb": 0,
            "disk_gb": 0.1,
            "gpu": "none",
        },
        "frozen_inputs": {path: sha256(PROJECT_ROOT / path) for path in sorted(inputs)},
        "frozen_tools": {name: {"path": path, "sha256": sha256(PROJECT_ROOT / path)} for name, path in tools.items()},
        "working_directory": str(PROJECT_ROOT),
        "command": [
            "/usr/bin/timeout", "1800s", PYTHON,
            "tools/v3/run_gse_cyclic_ordered_unimodal_slot_transport_readiness_v1.py",
            "--spec", str(SPEC),
            "--run-dir", str(PROJECT_ROOT / f"results/gate3_semantics/{RUN_ID}"),
        ],
    }
    write_json(SPEC, spec)
    print(SPEC.relative_to(PROJECT_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
