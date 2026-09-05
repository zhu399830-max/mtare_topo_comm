#!/usr/bin/env python3
"""Freeze the one formal COUST complex-cardinality attribution spec."""
from __future__ import annotations

import hashlib

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, validate_data_card, write_json


RUN_ID = "gate3_20260829_gse_coust_complex_cardinality_failure_attribution_v1_seed0"
SPEC = PROJECT_ROOT / "configs/v3/gate3/gse_coust_complex_cardinality_failure_attribution_v1.json"
DATA_CARD = "configs/v3/gate3/data_cards/gse_coust_complex_cardinality_failure_attribution_v1.json"
PYTHON = "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    if SPEC.exists():
        raise RuntimeError("COUST complex-cardinality attribution spec already exists")
    card = load_json(PROJECT_ROOT / DATA_CARD)
    validation = validate_data_card(card)
    if not validation.passed:
        raise RuntimeError(f"COUST attribution card invalid: {validation.errors}")
    teacher = "results/gate2_representation/gate2_20260829_gse_circular_exit_geometry_field_teacher_export_v1_seed0"
    training = "results/gate3_semantics/gate3_20260829_gse_cyclic_ordered_unimodal_slot_transport_three_seed_training_v1_seed0"
    baseline = "results/gate3_semantics/gate3_20260829_gse_cardinality_conditioned_circular_slot_transport_three_seed_training_v1_seed0"
    readiness = "results/gate3_semantics/gate3_20260829_gse_cyclic_ordered_unimodal_slot_transport_readiness_v1r_seed0"
    inputs = [
        DATA_CARD,
        f"{teacher}/RUN_STATE.json", f"{teacher}/metrics/summary.json", f"{teacher}/artifacts/evidence_sha256.txt",
        f"{training}/RUN_STATE.json", f"{training}/metrics/summary.json", f"{training}/metrics/selection/summary.json", f"{training}/artifacts/evidence_sha256.txt",
        f"{baseline}/RUN_STATE.json", f"{baseline}/metrics/summary.json", f"{baseline}/metrics/selection/summary.json", f"{baseline}/artifacts/evidence_sha256.txt",
        f"{readiness}/RUN_STATE.json", f"{readiness}/metrics/summary.json", f"{readiness}/artifacts/evidence_sha256.txt",
        "configs/v3/gate2/environments/phase3_torch290_cu129_zarr2187_v1r2.json",
    ]
    for seed in range(3):
        inputs.append(f"{training}/artifacts/models/seed{seed}/summary.json")
        inputs.extend(str(path.relative_to(PROJECT_ROOT)) for path in sorted((PROJECT_ROOT / training / f"artifacts/models/seed{seed}/development_predictions").glob("*.npz")))
    tools = {
        "coust": "src/mtare_topo/representation/gse_cyclic_ordered_unimodal_slot_transport.py",
        "slot_transport": "src/mtare_topo/representation/gse_cardinality_conditioned_circular_slot_transport.py",
        "coust_evaluator": "tools/v3/evaluate_gse_cyclic_ordered_unimodal_slot_transport_selection_v1.py",
        "base_attribution": "tools/v3/execute_gse_cardinality_conditioned_circular_slot_transport_failure_attribution_v1.py",
        "executor": "tools/v3/execute_gse_coust_complex_cardinality_failure_attribution_v1.py",
        "runner": "tools/v3/run_gse_coust_complex_cardinality_failure_attribution_v1.py",
        "freezer": "tools/v3/freeze_gse_coust_complex_cardinality_failure_attribution_spec_v1.py",
        "tests": "tests/v3/unit/test_gse_coust_complex_cardinality_failure_attribution.py",
        "governance": "src/mtare_topo/governance.py",
        "preflight": "tools/v3/preflight.py",
        "create_run": "tools/v3/create_run.py",
    }
    spec = {
        "schema_version": "v3_run_spec_v1", "gate": 3, "execution_phase": 3, "date": "20260829",
        "slug": "gse_coust_complex_cardinality_failure_attribution_v1", "seed": 0,
        "operation": "audit", "data_card": DATA_CARD,
        "question": "Why does COUST improve aggregate one/two-exit geometry while collapsing on strict three/four-exit sets and transferring concentration confidence poorly?",
        "method": "Read all sealed C07/C08 Teacher and three-seed COUST predictions once; compare each seed, cyclic ensemble and unrestricted assignment ensemble; quantify K=3/4 slot spacing, cyclic assignment optimality, kappa and concentration separability.",
        "baseline": "Formal COUST cyclic ensemble versus the preceding Slot Transport K=3/4 exact-set results.",
        "fallback": "If unrestricted alignment or a single seed does not recover both development splits, stop post-processing and classify the failure at the representation/objective level before designing a new method.",
        "user_authorization": card["approval"],
        "acceptance_criteria": [
            "Exactly 10 C07 plus 10 C08 worlds, 21548/24394 observations and 45504/51537 exits; all joins exact.",
            "Report single-seed, cyclic-ensemble and unrestricted-ensemble K=3/4 exact sets at 2/4/10 degrees.",
            "Report predicted versus target minimum circular gap, collapse fraction, cyclic-optimal assignment fraction and cyclic excess error for K=3/4.",
            "Report kappa and concentration exact-set separability overall and by cardinality without selecting a new deployment threshold.",
            "Return one predeclared cause classification; zero optimizer, new inference, C09/C10/M-TARE/graph/planner; all frozen inputs unchanged."
        ],
        "expected_counts": {"c07_worlds": 10, "c08_worlds": 10, "c07_observations": 21548, "c08_observations": 24394, "c07_exits": 45504, "c08_exits": 51537, "seeds": 3, "decoders": 5, "optimizer_steps": 0, "new_model_inference_observations": 0, "c09_worlds_read": 0, "c10_worlds_read": 0, "mtare_worlds_read": 0, "graph_replays": 0},
        "expected_evidence": ["Single-seed/cyclic/unrestricted complex-cardinality table.", "Slot-spacing, cyclic-assignment and confidence-separability attribution.", "PNG/PDF/SVG/source, CSV, environment, log, RUN_STATE and SHA-256 seal."],
        "estimated_cost": {"compute": "CPU-only deterministic frozen-output attribution", "wall_time_hours": 0.2, "host_ram_gb": 8, "gpu_memory_gb": 0, "disk_gb": 0.1},
        "frozen_inputs": {path: sha256(PROJECT_ROOT / path) for path in sorted(inputs)},
        "frozen_tools": {name: {"path": path, "sha256": sha256(PROJECT_ROOT / path)} for name, path in tools.items()},
        "working_directory": str(PROJECT_ROOT),
        "command": ["/usr/bin/timeout", "7200s", PYTHON, "tools/v3/run_gse_coust_complex_cardinality_failure_attribution_v1.py", "--spec", str(SPEC), "--run-dir", str(PROJECT_ROOT / f"results/gate3_semantics/{RUN_ID}")]
    }
    write_json(SPEC, spec)
    print(SPEC.relative_to(PROJECT_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
