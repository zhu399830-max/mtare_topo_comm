#!/usr/bin/env python3
"""Freeze the one formal Joint Cyclic Gap Simplex readiness spec."""
from __future__ import annotations

import hashlib

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, validate_data_card, write_json


RUN_ID = "gate3_20260829_gse_joint_cyclic_gap_simplex_readiness_v1_seed0"
SPEC = PROJECT_ROOT / "configs/v3/gate3/gse_joint_cyclic_gap_simplex_readiness_v1.json"
DATA_CARD = "configs/v3/gate3/data_cards/gse_joint_cyclic_gap_simplex_readiness_v1.json"
PYTHON = "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    if SPEC.exists():
        raise RuntimeError("gap-simplex readiness spec already exists")
    card = load_json(PROJECT_ROOT / DATA_CARD); validation = validate_data_card(card)
    if not validation.passed:
        raise RuntimeError(f"gap-simplex Data Card invalid: {validation.errors}")
    dataset = "results/gate2_representation/gate2_20260824_gse_deduplicated_dataset_export_v1_seed0"
    teacher = "results/gate2_representation/gate2_20260829_gse_circular_exit_geometry_field_teacher_export_v1_seed0"
    attribution = "results/gate3_semantics/gate3_20260829_gse_coust_complex_cardinality_failure_attribution_v1_seed0"
    coust = "results/gate3_semantics/gate3_20260829_gse_cyclic_ordered_unimodal_slot_transport_three_seed_training_v1_seed0"
    inputs = [
        DATA_CARD,
        f"{dataset}/RUN_STATE.json", f"{dataset}/metrics/summary.json", f"{dataset}/artifacts/evidence_sha256.txt",
        f"{teacher}/RUN_STATE.json", f"{teacher}/metrics/summary.json", f"{teacher}/artifacts/evidence_sha256.txt",
        f"{attribution}/RUN_STATE.json", f"{attribution}/metrics/summary.json", f"{attribution}/artifacts/evidence_sha256.txt",
        f"{coust}/RUN_STATE.json", f"{coust}/metrics/summary.json", f"{coust}/artifacts/evidence_sha256.txt",
        "configs/v3/gate2/environments/phase3_torch290_cu129_zarr2187_v1r2.json",
    ]
    tools = {
        "gap_simplex": "src/mtare_topo/representation/gse_joint_cyclic_gap_simplex.py",
        "slot_transport": "src/mtare_topo/representation/gse_cardinality_conditioned_circular_slot_transport.py",
        "circular_backbone": "src/mtare_topo/representation/gse_circular_peak_geometry_model.py",
        "readiness_helpers": "tools/v3/execute_gse_circular_peak_geometry_model_readiness_v1.py",
        "executor": "tools/v3/execute_gse_joint_cyclic_gap_simplex_readiness_v1.py",
        "runner": "tools/v3/run_gse_joint_cyclic_gap_simplex_readiness_v1.py",
        "freezer": "tools/v3/freeze_gse_joint_cyclic_gap_simplex_readiness_spec_v1.py",
        "tests": "tests/v3/unit/test_gse_joint_cyclic_gap_simplex.py",
        "governance": "src/mtare_topo/governance.py", "preflight": "tools/v3/preflight.py", "create_run": "tools/v3/create_run.py",
    }
    spec = {
        "schema_version": "v3_run_spec_v1", "gate": 3, "execution_phase": 3, "date": "20260829", "slug": "gse_joint_cyclic_gap_simplex_readiness_v1", "seed": 0,
        "operation": "audit", "data_card": DATA_CARD,
        "question": "Can one equivariant phase plus a positive closed K-gap simplex jointly represent strict one-to-four exit geometry without independent-slot duplicate, missing, crossing or uncertainty defects?",
        "method": "Audit all 80 C01-C08 Teacher-source joins; test a 789650-parameter five-frame circular model with one phase field per cardinality, positive normalized gap branches, cyclic-relabel invariant joint loss, periodic geometry sampling and proper phase/gap scales on eight deterministic real rows.",
        "baseline": "COUST independent slots and its sealed complex-cardinality attribution.",
        "fallback": "Stop before training if any population, closure, positivity, cyclic/rotation, gradient, uncertainty, causal or finite-backward contract fails.",
        "user_authorization": card["approval"],
        "acceptance_criteria": [
            "Exactly 80 worlds, 188126 observations, 396913 exits and cardinality 7525/154279/24458/1864; zero join drift.",
            "Each cardinality branch has positive gaps summing to one; decoded gaps close at 360 degrees and wrap correctly.",
            "Rotated phase mass rolls, gap/scales remain invariant, bearings rotate and bound exit geometry remains invariant within 3e-5.",
            "Collapsed K3 gaps receive nonzero corrective gradient and proper uncertainty scale is minimized at known error.",
            "All outputs, losses and gradients finite on the fixed real 1-4 exit batch; zero optimizer/checkpoint/threshold/C09/C10/M-TARE/graph/planner."
        ],
        "expected_counts": {"worlds": 80, "observations": 188126, "visible_exits": 396913, "cardinality_1": 7525, "cardinality_2": 154279, "cardinality_3": 24458, "cardinality_4": 1864, "real_rows": 8, "parameters": 789650, "optimizer_steps": 0, "checkpoint_writes": 0, "c09_worlds_read": 0, "c10_worlds_read": 0, "mtare_worlds_read": 0, "graph_replays": 0},
        "expected_evidence": ["Population/causal join and real-row manifest.", "Closure, collapse-gradient, uncertainty and symmetry metrics.", "PNG/PDF/SVG/source, environment, log, RUN_STATE and SHA-256 seal."],
        "estimated_cost": {"compute": "CPU metadata audit plus eight-row forward/backward", "wall_time_hours": 0.1, "host_ram_gb": 6, "gpu_memory_gb": 0, "disk_gb": 0.1},
        "frozen_inputs": {path: sha256(PROJECT_ROOT / path) for path in sorted(inputs)},
        "frozen_tools": {name: {"path": path, "sha256": sha256(PROJECT_ROOT / path)} for name, path in tools.items()},
        "working_directory": str(PROJECT_ROOT),
        "command": ["/usr/bin/timeout", "1800s", PYTHON, "tools/v3/run_gse_joint_cyclic_gap_simplex_readiness_v1.py", "--spec", str(SPEC), "--run-dir", str(PROJECT_ROOT / f"results/gate3_semantics/{RUN_ID}")]
    }
    write_json(SPEC, spec); print(SPEC.relative_to(PROJECT_ROOT)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
