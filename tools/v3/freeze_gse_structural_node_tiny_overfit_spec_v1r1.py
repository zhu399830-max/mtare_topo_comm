#!/usr/bin/env python3
"""Freeze V1R1 after correcting only the Data Card world enumeration."""

from __future__ import annotations

import hashlib

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import write_json


SPEC = PROJECT_ROOT / "configs/v3/gate3/gse_structural_node_tiny_overfit_v1r1.json"
RUN_ID = "gate3_20260904_gse_structural_node_tiny_overfit_v1r1_seed0"
PYTHON = "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python"


def _sha(path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    if SPEC.exists():
        raise RuntimeError("V1R1 spec exists; overwrite is forbidden")
    source = "results/gate3_semantics/gate3_20260904_gse_structural_node_tiny_overfit_v1_seed0"
    capacity = "results/gate3_semantics/gate3_20260904_gse_structural_node_graph_consistency_attribution_v1_seed0"
    inputs = [
        f"{source}/RUN_STATE.json",
        f"{source}/metrics/summary.json",
        f"{source}/artifacts/evidence_sha256.txt",
        f"{source}/artifacts/tiny_overfit/frozen_tiny_features.npz",
        f"{source}/artifacts/tiny_overfit/sample_manifest.json",
        f"{capacity}/RUN_STATE.json",
        f"{capacity}/metrics/attribution/summary.json",
        f"{capacity}/artifacts/evidence_sha256.txt",
    ]
    tools = {
        "evidence": "src/mtare_topo/representation/gse_structural_node_evidence.py",
        "student_inputs": "src/mtare_topo/representation/gse_structural_node_student.py",
        "dual_readout": "src/mtare_topo/representation/gse_structural_node_dual_readout.py",
        "trainer": "tools/v3/train_gse_structural_node_tiny_overfit_v1r.py",
        "runner_implementation": "tools/v3/run_gse_structural_node_tiny_overfit_v1r.py",
        "runner": "tools/v3/run_gse_structural_node_tiny_overfit_v1r1.py",
        "freezer": "tools/v3/freeze_gse_structural_node_tiny_overfit_spec_v1r1.py",
        "tests_evidence": "tests/v3/unit/test_gse_structural_node_evidence.py",
        "tests_student": "tests/v3/unit/test_gse_structural_node_student.py",
        "tests_dual": "tests/v3/unit/test_gse_structural_node_dual_readout.py",
        "governance": "src/mtare_topo/governance.py",
        "preflight": "tools/v3/preflight.py",
        "create_run": "tools/v3/create_run.py",
    }
    approval = {
        "status": "APPROVED",
        "approved_by": "user-standing-authorization",
        "approved_at": "2026-09-04T23:45:00+08:00",
        "authorized_operations": ["training"],
        "authorized_gates": [3],
        "scope": "One immutable readout corrective on the sealed 180-observation cache; exactly 500 updates and zero C07-C10/graph/M-TARE.",
        "confirmation_reference": "User authorized continuous optimal in-scope execution; the prior preflight created no run and this version changes only trajectory enumeration.",
    }
    spec = {
        "schema_version": "v3_run_spec_v1",
        "gate": 3,
        "execution_phase": 3,
        "date": "20260904",
        "slug": "gse_structural_node_tiny_overfit_v1r1",
        "seed": 0,
        "operation": "training",
        "question": "Can separated normalized geometry and place-association readouts fit the unchanged structural-node tiny population within 500 updates?",
        "method": "Use the sealed V1 180x64x44 cache. Train only a permutation-invariant dual readout: analytically scaled geometry regression and an independent normalized association embedding with balanced and supervised-contrastive losses.",
        "baseline": "V1 shared readout: degree accuracy 0.99444, geometry RMSE 0.14321, loss reduction 0.69449 and zero safe association at precision>=0.99.",
        "fallback": "If V1R1 fails, stop this student interface before one-seed/three-seed training; do not add steps, regenerate data, unfreeze the backbone or lower gates.",
        "config_path": "configs/v3/gate3/data_cards/gse_structural_node_tiny_overfit_v1r1.json",
        "data_card": "configs/v3/gate3/data_cards/gse_structural_node_tiny_overfit_v1r1.json",
        "user_authorization": approval,
        "acceptance_criteria": [
            "Exactly 180 observations, 100 nodes, ten parents and 80 positive pairs from the sealed V1 cache.",
            "Exactly 500 readout-only updates; total loss reduction>=95%, raw geometry RMSE<=0.01 and degree accuracy>=0.99.",
            "Tie-safe association precision>=0.99 and recall>=0.95 with nonempty accepted pairs.",
            "All gradients finite, permutation and repeat tests pass, and sealed inputs remain unchanged.",
            "Zero backbone inference, C07-C10, graph replay and M-TARE."
        ],
        "expected_evidence": ["Losses, geometry error, association distributions, checkpoint, PNG/PDF/SVG diagnostic, resources, environment, RUN_STATE and seal."],
        "estimated_cost": {"compute": "500 full-batch updates on a cached 180-observation tensor", "wall_time_hours": 0.1, "host_ram_gb": 4, "disk_gb": 0.1, "gpu": 1},
        "expected_counts": {"families": 10, "tasks": 10, "observations": 180, "nodes": 100, "positive_pairs": 80, "candidate_pairs": 1530, "optimizer_steps": 500, "model_inference_sequences": 0, "c07_rows_read": 0, "c08_c09_c10_worlds_read": 0, "graph_replays": 0, "mtare_runs": 0},
        "frozen_inputs": {relative: _sha(PROJECT_ROOT / relative) for relative in inputs},
        "frozen_tools": {name: {"path": relative, "sha256": _sha(PROJECT_ROOT / relative)} for name, relative in tools.items()},
        "command": [
            "/usr/bin/timeout", "--signal=INT", "--kill-after=30s", "600s", PYTHON,
            "tools/v3/run_gse_structural_node_tiny_overfit_v1r1.py",
            "--spec", str(SPEC),
            "--run-dir", str(PROJECT_ROOT / f"results/gate3_semantics/{RUN_ID}"),
        ],
        "working_directory": str(PROJECT_ROOT),
    }
    write_json(SPEC, spec)
    print({"spec": str(SPEC), "frozen_inputs": len(inputs), "frozen_tools": len(tools)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
