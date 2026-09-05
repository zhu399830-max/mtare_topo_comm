#!/usr/bin/env python3
from __future__ import annotations

import hashlib

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import write_json


RUN_ID = "gate3_20260828_gse_geometry_anchored_spatial_event_readiness_v2_seed0"
SPEC = PROJECT_ROOT / "configs/v3/gate3/gse_geometry_anchored_spatial_event_readiness_v2.json"
DATA_CARD = "configs/v3/gate3/data_cards/gse_geometry_anchored_spatial_event_readiness_v2.json"
PYTHON = "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    if SPEC.exists():
        raise RuntimeError("readiness V2 spec already exists")
    teacher = "results/gate2_representation/gate2_20260828_gse_observable_spatial_event_teacher_v2_seed0"
    source = "results/gate2_representation/gate2_20260824_gse_deduplicated_dataset_export_v1_seed0"
    inputs = [
        DATA_CARD,
        f"{teacher}/RUN_STATE.json", f"{teacher}/metrics/summary.json", f"{teacher}/artifacts/evidence_sha256.txt", f"{teacher}/artifacts/export/summary.json", f"{teacher}/artifacts/export/teacher_shard_manifest.jsonl",
        f"{source}/RUN_STATE.json", f"{source}/artifacts/evidence_sha256.txt", f"{source}/artifacts/shard_manifest.json",
    ]
    tools = {
        "model": "src/mtare_topo/representation/gse_geometry_anchored_event.py",
        "set_loss": "src/mtare_topo/representation/gse_spatial_event_set.py",
        "executor": "tools/v3/execute_gse_geometry_anchored_spatial_event_readiness_v2.py",
        "runner": "tools/v3/run_gse_geometry_anchored_spatial_event_readiness_v2.py",
        "freezer": "tools/v3/freeze_gse_geometry_anchored_spatial_event_readiness_v2_spec.py",
        "tests": "tests/v3/unit/test_gse_geometry_anchored_event.py",
        "teacher_tests": "tests/v3/unit/test_gse_observable_spatial_event_teacher.py",
        "set_tests": "tests/v3/unit/test_gse_spatial_event_set.py",
        "governance": "src/mtare_topo/governance.py",
        "preflight": "tools/v3/preflight.py",
        "create_run": "tools/v3/create_run.py",
    }
    authorization = {"status": "APPROVED", "approved_by": "user-standing-authorization", "approved_at": "2026-08-28T21:20:00+08:00", "authorized_gates": [3], "authorized_operations": ["audit"], "scope": "One immutable C01-C08 observable-Teacher geometry-anchored readiness V2.", "confirmation_reference": "User instructed automatic best-choice execution without routine approval prompts."}
    spec = {
        "schema_version": "v3_run_spec_v1", "gate": 3, "execution_phase": 3, "date": "20260828", "slug": "gse_geometry_anchored_spatial_event_readiness_v2", "seed": 0, "operation": "audit", "data_card": DATA_CARD,
        "question": "Does the pose-free five-frame local free-range-envelope encoder express all observable Teacher V2 event tokens and satisfy deterministic circular set-prediction contracts before training?",
        "method": "Audit all V2 Teacher/source joins and every token against max five-frame range, one circular neighboring polar bin and the existing 0.25m LOS margin; verify the random-initialized 264134-parameter range/valid-only encoder on real cardinality 0-5 targets.",
        "baseline": "Readiness V1 passed all model/interface gates but failed because old continuous-LOS Teacher included 1631 targets outside the frozen vertical FOV.",
        "fallback": "Any V2 support/interface failure stops joint training; revise only the predeclared encoder/Teacher contract without test access.",
        "user_authorization": authorization,
        "acceptance_criteria": [
            "Exact 80 worlds, 188126 unique rows, 131424 tokens, 1076 identities and frozen V2 type/cardinality counts.",
            "All 131424 tokens within five-frame local free-range support plus fixed 0.25m LOS margin; NumPy/Torch support parity <=1e-6.",
            "Only range/valid inputs, 264134 parameters, real cardinality 0-5 finite loss/backward, query permutation <=1e-6.",
            "18-degree rotation xyz <=1mm, invariants <=2e-5, seed replay exact and predicted range bounded by anchor.",
            "Zero optimization, trained inference, threshold selection, C09/C10/M-TARE, graph or planner; sources unchanged.",
        ],
        "expected_counts": {"worlds": 80, "observations": 188126, "tokens": 131424, "event_identities": 1076, "parameters": 264134, "optimizer_steps": 0, "c09_worlds_read": 0, "c10_worlds_read": 0, "mtare_worlds_read": 0},
        "expected_evidence": ["Full 80-world Teacher/source/support log and support margins.", "Interface/equivariance/permutation/determinism/gradient summary, environment, RUN_STATE and SHA-256 seal."],
        "estimated_cost": {"compute": "Sequential CPU read of C01-C08 organized scans plus four six-row random forwards and one backward.", "wall_time_hours": 0.25, "host_ram_gb": 4, "gpu_memory_gb": 0, "disk_gb": 0.05, "gpu": "none"},
        "frozen_inputs": {path: sha256(PROJECT_ROOT / path) for path in sorted(inputs)},
        "frozen_tools": {name: {"path": path, "sha256": sha256(PROJECT_ROOT / path)} for name, path in tools.items()},
        "working_directory": str(PROJECT_ROOT),
        "command": ["/usr/bin/timeout", "2100s", PYTHON, "tools/v3/run_gse_geometry_anchored_spatial_event_readiness_v2.py", "--spec", str(SPEC), "--run-dir", str(PROJECT_ROOT / f"results/gate3_semantics/{RUN_ID}")],
    }
    write_json(SPEC, spec)
    print(SPEC.relative_to(PROJECT_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
