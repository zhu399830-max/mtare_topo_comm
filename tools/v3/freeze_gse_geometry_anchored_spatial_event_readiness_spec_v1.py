#!/usr/bin/env python3
from __future__ import annotations

import hashlib

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import write_json


RUN_ID = "gate3_20260828_gse_geometry_anchored_spatial_event_readiness_v1_seed0"
SPEC = PROJECT_ROOT / "configs/v3/gate3/gse_geometry_anchored_spatial_event_readiness_v1.json"
DATA_CARD = "configs/v3/gate3/data_cards/gse_geometry_anchored_spatial_event_readiness_v1.json"
PYTHON = "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    if SPEC.exists():
        raise RuntimeError("geometry-anchored readiness spec already exists")
    teacher = "results/gate2_representation/gate2_20260828_gse_spatial_multi_event_teacher_export_v1_seed0"
    source = "results/gate2_representation/gate2_20260824_gse_deduplicated_dataset_export_v1_seed0"
    inputs = [
        DATA_CARD,
        f"{teacher}/RUN_STATE.json",
        f"{teacher}/metrics/summary.json",
        f"{teacher}/artifacts/evidence_sha256.txt",
        f"{teacher}/artifacts/export/summary.json",
        f"{teacher}/artifacts/export/teacher_shard_manifest.jsonl",
        f"{source}/RUN_STATE.json",
        f"{source}/artifacts/evidence_sha256.txt",
        f"{source}/artifacts/shard_manifest.json",
    ]
    tools = {
        "model": "src/mtare_topo/representation/gse_geometry_anchored_event.py",
        "set_loss": "src/mtare_topo/representation/gse_spatial_event_set.py",
        "executor": "tools/v3/execute_gse_geometry_anchored_spatial_event_readiness_v1.py",
        "runner": "tools/v3/run_gse_geometry_anchored_spatial_event_readiness_v1.py",
        "freezer": "tools/v3/freeze_gse_geometry_anchored_spatial_event_readiness_spec_v1.py",
        "tests": "tests/v3/unit/test_gse_geometry_anchored_event.py",
        "set_tests": "tests/v3/unit/test_gse_spatial_event_set.py",
        "governance": "src/mtare_topo/governance.py",
        "preflight": "tools/v3/preflight.py",
        "create_run": "tools/v3/create_run.py",
    }
    authorization = {
        "status": "APPROVED",
        "approved_by": "user-standing-authorization",
        "approved_at": "2026-08-28T20:00:00+08:00",
        "authorized_gates": [3],
        "authorized_operations": ["audit"],
        "scope": "One immutable zero-training C01-C08 geometry-anchored encoder readiness proof; no C09/C10/M-TARE, graph or planner access.",
        "confirmation_reference": "User instructed automatic best-choice execution without routine approval prompts.",
    }
    spec = {
        "schema_version": "v3_run_spec_v1",
        "gate": 3,
        "execution_phase": 3,
        "date": "20260828",
        "slug": "gse_geometry_anchored_spatial_event_readiness_v1",
        "seed": 0,
        "operation": "audit",
        "data_card": DATA_CARD,
        "question": "Can a pose-free coordinate-aware polar encoder express every frozen C01-C08 terminal/junction token inside current LiDAR free-range support while satisfying deterministic circular set-prediction contracts before training?",
        "method": "Audit every Teacher/source join and current scan support, then run a 264134-parameter random-initialized circular encoder on real cardinalities 0-5 to verify range/valid-only inputs, free-range-bounded xyz, rotation, query permutation, determinism and finite backward.",
        "baseline": "The sealed frozen old encoder plus unconstrained set decoder failed selection precision/recall/F1 and all-query typed 4 m coverage.",
        "fallback": "If support or typed interface readiness fails, stop before training and revise only the geometry-anchored representation contract; do not change Teacher, split or downstream graph.",
        "user_authorization": authorization,
        "acceptance_criteria": [
            "Exact 80 worlds, 188126 unique observations, 133055 tokens, 1076 identities and frozen type/cardinality distributions.",
            "Every Teacher event is within current organized-LiDAR free-range support in its polar cell.",
            "Only five-frame range/valid input; 264134 parameters; real cardinality 0-5 finite loss/backward.",
            "Query-permutation loss <=1e-6, 18-degree rotation xyz error <=1mm, invariant error <=2e-5, deterministic replay exact and predicted radial range never exceeds its free-range anchor.",
            "Zero optimization, trained inference, threshold selection, C09/C10/M-TARE, graph or planner access; sources unchanged.",
        ],
        "expected_counts": {
            "worlds": 80,
            "fit_worlds": 60,
            "selection_worlds": 20,
            "observations": 188126,
            "tokens": 133055,
            "event_identities": 1076,
            "parameters": 264134,
            "optimizer_steps": 0,
            "trained_model_inference_frames": 0,
            "c09_worlds_read": 0,
            "c10_worlds_read": 0,
            "mtare_worlds_read": 0,
        },
        "expected_evidence": [
            "Full 80-world Teacher/source/support audit log.",
            "Machine-readable support margins and any failing examples.",
            "Interface/equivariance/permutation/determinism/gradient summary, environment, RUN_STATE and SHA-256 seal.",
        ],
        "estimated_cost": {
            "compute": "Sequential CPU read of each current C01-C08 organized scan and four six-row random-initialized forwards plus one backward; no GPU.",
            "wall_time_hours": 0.25,
            "host_ram_gb": 4,
            "gpu_memory_gb": 0,
            "disk_gb": 0.05,
            "gpu": "none",
        },
        "frozen_inputs": {path: sha256(PROJECT_ROOT / path) for path in sorted(inputs)},
        "frozen_tools": {
            name: {"path": path, "sha256": sha256(PROJECT_ROOT / path)}
            for name, path in tools.items()
        },
        "working_directory": str(PROJECT_ROOT),
        "command": [
            "/usr/bin/timeout",
            "2100s",
            PYTHON,
            "tools/v3/run_gse_geometry_anchored_spatial_event_readiness_v1.py",
            "--spec",
            str(SPEC),
            "--run-dir",
            str(PROJECT_ROOT / f"results/gate3_semantics/{RUN_ID}"),
        ],
    }
    write_json(SPEC, spec)
    print(SPEC.relative_to(PROJECT_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
