#!/usr/bin/env python3
"""Freeze the formal axis-anchored event-relation Teacher feasibility spec."""
from __future__ import annotations

import hashlib

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, validate_data_card, write_json


RUN_ID = "gate3_20260829_gse_axis_anchored_event_relation_teacher_feasibility_v1_seed0"
SPEC = PROJECT_ROOT / "configs/v3/gate3/gse_axis_anchored_event_relation_teacher_feasibility_v1.json"
DATA_CARD = "configs/v3/gate3/data_cards/gse_axis_anchored_event_relation_teacher_feasibility_v1.json"
PYTHON = "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    if SPEC.exists():
        raise RuntimeError("axis-anchored Teacher feasibility spec already exists")
    card = load_json(PROJECT_ROOT / DATA_CARD)
    validation = validate_data_card(card)
    if not validation.passed:
        raise RuntimeError(f"Teacher feasibility card invalid: {validation.errors}")
    dataset = "results/gate2_representation/gate2_20260824_gse_deduplicated_dataset_export_v1_seed0"
    exit_action = "results/gate3_semantics/gate3_20260828_gse_exit_action_transport_feasibility_v1_seed0"
    commit = "results/gate3_semantics/gate3_20260829_gse_relational_commit_policy_feasibility_v1_seed0"
    inputs = [
        DATA_CARD,
        f"{dataset}/RUN_STATE.json", f"{dataset}/metrics/summary.json", f"{dataset}/metrics/runner_summary.json",
        f"{dataset}/artifacts/sequence_manifest.jsonl", f"{dataset}/artifacts/shard_manifest.json",
        f"{dataset}/artifacts/association_identity_map.json", f"{dataset}/artifacts/exit_identity_map.json",
        f"{dataset}/artifacts/evidence_sha256.txt",
        f"{exit_action}/RUN_STATE.json", f"{exit_action}/metrics/summary.json", f"{exit_action}/artifacts/evidence_sha256.txt",
        f"{commit}/RUN_STATE.json", f"{commit}/metrics/summary.json", f"{commit}/metrics/feasibility/summary.json", f"{commit}/artifacts/evidence_sha256.txt",
        "configs/v3/gate2/environments/phase3_torch290_cu129_zarr2187_v1r2.json",
    ]
    tools = {
        "executor": "tools/v3/execute_gse_axis_anchored_event_relation_teacher_feasibility_v1.py",
        "runner": "tools/v3/run_gse_axis_anchored_event_relation_teacher_feasibility_v1.py",
        "freezer": "tools/v3/freeze_gse_axis_anchored_event_relation_teacher_feasibility_spec_v1.py",
        "tests": "tests/v3/unit/test_gse_axis_anchored_event_relation_teacher_feasibility.py",
        "event_contract": "src/mtare_topo/semantics/geometric_semantics.py",
        "governance": "src/mtare_topo/governance.py",
        "preflight": "tools/v3/preflight.py",
        "create_run": "tools/v3/create_run.py",
    }
    spec = {
        "schema_version": "v3_run_spec_v1", "gate": 3, "execution_phase": 3, "date": "20260829",
        "slug": "gse_axis_anchored_event_relation_teacher_feasibility_v1", "seed": 0,
        "operation": "audit", "data_card": DATA_CARD,
        "question": "Does the sealed C01-C08 Teacher provide sufficient causal route anchoring, independent event identities, cross-view relation stars, temporal branch changes and geometry masks to justify implementing Axis-Anchored Geometry-Semantic Event Relation?",
        "method": "Read each sealed C01-C08 observation once; audit event/cardinality/geometry populations, sensor-forward route-axis causality, exact canonical branch stars per association identity, junction route diversity, persistent/reveal/withdraw relations, visible-exit separation and frozen prior descriptor/commit evidence.",
        "baseline": "The failed one-shot complete-set family plus the old high-precision/low-recall hand-written stateful commit policy, retained only as ablations.",
        "fallback": "If any Teacher, split, identity, geometry or causal-anchor contract fails, stop this candidate before training; do not repair labels or compensate in graph/planner.",
        "user_authorization": card["approval"],
        "acceptance_criteria": [
            "Exactly 80 worlds, 252430 raw frames, 188126 observations, 396913 visible exits and 4493 association identities.",
            "Route axis has maximum absolute lateral component <=1e-6 and minimum forward component >0.85 in the sensor frame.",
            "Every event identity has one exact canonical full branch star; every junction identity has at least four route views.",
            "At least 95% of fit and selection junction identities have K3/K4 canonical stars; report all locally K2/provisional identities.",
            "Every structural event has at least 50 selection identities; both partitions contain predeclared reveal/withdraw supervision.",
            "Global geometry masks cover at least 98%; frozen descriptor transport precision/recall are at least 0.98.",
            "Five unit tests pass; full 33083-entry dataset seal verifies; RAM <=8 GiB; zero training/new inference/checkpoint selection/C09/C10/M-TARE/graph/planner; sources unchanged."
        ],
        "expected_counts": {
            "worlds": 80, "raw_frames": 252430, "fit_observations": 142184,
            "selection_observations": 45942, "observations": 188126,
            "visible_exit_tokens": 396913, "association_identities": 4493,
            "unit_tests": 5, "optimizer_steps": 0, "new_model_inference_observations": 0,
            "checkpoint_selection_observations": 0, "c09_worlds_read": 0, "c10_worlds_read": 0,
            "mtare_worlds_read": 0, "graph_replays": 0
        },
        "expected_evidence": [
            "Exact event/identity/temporal/geometry populations and all predeclared feasibility checks.",
            "Identity and event CSVs plus paper-ready PNG/PDF/SVG and figure source.",
            "Unit-test and raw audit logs, environment, source-integrity record, RUN_STATE and SHA-256 seal."
        ],
        "estimated_cost": {"compute": "CPU-only deterministic Teacher feasibility audit including full source-seal verification", "wall_time_hours": 0.1, "host_ram_gb": 8, "gpu_memory_gb": 0, "disk_gb": 0.1},
        "frozen_inputs": {path: sha256(PROJECT_ROOT / path) for path in sorted(inputs)},
        "frozen_tools": {name: {"path": path, "sha256": sha256(PROJECT_ROOT / path)} for name, path in tools.items()},
        "working_directory": str(PROJECT_ROOT),
        "command": [
            "/usr/bin/timeout", "7200s", PYTHON,
            "tools/v3/run_gse_axis_anchored_event_relation_teacher_feasibility_v1.py",
            "--spec", str(SPEC),
            "--run-dir", str(PROJECT_ROOT / f"results/gate3_semantics/{RUN_ID}"),
        ],
    }
    write_json(SPEC, spec)
    print(SPEC.relative_to(PROJECT_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
