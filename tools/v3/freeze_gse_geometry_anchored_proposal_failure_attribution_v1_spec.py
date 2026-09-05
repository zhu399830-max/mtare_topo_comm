#!/usr/bin/env python3
from __future__ import annotations

import hashlib

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import write_json


RUN_ID = "gate3_20260828_gse_geometry_anchored_proposal_failure_attribution_v1_seed0"
SPEC = PROJECT_ROOT / "configs/v3/gate3/gse_geometry_anchored_proposal_failure_attribution_v1.json"
PYTHON = "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python"
SLUG = "gse_geometry_anchored_proposal_failure_attribution_v1"
EXECUTOR_TOOL = "tools/v3/execute_gse_geometry_anchored_proposal_failure_attribution_v1.py"
RUNNER_TOOL = "tools/v3/run_gse_geometry_anchored_proposal_failure_attribution_v1.py"
FREEZER_TOOL = "tools/v3/freeze_gse_geometry_anchored_proposal_failure_attribution_v1_spec.py"


def _sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    if SPEC.exists():
        raise RuntimeError("proposal failure attribution spec already exists")
    teacher = "results/gate2_representation/gate2_20260828_gse_observable_spatial_event_teacher_v2_seed0"
    capacity = "results/gate3_semantics/gate3_20260828_gse_geometry_anchored_joint_capacity_v1_seed0"
    inputs = [
        f"{teacher}/RUN_STATE.json",
        f"{teacher}/metrics/summary.json",
        f"{teacher}/artifacts/evidence_sha256.txt",
        f"{teacher}/artifacts/export/summary.json",
        f"{teacher}/artifacts/export/teacher_shard_manifest.jsonl",
        f"{capacity}/RUN_STATE.json",
        f"{capacity}/metrics/summary.json",
        f"{capacity}/metrics/capacity/summary.json",
        f"{capacity}/metrics/capacity/figure_source.json",
        f"{capacity}/artifacts/evidence_sha256.txt",
    ]
    for seed in (0, 1, 2):
        inputs.extend((
            f"{capacity}/artifacts/models/seed{seed}/summary.json",
            f"{capacity}/artifacts/models/seed{seed}/selection_outputs.npz",
        ))
    tools = {
        "teacher_loader": "src/mtare_topo/data/gse_observable_spatial_event_dataset.py",
        "metrics": "src/mtare_topo/evaluation/gse_spatial_event_set_metrics.py",
        "executor": EXECUTOR_TOOL,
        "runner": RUNNER_TOOL,
        "freezer": FREEZER_TOOL,
        "executor_base": "tools/v3/execute_gse_geometry_anchored_proposal_failure_attribution_v1.py",
        "runner_base": "tools/v3/run_gse_geometry_anchored_proposal_failure_attribution_v1.py",
        "freezer_base": "tools/v3/freeze_gse_geometry_anchored_proposal_failure_attribution_v1_spec.py",
        "tests": "tests/v3/unit/test_gse_geometry_anchored_proposal_failure_attribution.py",
        "governance": "src/mtare_topo/governance.py",
        "preflight": "tools/v3/preflight.py",
        "create_run": "tools/v3/create_run.py",
    }
    authorization = {
        "status": "APPROVED",
        "approved_by": "user-standing-authorization",
        "approved_at": "2026-08-28T22:55:00+08:00",
        "authorized_gates": [3],
        "authorized_operations": ["audit"],
        "scope": "One immutable read-only C07-C08 attribution of the sealed three-seed geometry-anchored proposal failure.",
        "confirmation_reference": "User instructed automatic best-choice execution without routine approval prompts.",
    }
    spec = {
        "schema_version": "v3_run_spec_v1",
        "gate": 3,
        "execution_phase": 3,
        "date": "20260828",
        "slug": SLUG,
        "seed": 0,
        "operation": "audit",
        "question": "Do the sealed joint models fail because of duplicate proposals, objectness/cardinality, event typing, or insufficient spatial proposal support?",
        "method": "Replay no model: on every sealed C07-C08 prediction, decompose fixed-threshold error, fixed 4m same-type NMS, all-16 typed and position-only one-to-one oracle coverage, empty-row false events, cardinality error, assignment conflict and angular/radial support by target distance.",
        "baseline": "The same-population sealed exclusive single-center baseline with F1=0.390133 and recall=0.262905, plus the three sealed joint results.",
        "fallback": "Choose exactly one predeclared outcome: duplicate, explicit objectness/cardinality, type conditioning, or structured polar proposal. No training or parameter change occurs in this audit.",
        "user_authorization": authorization,
        "hyperparameters": {
            "sealed_thresholds": [0.95, 0.95, 0.95],
            "match_and_nms_radius_m": 4.0,
            "distance_bins_m": [0, 4, 8, 12, 16, 24, 32, 40, 50.0001],
            "f1_gain_gate": 0.05,
            "oracle_recall_gain_gate": 0.10,
        },
        "expected_counts": {
            "worlds": 20,
            "observations": 45942,
            "target_tokens": 33145,
            "target_types": [7578, 25567],
            "cardinality": [19159, 20710, 5786, 285, 2, 0],
            "event_identities": 275,
            "seed_archives": 3,
            "optimizer_steps": 0,
            "model_inference_frames": 0,
            "threshold_selection_steps": 0,
            "c09_worlds_read": 0,
            "c10_worlds_read": 0,
            "mtare_worlds_read": 0,
        },
        "acceptance_criteria": [
            "Exact 20 worlds, 45942 observations, 33145 targets, type/cardinality counts and 275 identities.",
            "All three sealed metrics replay exactly; thresholds remain 0.95 and no prediction, target or source is changed.",
            "Report empty/nonempty false events, cardinality, duplicates, assignment, type, angular/radial and distance-stratified spatial support.",
            "Emit exactly one outcome from frozen NMS, typed-oracle and position-oracle gates.",
            "Zero optimizer, inference, threshold selection, C09/C10/M-TARE, graph or planner; complete paper figure and seal.",
        ],
        "expected_evidence": [
            "Three per-seed selected/NMS/typed-oracle/position-oracle records and explicit error decompositions.",
            "PNG/PDF/SVG diagnostic figure and exact JSON source.",
            "Commands, raw log, source hashes, summary, RUN_STATE and SHA-256 seal.",
        ],
        "estimated_cost": {
            "compute": "CPU read-only vector and deterministic matching audit over three sealed C07-C08 archives; no model inference.",
            "wall_time_hours": 0.15,
            "host_ram_gb": 4,
            "gpu_memory_gb": 0,
            "disk_gb": 0.1,
            "gpu": "none",
        },
        "frozen_inputs": {path: _sha(PROJECT_ROOT / path) for path in sorted(inputs)},
        "frozen_tools": {name: {"path": path, "sha256": _sha(PROJECT_ROOT / path)} for name, path in tools.items()},
        "working_directory": str(PROJECT_ROOT),
        "command": [
            "/usr/bin/timeout",
            "1500s",
            PYTHON,
            RUNNER_TOOL,
            "--spec", str(SPEC),
            "--run-dir", str(PROJECT_ROOT / f"results/gate3_semantics/{RUN_ID}"),
        ],
    }
    write_json(SPEC, spec)
    print(SPEC.relative_to(PROJECT_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
