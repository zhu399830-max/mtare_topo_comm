#!/usr/bin/env python3
"""Freeze one immutable circular exit-geometry peak-field Teacher export."""

from __future__ import annotations

import hashlib

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import write_json


RUN_ID = "gate2_20260829_gse_circular_exit_geometry_field_teacher_export_v1_seed0"
SPEC = PROJECT_ROOT / "configs/v3/gate2/gse_circular_exit_geometry_field_teacher_export_v1.json"
DATA_CARD = "configs/v3/gate2/data_cards/gse_circular_exit_geometry_field_teacher_export_v1.json"
PYTHON = "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    if SPEC.exists():
        raise RuntimeError("circular Teacher export V1 spec already exists")
    dataset = "results/gate2_representation/gate2_20260824_gse_deduplicated_dataset_export_v1_seed0"
    feasibility = "results/gate3_semantics/gate3_20260829_gse_circular_exit_geometry_field_feasibility_v1r_seed0"
    inputs = [
        DATA_CARD,
        f"{dataset}/RUN_STATE.json",
        f"{dataset}/metrics/summary.json",
        f"{dataset}/artifacts/shard_manifest.json",
        f"{dataset}/artifacts/evidence_sha256.txt",
        f"{feasibility}/RUN_STATE.json",
        f"{feasibility}/metrics/summary.json",
        f"{feasibility}/artifacts/audit/summary.json",
        f"{feasibility}/artifacts/evidence_sha256.txt",
    ]
    tools = {
        "representation": "src/mtare_topo/representation/gse_circular_exit_geometry_field.py",
        "executor": "tools/v3/execute_gse_circular_exit_geometry_field_teacher_export_v1.py",
        "runner": "tools/v3/run_gse_circular_exit_geometry_field_teacher_export_v1.py",
        "freezer": "tools/v3/freeze_gse_circular_exit_geometry_field_teacher_export_spec_v1.py",
        "tests": "tests/v3/unit/test_gse_circular_exit_geometry_field.py",
        "governance": "src/mtare_topo/governance.py",
        "preflight": "tools/v3/preflight.py",
        "create_run": "tools/v3/create_run.py",
    }
    spec = {
        "schema_version": "v3_run_spec_v1",
        "gate": 2,
        "execution_phase": 2,
        "date": "20260829",
        "slug": "gse_circular_exit_geometry_field_teacher_export_v1",
        "seed": 0,
        "operation": "teacher_generation",
        "data_card": DATA_CARD,
        "question": "Can the qualified circular exit-geometry peak field be exported exactly once for all 188126 C01-C08 observations without duplicating LiDAR or storing identity?",
        "method": "One compressed Zarr Teacher shard per development world containing 180-bin peak presence, sub-bin heading residual, width/mask, vertical profile and causal references.",
        "baseline": "The bound feasibility run already contains the wide-sector and raw current-range comparisons; no baseline is rerun during export.",
        "fallback": "Any input drift, count mismatch, forbidden array, non-exact round trip, heading error or oversized export stops before model readiness.",
        "user_authorization": {
            "status": "APPROVED",
            "approved_by": "user-standing-authorization",
            "approved_at": "2026-08-29T02:50:00+08:00",
            "authorized_gates": [2],
            "authorized_operations": ["teacher_generation"],
            "scope": "One immutable C01-C08 circular peak-field Teacher export; no training, inference, test-world access or graph replay.",
            "confirmation_reference": "User explicitly delegated best-method decisions and continuous execution.",
        },
        "acceptance_criteria": [
            "Exactly 80 shards, 188126 observations, 396913 peaks and 389026/7887 valid/invalid width targets.",
            "Fit/selection remain 60/20 worlds, 142184/45942 observations and 299872/97041 peaks.",
            "Every array reopens elementwise equal; canonical heading round trip is at most 1e-4 degree.",
            "No LiDAR, valid-scan, pose, node/exit identity, future target or test-world content is exported.",
            "Total compressed Teacher size remains below 1 GiB and all source files remain unchanged.",
            "Zero optimizer, inference, normalization, threshold selection, graph replay, C09, C10 or M-TARE access.",
        ],
        "expected_counts": {
            "worlds": 80,
            "fit_worlds": 60,
            "selection_worlds": 20,
            "raw_frames": 252430,
            "observations": 188126,
            "fit_observations": 142184,
            "selection_observations": 45942,
            "visible_exit_tokens": 396913,
            "fit_visible_exit_tokens": 299872,
            "selection_visible_exit_tokens": 97041,
            "width_valid_tokens": 389026,
            "width_invalid_tokens": 7887,
            "bearing_bins_per_observation": 180,
            "teacher_shards": 80,
            "optimizer_steps": 0,
            "model_inference_frames": 0,
            "normalization_steps": 0,
            "threshold_selection_steps": 0,
            "graph_replays": 0,
            "c09_worlds_read": 0,
            "c10_worlds_read": 0,
            "mtare_worlds_read": 0,
        },
        "expected_evidence": [
            "80 compressed Teacher shards and exact per-shard tree-hash manifest.",
            "Population, partition, cardinality, round-trip and forbidden-content checks.",
            "PNG/PDF/SVG summary figure and machine-readable figure source.",
            "Environment, commands, raw log, metrics, RUN_STATE and SHA-256 seal.",
        ],
        "estimated_cost": {
            "compute": "CPU-only encode, compressed Zarr write and full readback",
            "wall_time_hours": 0.2,
            "host_ram_gb": 4,
            "gpu_memory_gb": 0,
            "disk_gb": 1,
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
            "1800s",
            PYTHON,
            "tools/v3/run_gse_circular_exit_geometry_field_teacher_export_v1.py",
            "--spec",
            str(SPEC),
            "--run-dir",
            str(PROJECT_ROOT / f"results/gate2_representation/{RUN_ID}"),
        ],
    }
    write_json(SPEC, spec)
    print(SPEC.relative_to(PROJECT_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
