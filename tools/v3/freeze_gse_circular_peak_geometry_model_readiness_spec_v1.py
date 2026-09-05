#!/usr/bin/env python3
"""Freeze one immutable circular peak geometry model readiness."""

from __future__ import annotations

import hashlib

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import write_json


RUN_ID = "gate3_20260829_gse_circular_peak_geometry_model_readiness_v1_seed0"
SPEC = PROJECT_ROOT / "configs/v3/gate3/gse_circular_peak_geometry_model_readiness_v1.json"
DATA_CARD = "configs/v3/gate3/data_cards/gse_circular_peak_geometry_model_readiness_v1.json"
PYTHON = "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    if SPEC.exists():
        raise RuntimeError("circular peak model readiness V1 spec already exists")
    dataset = "results/gate2_representation/gate2_20260824_gse_deduplicated_dataset_export_v1_seed0"
    teacher = "results/gate2_representation/gate2_20260829_gse_circular_exit_geometry_field_teacher_export_v1_seed0"
    inputs = [
        DATA_CARD,
        f"{dataset}/RUN_STATE.json", f"{dataset}/metrics/summary.json",
        f"{dataset}/artifacts/shard_manifest.json", f"{dataset}/artifacts/evidence_sha256.txt",
        f"{teacher}/RUN_STATE.json", f"{teacher}/metrics/summary.json",
        f"{teacher}/artifacts/export/teacher_shard_manifest.jsonl", f"{teacher}/artifacts/evidence_sha256.txt",
        "configs/v3/gate2/environments/phase3_torch290_cu129_zarr2187_v1r2.json",
    ]
    tools = {
        "model": "src/mtare_topo/representation/gse_circular_peak_geometry_model.py",
        "circular_layers": "src/mtare_topo/representation/phase3_structural_semantics.py",
        "executor": "tools/v3/execute_gse_circular_peak_geometry_model_readiness_v1.py",
        "runner": "tools/v3/run_gse_circular_peak_geometry_model_readiness_v1.py",
        "freezer": "tools/v3/freeze_gse_circular_peak_geometry_model_readiness_spec_v1.py",
        "model_tests": "tests/v3/unit/test_gse_circular_peak_geometry_model.py",
        "teacher_tests": "tests/v3/unit/test_gse_circular_exit_geometry_field.py",
        "governance": "src/mtare_topo/governance.py",
        "preflight": "tools/v3/preflight.py",
        "create_run": "tools/v3/create_run.py",
    }
    spec = {
        "schema_version": "v3_run_spec_v1",
        "gate": 3,
        "execution_phase": 3,
        "date": "20260829",
        "slug": "gse_circular_peak_geometry_model_readiness_v1",
        "seed": 0,
        "operation": "audit",
        "data_card": DATA_CARD,
        "question": "Can one five-frame causal circular LiDAR model expose the frozen 180-bin exit peaks and full tunnel geometry with correct masks, equivariance and finite real-batch backward before training?",
        "method": "Zero-training 769268-parameter circular backbone with dense peak presence/residual/width/profile/descriptor/uncertainty and local axis plus width/height/slope/curvature heads.",
        "baseline": "Bound raw current-range envelope AP 0.052647/0.051493; no performance baseline is rerun in readiness.",
        "fallback": "Any real-data join, causality, free-query, rotation, mask or finite-backward failure stops this model before training.",
        "user_authorization": {
            "status": "APPROVED", "approved_by": "user-standing-authorization",
            "approved_at": "2026-08-29T03:20:00+08:00", "authorized_gates": [3],
            "authorized_operations": ["audit"],
            "scope": "One immutable zero-training C01-C08 model readiness; no checkpoint, test world or graph.",
            "confirmation_reference": "User explicitly delegated best-method choices and continuous execution."
        },
        "acceptance_criteria": [
            "Exact 80 worlds, 188126 observations, 396913 peaks and fit/selection/cardinality counts.",
            "All Teacher/source joins and five-frame references exact, contiguous and past-only; no hidden Teacher LiDAR/pose/identity.",
            "Forward accepts scans only, contains no free-query parameter and emits the full typed structural geometry interface.",
            "40-column roll gives 10-bin dense roll, rotated local axis and invariant global outputs with max error <=3e-5.",
            "Balanced/masked real-batch losses and all parameter gradients are finite; invalid targets have exact zero effect.",
            "Zero optimizer/checkpoint/trained inference/normalization/threshold/graph/C09/C10/M-TARE."
        ],
        "expected_counts": {
            "worlds": 80, "raw_frames": 252430, "observations": 188126,
            "fit_observations": 142184, "selection_observations": 45942,
            "visible_exit_peaks": 396913, "fit_visible_exit_peaks": 299872,
            "selection_visible_exit_peaks": 97041, "width_valid_peaks": 389026,
            "width_invalid_peaks": 7887, "bearing_bins": 180,
            "real_readiness_rows": 8, "parameters": 769268, "optimizer_steps": 0,
            "model_inference_frames": 0, "checkpoint_writes": 0,
            "threshold_selection_steps": 0, "graph_replays": 0,
            "c09_worlds_read": 0, "c10_worlds_read": 0, "mtare_worlds_read": 0
        },
        "expected_evidence": [
            "Exact population/reference/join and hidden-input audit.",
            "Typed shapes, parameter inventory, circular/batch/repeat errors and real-batch losses.",
            "Readiness check CSV and PNG/PDF/SVG/source.",
            "Environment, command, log, metrics, RUN_STATE, source-integrity records and seal."
        ],
        "estimated_cost": {
            "compute": "CPU-only full reference audit and eight-row forward/backward",
            "wall_time_hours": 0.08, "host_ram_gb": 4, "gpu_memory_gb": 0,
            "disk_gb": 0.1, "gpu": "none"
        },
        "frozen_inputs": {path: sha256(PROJECT_ROOT / path) for path in sorted(inputs)},
        "frozen_tools": {name: {"path": path, "sha256": sha256(PROJECT_ROOT / path)} for name, path in tools.items()},
        "working_directory": str(PROJECT_ROOT),
        "command": [
            "/usr/bin/timeout", "1200s", PYTHON,
            "tools/v3/run_gse_circular_peak_geometry_model_readiness_v1.py",
            "--spec", str(SPEC),
            "--run-dir", str(PROJECT_ROOT / f"results/gate3_semantics/{RUN_ID}")
        ]
    }
    write_json(SPEC, spec)
    print(SPEC.relative_to(PROJECT_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
