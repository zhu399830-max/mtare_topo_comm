#!/usr/bin/env python3
"""Freeze one immutable circular exit-geometry field feasibility run."""

from __future__ import annotations

import hashlib

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import write_json


RUN_ID = "gate3_20260829_gse_circular_exit_geometry_field_feasibility_v1_seed0"
SPEC = PROJECT_ROOT / "configs/v3/gate3/gse_circular_exit_geometry_field_feasibility_v1.json"
DATA_CARD = "configs/v3/gate3/data_cards/gse_circular_exit_geometry_field_feasibility_v1.json"
PYTHON = "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    if SPEC.exists():
        raise RuntimeError("circular field feasibility spec already exists")
    dataset = "results/gate2_representation/gate2_20260824_gse_deduplicated_dataset_export_v1_seed0"
    exit_audit = "results/gate2_representation/gate2_20260824_gse_exit_token_audit_v1r_seed0"
    motivation = "results/gate3_semantics/gate3_20260829_gse_token_validity_action_track_feasibility_v1r2_seed0"
    inputs = [
        DATA_CARD,
        f"{dataset}/RUN_STATE.json", f"{dataset}/metrics/summary.json", f"{dataset}/artifacts/shard_manifest.json", f"{dataset}/artifacts/evidence_sha256.txt",
        f"{exit_audit}/RUN_STATE.json", f"{exit_audit}/metrics/summary.json", f"{exit_audit}/artifacts/exit_token_audit.jsonl", f"{exit_audit}/artifacts/evidence_sha256.txt",
        f"{motivation}/RUN_STATE.json", f"{motivation}/metrics/summary.json", f"{motivation}/artifacts/evidence_sha256.txt",
    ]
    tools = {
        "representation": "src/mtare_topo/representation/gse_circular_exit_geometry_field.py",
        "ranking_metrics": "src/mtare_topo/evaluation/gse_token_validity_feasibility.py",
        "executor": "tools/v3/execute_gse_circular_exit_geometry_field_feasibility_v1.py",
        "runner": "tools/v3/run_gse_circular_exit_geometry_field_feasibility_v1.py",
        "freezer": "tools/v3/freeze_gse_circular_exit_geometry_field_feasibility_spec_v1.py",
        "tests": "tests/v3/unit/test_gse_circular_exit_geometry_field.py",
        "governance": "src/mtare_topo/governance.py", "preflight": "tools/v3/preflight.py", "create_run": "tools/v3/create_run.py",
    }
    spec = {
        "schema_version": "v3_run_spec_v1", "gate": 3, "execution_phase": 3, "date": "20260829", "slug": "gse_circular_exit_geometry_field_feasibility_v1", "seed": 0,
        "operation": "audit", "data_card": DATA_CARD,
        "question": "Can every C01-C08 visible directed exit be represented as a unique 2-degree circular geometry peak without identity, future frames or free-query objectness?",
        "method": "Verify all sources; rasterize 180 center-peak bins with signed residual/width-mask/profile; audit collision, exact round-trip, slot permutation, 20-degree roll and five-frame causality; compare wide-sector components and a current-scan raw-range envelope.",
        "baseline": "Wide opening sectors decoded as circular connected components and a no-learning current-scan free-range envelope.",
        "fallback": "Any center collision, non-lossless geometry, rotation/causality failure or hidden identity/future dependence stops the representation before Teacher export.",
        "user_authorization": {"status": "APPROVED", "approved_by": "user-standing-authorization", "approved_at": "2026-08-29T02:10:00+08:00", "authorized_gates": [3], "authorized_operations": ["audit"], "scope": "One immutable zero-training C01-C08 circular peak-field representation audit.", "confirmation_reference": "User explicitly delegated best-method choices and continuous execution."},
        "acceptance_criteria": ["Exact 80 worlds, 252430 raw frames, 188126 observations, 396913 visible exits and 389026/7887 valid/invalid widths.", "Zero same-bin and adjacent-bin center collisions; exact 1/2/3/4 cardinalities 7525/154279/24458/1864.", "Heading/rotation residual error<=1e-4 degree; width/profile exact; slot permutation invariant; all five-frame references contiguous and past-only.", "Reproduce rejected wide-sector 22626 component mismatch and 22381 overlap rows; raw-range ranking is diagnostic only.", "Zero Teacher export/optimizer/new inference/threshold fitting/graph/C09/C10/M-TARE; source unchanged and complete evidence."],
        "expected_counts": {"worlds": 80, "raw_frames": 252430, "observations": 188126, "fit_observations": 142184, "selection_observations": 45942, "visible_exit_tokens": 396913, "width_valid_tokens": 389026, "width_invalid_tokens": 7887, "bearing_bins_per_observation": 180, "same_bin_collision_rows": 0, "adjacent_bin_collision_rows": 0, "wide_sector_component_mismatch_rows": 22626, "wide_sector_overlap_rows": 22381, "optimizer_steps": 0, "teacher_export_rows": 0, "model_inference_frames": 0, "threshold_selection_steps": 0, "graph_replays": 0, "c09_worlds_read": 0, "c10_worlds_read": 0, "mtare_worlds_read": 0},
        "expected_evidence": ["Exact collision/cardinality/geometry/rotation/causality metrics.", "Wide-sector failure attribution and raw-range baseline ranking.", "Representation comparison CSV and PNG/PDF/SVG with exact JSON source.", "Raw log, environment, RUN_STATE, input/tool hashes and seal."],
        "estimated_cost": {"compute": "CPU read-only full shard and 547MB streaming representation audit", "wall_time_hours": 0.15, "host_ram_gb": 6, "gpu_memory_gb": 0, "disk_gb": 0.1, "gpu": "none"},
        "frozen_inputs": {path: sha256(PROJECT_ROOT / path) for path in sorted(inputs)},
        "frozen_tools": {name: {"path": path, "sha256": sha256(PROJECT_ROOT / path)} for name, path in tools.items()},
        "working_directory": str(PROJECT_ROOT),
        "command": ["/usr/bin/timeout", "1800s", PYTHON, "tools/v3/run_gse_circular_exit_geometry_field_feasibility_v1.py", "--spec", str(SPEC), "--run-dir", str(PROJECT_ROOT / f"results/gate3_semantics/{RUN_ID}")],
    }
    write_json(SPEC, spec)
    print(SPEC.relative_to(PROJECT_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
