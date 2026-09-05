#!/usr/bin/env python3
"""Freeze the one-off objective spatial graph rescore audit."""

from __future__ import annotations

import hashlib

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import write_json


RUN_ID = "gate3_20260828_gse_objective_spatial_graph_rescore_audit_v1_seed0"
SPEC = PROJECT_ROOT / "configs/v3/gate3/gse_objective_spatial_graph_rescore_audit_v1.json"
CARD = PROJECT_ROOT / "configs/v3/gate3/data_cards/gse_objective_spatial_graph_rescore_audit_v1.json"
PYTHON = "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python"


def _sha(path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    if SPEC.exists() or CARD.exists():
        raise RuntimeError("objective spatial graph rescore card/spec exists; overwrite is forbidden")
    predecessor = "results/gate3_semantics/gate3_20260828_gse_projected_trace_commit_capacity_v1_seed0"
    spatial = "results/gate3_semantics/gate3_20260828_gse_spatial_trace_commit_requalification_v1r2_seed0"
    teacher_npz = "results/gate3_semantics/gate3_20260828_gse_event_center_offset_training_v1r3_seed0/artifacts/teacher/event_center_teacher.npz"
    teacher_jsonl = "results/gate2_representation/gate2_20260827_gse_corrected_causal_teacher_manifest_v1r_seed0/artifacts/teacher_observations.jsonl"
    approval = {
        "status": "APPROVED", "approved_by": "user-standing-authorization",
        "approved_at": "2026-08-28T16:10:00+08:00", "authorized_gates": [3],
        "authorized_operations": ["audit"],
        "scope": "One immutable read-only C07-C08 objective spatial graph rescore audit; no runtime/model/Teacher mutation and zero C09/C10/M-TARE.",
        "confirmation_reference": "User instructed automatic best-choice execution without routine approval prompts.",
    }
    card = {
        "schema_version": "v3_data_card_v1",
        "card_id": "gse_objective_spatial_graph_rescore_audit_v1",
        "title": "Objective 3D one-to-one graph rescore audit",
        "status": "APPROVED_FOR_ONE_IMMUTABLE_GSE_OBJECTIVE_SPATIAL_GRAPH_RESCORE_AUDIT_V1",
        "approval": approval,
        "purpose": "Determine whether exact trigger-row Teacher identity incorrectly penalizes spatially correct decision nodes, without changing either sealed replay.",
        "operation": "read-only post-replay scoring audit and paper figure generation",
        "worlds": {
            "audit": "20 disjoint development-selection worlds S01-S10 C07-C08",
            "forbidden": "C09, C10, strict test and all M-TARE worlds",
        },
        "sampling": {
            "raw_observations": 188126, "selection_observations": 45942,
            "objective_decision_rows": 8839, "objective_node_identities": 274,
            "objective_trace_relations": 13,
            "independent_unit": "Objective TNG decision-node identity and observed verified relation",
        },
        "trajectories": {
            "source": "All sealed C07-C08 directed traversals already replayed",
            "spatial_spacing_m": 1.0,
            "temporal_context": "Current plus four strictly past observations; no future frame",
        },
        "split_logic": "Post-hoc audit only on the already frozen C07-C08 selection partition; no selection, normalization, threshold or checkpoint change.",
        "teacher": {
            "source": "Sealed objective TNG decision-node identity, event, exact 3D center and observed traversal relation",
            "usage": "Only after both replays are complete; zero runtime access",
            "center_uniqueness": "Each of 274 objective identities must have one bitwise-identical 3D center across all valid rows.",
        },
        "methods": {
            "main": "Arithmetic mean of each committed hypothesis's frozen evidence-row centers, followed by maximum-cardinality/minimum-distance one-to-one matching within identical world/event and the unchanged 4 m cap; verified edges score only through this mapping.",
            "baseline": "Legacy exact evidence-row identity score retained side-by-side for both scalar and spatial-center graphs.",
            "fallback": "If scorer population, uniqueness, one-to-one or reproduction contracts fail, seal system FAIL and retain the legacy conclusion.",
        },
        "acceptance": {
            "population": "Exactly 188,126 total and 45,942 selection observations, 20 worlds, 274 objective nodes and 13 objective relations.",
            "matching": "World/event are hard constraints; distance <=4.0 m; one predicted and one objective node at most once; duplicates and unmatched predictions remain false positives.",
            "isolation": "Teacher is read only after replay; optimizer/model inference/update and C09/C10/M-TARE reads are all zero.",
            "evidence": "Per-node/per-edge JSONL, old/new CSV, PNG/PDF/SVG figure, provenance, logs, metrics and SHA-256 seal.",
        },
        "leakage_audit": {
            "future_frames": 0, "objective_identity_in_runtime": 0,
            "c09_worlds": 0, "c10_worlds": 0, "mtare_worlds": 0,
        },
        "estimated_cost": {"compute": "CPU read-only", "wall_time_hours": 0.25, "host_ram_gb": 4, "gpu_memory_gb": 0, "disk_gb": 0.1},
        "retention": "Retain all compact tables, vector/raster paper figure, figure source, provenance, logs and seal; duplicate no sensor payload.",
        "sources": {
            "predecessor": predecessor, "spatial": spatial,
            "objective_teacher": teacher_npz, "observation_teacher": teacher_jsonl,
        },
    }
    write_json(CARD, card)

    inputs = [teacher_npz, teacher_jsonl]
    for run in (predecessor, spatial):
        inputs.extend([
            f"{run}/RUN_STATE.json", f"{run}/artifacts/evidence_sha256.txt",
            f"{run}/artifacts/replay/summary.json",
            f"{run}/artifacts/replay/verified_nodes.jsonl",
            f"{run}/artifacts/replay/verified_edges.jsonl",
        ])
    inputs.extend([
        f"{predecessor}/artifacts/projection/event_center_projection.npz",
        f"{spatial}/artifacts/projection/spatial_center_ensemble_all_rows.npz",
    ])
    tools = {
        "data_card": str(CARD.relative_to(PROJECT_ROOT)),
        "scorer": "src/mtare_topo/evaluation/gse_objective_spatial_graph_score.py",
        "relation_helper": "src/mtare_topo/evaluation/gse_trace_commit_failure_funnel.py",
        "executor": "tools/v3/execute_gse_objective_spatial_graph_rescore_audit_v1.py",
        "runner": "tools/v3/run_gse_objective_spatial_graph_rescore_audit_v1.py",
        "freezer": "tools/v3/freeze_gse_objective_spatial_graph_rescore_audit_spec_v1.py",
        "unit_test": "tests/v3/unit/test_gse_objective_spatial_graph_score.py",
    }
    spec = {
        "schema_version": "v3_run_spec_v1", "gate": 3, "operation": "audit",
        "date": "20260828", "slug": "gse_objective_spatial_graph_rescore_audit_v1", "seed": 0,
        "question": "Does objective 3D one-to-one node matching change the scientific conclusion of the sealed trace-commit graphs?",
        "config_path": str(CARD.relative_to(PROJECT_ROOT)),
        "data_card": str(CARD.relative_to(PROJECT_ROOT)), "user_authorization": approval,
        "estimated_resources": {"wall_time_minutes": 15, "ram_gb": 4, "disk_gb": 0.1, "gpu_count": 0},
        "acceptance_criteria": list(card["acceptance"].values()),
        "frozen_inputs": {relative: _sha(PROJECT_ROOT / relative) for relative in inputs},
        "frozen_tools": {
            name: {"path": relative, "sha256": _sha(PROJECT_ROOT / relative)}
            for name, relative in tools.items()
        },
        "expected_evidence": [
            "Legacy and objective-spatial scores for both sealed methods.",
            "One-to-one match distances and unmatched node/edge records.",
            "Machine-readable science gates without automatic Gate advancement.",
            "Paper-ready PNG/PDF/SVG figure with source hashes.",
        ],
        "command": [
            "/usr/bin/timeout", "--signal=INT", "--kill-after=30s", "900s", PYTHON,
            "tools/v3/run_gse_objective_spatial_graph_rescore_audit_v1.py",
            "--spec", str(SPEC),
            "--run-dir", str(PROJECT_ROOT / "results/gate3_semantics" / RUN_ID),
        ],
    }
    write_json(SPEC, spec)
    print(SPEC.relative_to(PROJECT_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
