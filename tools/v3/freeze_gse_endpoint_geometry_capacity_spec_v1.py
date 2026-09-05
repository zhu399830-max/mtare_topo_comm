#!/usr/bin/env python3
"""Freeze the C01-C08 execution-endpoint geometry capacity run."""

from __future__ import annotations

import hashlib

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import write_json


RUN_ID = "gate3_20260828_gse_endpoint_geometry_capacity_v1_seed0"
SPEC = PROJECT_ROOT / "configs/v3/gate3/gse_endpoint_geometry_capacity_v1.json"
CARD = PROJECT_ROOT / "configs/v3/gate3/data_cards/gse_endpoint_geometry_capacity_v1.json"
PYTHON = "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python"


def _sha(path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    if SPEC.exists() or CARD.exists():
        raise RuntimeError("endpoint geometry capacity card/spec exists; overwrite is forbidden")
    teacher_root = "results/gate2_representation/gate2_20260827_gse_corrected_causal_teacher_manifest_v1r_seed0/artifacts"
    dataset_root = "results/gate2_representation/gate2_20260824_gse_deduplicated_dataset_export_v1_seed0"
    replay_root = "results/gate3_semantics/gate3_20260828_gse_spatial_trace_commit_requalification_v1r2_seed0"
    baseline = "results/gate3_semantics/gate3_20260828_gse_post_commit_endpoint_inventory_v1_seed0"
    approval = {
        "status": "APPROVED", "approved_by": "user-standing-authorization",
        "approved_at": "2026-08-28T18:45:00+08:00", "authorized_gates": [3],
        "authorized_operations": ["audit"],
        "scope": "One immutable C01-C08 development-capacity replay using completed route endpoint geometry; zero training/inference/C09/C10/M-TARE.",
        "confirmation_reference": "User instructed automatic best-choice execution without routine approval prompts.",
    }
    card = {
        "schema_version": "v3_data_card_v1", "card_id": "gse_endpoint_geometry_capacity_v1",
        "title": "Execution-anchored node and verified-edge development capacity",
        "status": "APPROVED_FOR_ONE_IMMUTABLE_GSE_ENDPOINT_GEOMETRY_CAPACITY_V1",
        "approval": approval,
        "purpose": "Test whether completed-route endpoint anchors and an observed branch witness close the three residual fit-edge safety failures without changing learned proposals, online radius or edge generation.",
        "operation": "read-only C01-C08 graph development-capacity replay; no training or threshold grid",
        "worlds": {
            "fit": "60 development worlds S01-S10 C01-C06",
            "internal_selection": "20 development worlds S01-S10 C07-C08; already used for mechanism confirmation and not an unseen result",
            "untouched_validation": "C09 remains unread and is the next one-time validation only if this capacity run passes",
            "strict_test": "C10 and M-TARE remain forbidden",
        },
        "sampling": {
            "causal_observations": 188126, "route_frames": 252430,
            "declared_directed_traversals": 16078, "framed_directed_traversals": 16076,
            "zero_sequence_traversals": ["S08_3d_loop_rich_C06:edge_0022:d0", "S08_3d_loop_rich_C06:edge_0022:d1"],
            "framed_physical_edges": 8038, "endpoint_geometries": 16076,
            "route_sample_spacing_m": 1.0,
            "independent_unit": "One completed directed traversal endpoint and one post-commit hypothesis/verified edge",
        },
        "trajectories": {
            "source": "Sealed C01-C08 frame axis_xyz_m and causal route records",
            "spatial_spacing_m": 1.0, "temporal_context": "5 causal LiDAR frames; endpoint geometry only from completed past traversal",
        },
        "split_logic": "C01-C06 and C07-C08 are development capacity only. C07-C08 confirmed the mechanism but selected no numeric threshold. C09 is untouched validation; C10 is strict test.",
        "teacher": {
            "runtime": "No TNG identity or objective center. Runtime consumes learned event center side, completed route endpoint anchors/directions, old factorized pair support and physical incidence.",
            "posthoc": "Objective TNG node/edge identities are read only after replay for one-to-one 4m scoring.",
            "anchor_bound": "Maximum two-endpoint error is 2 * fixed 1m route sampling spacing; no fitted tolerance.",
            "branch_witness": "At least three executed physical incidences, or for exactly two, positive outward-vector dot product; no angle threshold grid.",
        },
        "methods": {
            "main": "Frozen union association and factorized support, completed-route endpoint anchor consensus, parameter-free two-edge branch witness, exact endpoint duplicate union, and execution-verified edge remap.",
            "baseline": "Sealed factorized-hybrid endpoint inventory before execution-anchor qualification: fit node P 0.980510 and edge P 0.833333; C07-C08 node/edge P 1.0.",
            "fallback": "If either development partition misses node/edge precision 0.98 or recall 0.25, stop before C09 and inventory the remaining mechanism; do not tune an angle or distance threshold.",
        },
        "acceptance": {
            "node": "Both C01-C06 and C07-C08 objective node precision >=0.98, recall >=0.25 and false-loop fraction <=0.01.",
            "edge": "Both partitions verified-edge precision >=0.98 and recall >=0.25.",
            "sources": "Exactly 80 worlds, 252430 route frames, 16078 declared/16076 framed traversals and the two declared zero-sequence traversals; all C01-C08 axis files independently match the dataset seal.",
            "method": "No threshold grid, no changed 4m online radius, no predicted untraversed edge, and C07-C08 explicitly marked development selection.",
            "isolation": "Zero optimizer/model inference/update and zero C09/C10/M-TARE reads.",
            "evidence": "Per-hypothesis qualification, per-edge audit, metrics CSV, PNG/PDF/SVG, provenance, logs and SHA-256 seal.",
        },
        "leakage_audit": {
            "future_frames": 0, "c07_c08_used_for_method_confirmation": 1,
            "c07_c08_used_for_numeric_threshold_selection": 0,
            "c09_worlds": 0, "c10_worlds": 0, "mtare_worlds": 0,
        },
        "estimated_cost": {"compute": "CPU read-only", "wall_time_hours": 0.1, "host_ram_gb": 4, "gpu_memory_gb": 0, "disk_gb": 0.1},
        "retention": "Retain compact graph audits, all paper figure formats, provenance, logs, metrics and seal.",
    }
    write_json(CARD, card)
    inputs = [
        f"{teacher_root}/teacher_observations.jsonl", f"{teacher_root}/traversal_manifest.jsonl",
        f"{dataset_root}/RUN_STATE.json", f"{dataset_root}/artifacts/evidence_sha256.txt", f"{dataset_root}/artifacts/frame_manifest.jsonl",
        "results/gate3_semantics/gate3_20260826_gse_exit_token_association_corrective_v2_seed0/artifacts/pair_cache/pairs.npz",
        "results/gate3_semantics/gate3_20260828_gse_event_center_offset_training_v1r3_seed0/artifacts/teacher/event_center_teacher.npz",
        f"{replay_root}/RUN_STATE.json", f"{replay_root}/artifacts/evidence_sha256.txt",
        f"{replay_root}/artifacts/replay/action_ensemble.npz", f"{replay_root}/artifacts/replay/association_pairs.npz",
        f"{replay_root}/artifacts/projection/spatial_center_ensemble_all_rows.npz",
        f"{baseline}/RUN_STATE.json", f"{baseline}/artifacts/evidence_sha256.txt", f"{baseline}/artifacts/inventory/summary.json",
    ]
    tools = {
        "data_card": str(CARD.relative_to(PROJECT_ROOT)),
        "consolidation_module": "src/mtare_topo/topology/gse_post_commit_consolidation.py",
        "incidence_module": "src/mtare_topo/evaluation/gse_partial_incidence_reliability.py",
        "objective_scorer": "src/mtare_topo/evaluation/gse_objective_spatial_graph_score.py",
        "executor": "tools/v3/execute_gse_endpoint_geometry_capacity_v1.py",
        "runner": "tools/v3/run_gse_endpoint_geometry_capacity_v1.py",
        "freezer": "tools/v3/freeze_gse_endpoint_geometry_capacity_spec_v1.py",
        "unit_test": "tests/v3/unit/test_gse_post_commit_consolidation.py",
    }
    spec = {
        "schema_version": "v3_run_spec_v1", "gate": 3, "operation": "audit", "date": "20260828",
        "slug": "gse_endpoint_geometry_capacity_v1", "seed": 0,
        "question": "Can completed-route endpoint geometry close the residual C01-C08 objective graph-safety gap?",
        "method": card["methods"]["main"], "baseline": card["methods"]["baseline"],
        "estimated_cost": card["estimated_cost"], "config_path": str(CARD.relative_to(PROJECT_ROOT)),
        "data_card": str(CARD.relative_to(PROJECT_ROOT)), "user_authorization": approval,
        "estimated_resources": {"wall_time_minutes": 6, "ram_gb": 4, "disk_gb": .1, "gpu_count": 0},
        "acceptance_criteria": list(card["acceptance"].values()),
        "frozen_inputs": {value: _sha(PROJECT_ROOT / value) for value in inputs},
        "frozen_tools": {name: {"path": value, "sha256": _sha(PROJECT_ROOT / value)} for name, value in tools.items()},
        "expected_evidence": [
            "Exact C01-C08 completed-route endpoint geometry inventory.",
            "Objective node/edge safety and recall before/after execution-anchor qualification.",
            "Machine-readable decision whether Gate-3 may freeze the method and first read C09 validation.",
        ],
        "command": [
            "/usr/bin/timeout", "--signal=INT", "--kill-after=30s", "900s", PYTHON,
            "tools/v3/run_gse_endpoint_geometry_capacity_v1.py", "--spec", str(SPEC),
            "--run-dir", str(PROJECT_ROOT / "results/gate3_semantics" / RUN_ID),
        ],
    }
    write_json(SPEC, spec)
    print(SPEC.relative_to(PROJECT_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
