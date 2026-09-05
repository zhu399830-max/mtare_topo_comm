#!/usr/bin/env python3
"""Freeze Data Card and run spec for the C09 relation-endpoint funnel."""

from __future__ import annotations

import hashlib

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, write_json


RUN_ID = "gate3_20260828_gse_c09_relation_endpoint_funnel_v1_seed0"
CARD = PROJECT_ROOT / "configs/v3/gate3/data_cards/gse_c09_relation_endpoint_funnel_v1.json"
SPEC = PROJECT_ROOT / "configs/v3/gate3/gse_c09_relation_endpoint_funnel_v1.json"
PYTHON = "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python"


def _sha(path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    if CARD.exists() or SPEC.exists():
        raise RuntimeError("C09 endpoint-funnel card/spec exists; overwrite is forbidden")
    validation = "results/gate3_semantics/gate3_20260828_gse_endpoint_geometry_c09_validation_v1_seed0"
    teacher = "results/gate2_representation/gate2_20260824_gse_teacher_manifest_v1_seed0"
    parents = "results/gate0_baseline/gate0_20260811_cano_100_topology_parent_recipe_reclassification_v2r_seed0"
    approval = {
        "status": "APPROVED", "approved_by": "user-standing-authorization",
        "approved_at": "2026-08-28T00:00:00+08:00", "authorized_gates": [3],
        "authorized_operations": ["audit"],
        "scope": "One immutable read-only causal attribution of the eight failed C09 relations; no inference, training, threshold adaptation, C10 or M-TARE.",
        "confirmation_reference": "User instructed automatic best-choice execution without routine approval prompts.",
    }
    card = {
        "schema_version": "v3_data_card_v1", "card_id": "gse_c09_relation_endpoint_funnel_v1",
        "title": "C09 frozen graph relation-endpoint causal failure funnel",
        "status": "APPROVED_FOR_ONE_IMMUTABLE_GSE_C09_RELATION_ENDPOINT_FUNNEL_V1",
        "purpose": "Determine whether each missing C09 relation is lost at proposal, event classification, association/commit, endpoint qualification, final spatial matching or raw edge assembly.",
        "approval": approval,
        "source": {
            "frozen_validation_run": validation, "teacher_run": teacher,
            "objective_parent_manifest": parents,
            "license_or_allowed_use": "Local research use of project-generated procedural worlds and locally trained outputs.",
        },
        "worlds": [f"S{index:02d}_{name}_C09" for index, name in enumerate((
            "flat_tree_small", "3d_tree_small", "flat_unicyclic_small", "3d_unicyclic_small",
            "flat_branch_medium", "3d_branch_medium", "flat_loop_rich", "3d_loop_rich",
            "flat_complex", "3d_complex",
        ), start=1)],
        "trajectories": {"directed_traversals": 2054, "physical_edges": 1027, "rule": "Read the exact sealed C09 replay; do not generate or remove trajectories."},
        "sampling": {
            "raw_frame_count": 32678, "effective_sample_count": 24462,
            "independent_units": "Eight observed objective relations and their sixteen distinct endpoint identities across ten C09 worlds.",
            "spatial_interval_m": 1.0, "temporal_context": "Existing five-frame causal observations; no new sensor read or inference.",
        },
        "split": {
            "fit": "None.", "validation": "Exact sealed C09 run, used only to attribute its already-declared failure.",
            "strict_test": "C10 remains unread.",
            "leakage_audit": "Teacher is posthoc and cannot alter the frozen graph, thresholds, checkpoints or subsequent C09 result.",
        },
        "teacher": {
            "source": "Sealed C09 event/identity rows and objective TNG centers.",
            "use": "Posthoc attribution of proposal evidence, hypothesis contents and objective relations only.",
        },
        "methods": {
            "main": "Reconstruct the exact proposal-to-hypothesis mapping from the sealed decision trace, reproduce six raw execution edges, then assign one mutually exclusive failure stage to every relation endpoint and relation.",
            "baseline": "Frozen final recovery: five of sixteen relation endpoints and one of eight relations.",
            "fallback": "If any intermediate population or mapping is not uniquely reproducible, fail the audit and report evidence insufficiency; do not infer or adapt.",
        },
        "acceptance": {
            "population": "Exactly 24462 observations, 1646 triggers, 442 hypotheses, 114 committed hypotheses, 74 qualified/final nodes, 6 raw edges, 1 final edge, 8 relations and 16 endpoints.",
            "attribution": "All sixteen endpoints and eight relations receive exactly one causal stage; final recovery reproduces 5 endpoints and 1 relation.",
            "isolation": "Zero optimizer, model update, inference, threshold selection, C10 and M-TARE reads.",
        },
        "estimated_cost": {"compute": "One CPU read-only trace reconstruction and plot.", "wall_time_hours": 0.05, "host_ram_gb": 2, "gpu_memory_gb": 0, "disk_gb": 0.1, "gpu": "none"},
        "retention": "Keep endpoint/relation/raw-edge tables, summary, PNG/PDF/SVG/source, logs, environment, RUN_STATE and seal as failure-analysis and paper evidence.",
        "failure_policy": "Any count, hash, alignment, replay or unique-attribution mismatch fails closed; do not modify the formal C09 run.",
    }
    write_json(CARD, card)
    inputs = [
        f"{validation}/RUN_STATE.json", f"{validation}/metrics/summary.json", f"{validation}/artifacts/evidence_sha256.txt",
        f"{validation}/artifacts/teacher_free_graph/inference_manifest.json",
        f"{validation}/artifacts/teacher_free_graph/action_ensemble.npz",
        f"{validation}/artifacts/teacher_free_graph/decision_trace.jsonl",
        f"{validation}/artifacts/teacher_free_graph/hypothesis_qualification.jsonl",
        f"{validation}/artifacts/teacher_free_graph/verified_nodes.jsonl",
        f"{validation}/artifacts/teacher_free_graph/verified_edges.jsonl",
        f"{validation}/artifacts/evaluation/summary.json",
        f"{teacher}/RUN_STATE.json", f"{teacher}/artifacts/evidence_sha256.txt",
        f"{teacher}/artifacts/teacher_observations.jsonl",
        f"{parents}/RUN_STATE.json", f"{parents}/artifacts/evidence_sha256.txt",
        f"{parents}/artifacts/accepted_parent_manifest.json",
    ]
    parent_manifest = load_json(PROJECT_ROOT / f"{parents}/artifacts/accepted_parent_manifest.json")
    inputs.extend(str(row["source_graph"]) for row in parent_manifest["parents"] if str(row.get("parent_id", "")).endswith("_C09"))
    tools = {
        "data_card": str(CARD.relative_to(PROJECT_ROOT)),
        "executor": "tools/v3/execute_gse_c09_relation_endpoint_funnel_v1.py",
        "runner": "tools/v3/run_gse_c09_relation_endpoint_funnel_v1.py",
        "freezer": "tools/v3/freeze_gse_c09_relation_endpoint_funnel_spec_v1.py",
        "c09_evaluator": "tools/v3/evaluate_gse_endpoint_geometry_c09_v1.py",
        "trigger_contract": "src/mtare_topo/evaluation/gse_causal_episode_metrics.py",
        "relation_contract": "src/mtare_topo/evaluation/gse_trace_commit_failure_funnel.py",
        "objective_score": "src/mtare_topo/evaluation/gse_objective_spatial_graph_score.py",
        "governance": "src/mtare_topo/governance.py", "preflight": "tools/v3/preflight.py",
        "create_run": "tools/v3/create_run.py",
    }
    spec = {
        "schema_version": "v3_run_spec_v1", "gate": 3, "execution_phase": 3,
        "date": "20260828", "slug": "gse_c09_relation_endpoint_funnel_v1", "seed": 0,
        "operation": "audit",
        "question": "At which causal stage are the seven missing frozen C09 relations lost?",
        "method": card["methods"]["main"], "baseline": card["methods"]["baseline"],
        "fallback": card["methods"]["fallback"], "data_card": str(CARD.relative_to(PROJECT_ROOT)),
        "config_path": str(CARD.relative_to(PROJECT_ROOT)), "user_authorization": approval,
        "acceptance_criteria": list(card["acceptance"].values()),
        "expected_counts": {
            "validation_worlds": 10, "causal_observations": 24462, "objective_relations": 8,
            "relation_endpoints": 16, "raw_edges": 6, "final_edges": 1,
            "model_inference_frames": 0, "optimizer_steps": 0, "model_updates": 0,
            "threshold_selection_steps": 0, "c10_worlds_read": 0, "mtare_worlds_read": 0,
        },
        "expected_evidence": ["Per-endpoint, per-relation and raw-edge causal tables; survival/stage CSV; PNG/PDF/SVG/source; environment, logs, RUN_STATE and SHA-256 seal."],
        "estimated_cost": card["estimated_cost"],
        "frozen_inputs": {path: _sha(PROJECT_ROOT / path) for path in inputs},
        "frozen_tools": {name: {"path": path, "sha256": _sha(PROJECT_ROOT / path)} for name, path in tools.items()},
        "working_directory": str(PROJECT_ROOT),
        "command": [
            "/usr/bin/timeout", "--signal=INT", "--kill-after=30s", "360s", PYTHON,
            "tools/v3/run_gse_c09_relation_endpoint_funnel_v1.py", "--spec", str(SPEC),
            "--run-dir", str(PROJECT_ROOT / "results/gate3_semantics" / RUN_ID),
        ],
    }
    write_json(SPEC, spec)
    print(CARD.relative_to(PROJECT_ROOT)); print(SPEC.relative_to(PROJECT_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
