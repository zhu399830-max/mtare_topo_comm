#!/usr/bin/env python3
"""Freeze the single read-only trace-commit failure-funnel audit."""

from __future__ import annotations

import hashlib

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import write_json


RUN_ID = "gate3_20260828_gse_trace_commit_failure_funnel_audit_v1_seed0"
SPEC = PROJECT_ROOT / "configs/v3/gate3/gse_trace_commit_failure_funnel_audit_v1.json"
CARD = PROJECT_ROOT / "configs/v3/gate3/data_cards/gse_trace_commit_failure_funnel_audit_v1.json"
PYTHON = "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python"


def _sha(path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    if SPEC.exists() or CARD.exists():
        raise RuntimeError("failure-funnel audit card/spec exists; overwrite is forbidden")
    current = "results/gate3_semantics/gate3_20260828_gse_spatial_trace_commit_requalification_v1r2_seed0"
    predecessor = "results/gate3_semantics/gate3_20260828_gse_projected_trace_commit_capacity_v1_seed0"
    teacher = "results/gate2_representation/gate2_20260827_gse_corrected_causal_teacher_manifest_v1r_seed0"
    source = "results/gate3_semantics/gate3_20260826_gse_exit_token_association_corrective_v2_seed0"
    approval = {
        "status": "APPROVED", "approved_by": "user-standing-authorization",
        "approved_at": "2026-08-28T13:00:00+08:00", "authorized_gates": [3],
        "authorized_operations": ["audit"],
        "scope": "One immutable read-only C07-C08 trace-commit failure-funnel audit; no model or graph changes and zero C09/C10/M-TARE.",
        "confirmation_reference": "User instructed automatic best-choice execution without routine approval prompts.",
    }
    card = {
        "schema_version": "v3_data_card_v1",
        "card_id": "gse_trace_commit_failure_funnel_audit_v1",
        "title": "GSE trace-commit node/edge causal failure funnel",
        "status": "APPROVED_FOR_ONE_IMMUTABLE_GSE_TRACE_COMMIT_FAILURE_FUNNEL_AUDIT_V1",
        "operation": "read-only audit and paper figure generation",
        "purpose": "Attribute every missed C07-C08 true node and trace relation to proposal, association radius, learned verifier, two-trace commit, or edge assembly without changing the method.",
        "worlds": {
            "audit": "20 disjoint development-validation worlds S01-S10 C07-C08",
            "forbidden": "C09, C10, strict test and all M-TARE worlds",
        },
        "trajectories": {
            "source": "All sealed C07-C08 directed traversals already replayed by V1R2",
            "directed_traversals_in_parent_population": 3972,
            "spatial_spacing_m": 1.0,
            "temporal_context": "Current plus four strictly past observations; no future frame",
        },
        "sampling": {
            "independent_unit": "Objective TNG node identity and observed semantic endpoint relation across physical directed traversals",
            "raw_observations": 45942,
            "proposal_rows": 1022,
            "effective_node_identities": 274,
            "effective_trace_relations": 13,
            "committed_nodes": 207,
            "verified_edges": 2,
        },
        "split_logic": "Audit only the already-sealed C07-C08 replay. No split, threshold, checkpoint, model, radius, vote or commit selection occurs.",
        "teacher": {
            "source": "Corrected objective TNG identity/event and observed traversal endpoint relations",
            "usage": "Post-replay attribution and scoring only; never candidate generation or association",
        },
        "methods": {
            "main": "For each true identity, follow proposal coverage, independent traversal coverage, <=4 m center overlap, frozen verifier acceptance and final commit. For each true relation, follow endpoint proposal, endpoint commit, common-trace support and edge assembly.",
            "baseline": "Sealed scalar-projected trace-commit predecessor metrics versus sealed spatial-center V1R2 metrics.",
            "fallback": "None. The audit selects no repair; it only identifies the causal stage for the next preregistered decision.",
        },
        "leakage_audit": {
            "model_inputs": 0, "future_frames": 0, "objective_identity_in_runtime": 0,
            "c09_worlds": 0, "c10_worlds": 0, "mtare_worlds": 0,
        },
        "acceptance": {
            "population": "Exactly 45,942 observations, 1,022 proposal rows, 274 true node identities, 13 true relations, 207 committed nodes and 2 verified edges.",
            "scorer_reproduction": "Committed/unique/mixed node and correct-edge counts exactly reproduce the sealed V1R2 scorer.",
            "attribution": "Every true node identity and true relation receives exactly one causal stage.",
            "evidence": "JSONL/CSV summaries plus PNG/PDF/SVG paper figure, provenance, logs and seal.",
        },
        "estimated_cost": {"compute": "CPU read-only", "wall_time_hours": 0.25, "host_ram_gb": 4, "gpu_memory_gb": 0, "disk_gb": 0.1},
        "retention": "Retain all compact audit tables, summary, paper figure formats, provenance, logs and seal; duplicate no sensor payload.",
        "approval": approval,
    }
    write_json(CARD, card)
    inputs = [
        f"{current}/RUN_STATE.json", f"{current}/metrics/summary.json", f"{current}/artifacts/evidence_sha256.txt",
        f"{current}/artifacts/projection/spatial_center_ensemble_all_rows.npz",
        f"{current}/artifacts/replay/summary.json", f"{current}/artifacts/replay/association_pairs.npz",
        f"{current}/artifacts/replay/decision_trace.jsonl", f"{current}/artifacts/replay/verified_nodes.jsonl", f"{current}/artifacts/replay/verified_edges.jsonl",
        f"{predecessor}/RUN_STATE.json", f"{predecessor}/metrics/summary.json", f"{predecessor}/artifacts/evidence_sha256.txt", f"{predecessor}/artifacts/replay/summary.json",
        f"{teacher}/RUN_STATE.json", f"{teacher}/artifacts/evidence_sha256.txt", f"{teacher}/artifacts/teacher_observations.jsonl",
        f"{source}/RUN_STATE.json", f"{source}/artifacts/evidence_sha256.txt", f"{source}/artifacts/pair_cache/pairs.npz",
    ]
    tools = {
        "data_card": str(CARD.relative_to(PROJECT_ROOT)),
        "funnel_module": "src/mtare_topo/evaluation/gse_trace_commit_failure_funnel.py",
        "executor": "tools/v3/execute_gse_trace_commit_failure_funnel_audit_v1.py",
        "runner": "tools/v3/run_gse_trace_commit_failure_funnel_audit_v1.py",
        "freezer": "tools/v3/freeze_gse_trace_commit_failure_funnel_audit_spec_v1.py",
        "preflight": "tools/v3/preflight.py", "create_run": "tools/v3/create_run.py",
    }
    run_dir = PROJECT_ROOT / "results/gate3_semantics" / RUN_ID
    spec = {
        "schema_version": "v3_run_spec_v1", "gate": 3, "execution_phase": 3,
        "date": "20260828", "slug": "gse_trace_commit_failure_funnel_audit_v1", "seed": 0,
        "operation": "audit",
        "question": "Which causal stage now limits node safety and execution-verified edge recall after the qualified spatial center?",
        "method": card["methods"]["main"], "baseline": card["methods"]["baseline"], "fallback": card["methods"]["fallback"],
        "data_card": str(CARD.relative_to(PROJECT_ROOT)), "config_path": str(CARD.relative_to(PROJECT_ROOT)),
        "user_authorization": approval,
        "acceptance_criteria": list(card["acceptance"].values()),
        "expected_counts": {
            "selection_observations": 45942, "proposal_rows": 1022,
            "true_node_identities": 274, "true_trace_relations": 13,
            "committed_nodes": 207, "verified_edges": 2,
            "optimizer_steps": 0, "model_updates": 0,
            "c09_worlds_read": 0, "c10_worlds_read": 0, "mtare_worlds_read": 0,
        },
        "estimated_cost": card["estimated_cost"],
        "expected_evidence": [
            "274-row node identity attribution and 13-row relation attribution.",
            "Metric comparison, node/edge funnel CSV, summary and paper-grade PNG/PDF/SVG figure.",
            "Source integrity, environment, raw log, RUN_STATE and complete SHA-256 seal.",
        ],
        "frozen_inputs": {relative: _sha(PROJECT_ROOT / relative) for relative in inputs},
        "frozen_tools": {name: {"path": relative, "sha256": _sha(PROJECT_ROOT / relative)} for name, relative in tools.items()},
        "command": [
            "/usr/bin/timeout", "--signal=INT", "--kill-after=30s", "900s", PYTHON,
            "tools/v3/run_gse_trace_commit_failure_funnel_audit_v1.py",
            "--spec", str(SPEC), "--run-dir", str(run_dir),
        ],
        "working_directory": str(PROJECT_ROOT),
    }
    write_json(SPEC, spec)
    print(SPEC.relative_to(PROJECT_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
