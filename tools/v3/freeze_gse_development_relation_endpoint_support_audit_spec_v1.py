#!/usr/bin/env python3
"""Freeze Data Card/spec for the C01-C08 relation-endpoint support audit."""

from __future__ import annotations

import hashlib

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import write_json


RUN_ID = "gate3_20260828_gse_development_relation_endpoint_support_audit_v1_seed0"
CARD = PROJECT_ROOT / "configs/v3/gate3/data_cards/gse_development_relation_endpoint_support_audit_v1.json"
SPEC = PROJECT_ROOT / "configs/v3/gate3/gse_development_relation_endpoint_support_audit_v1.json"
PYTHON = "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python"


def _sha(path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    if CARD.exists() or SPEC.exists(): raise RuntimeError("development endpoint-support card/spec exists; overwrite forbidden")
    teacher = "results/gate2_representation/gate2_20260827_gse_corrected_causal_teacher_manifest_v1r_seed0"
    pair = "results/gate3_semantics/gate3_20260826_gse_exit_token_association_corrective_v2_seed0"
    replay = "results/gate3_semantics/gate3_20260828_gse_spatial_trace_commit_requalification_v1r2_seed0"
    capacity = "results/gate3_semantics/gate3_20260828_gse_endpoint_geometry_capacity_v1_seed0"
    approval = {"status": "APPROVED", "approved_by": "user-standing-authorization", "approved_at": "2026-08-28T00:00:00+08:00", "authorized_gates": [3], "authorized_operations": ["audit"], "scope": "One immutable C01-C08 read-only relation-endpoint support audit; no training, model inference, C09, C10 or M-TARE.", "confirmation_reference": "User instructed automatic best-choice execution without routine approval prompts."}
    card = {
        "schema_version": "v3_data_card_v1", "card_id": "gse_development_relation_endpoint_support_audit_v1",
        "title": "GSE development relation-endpoint supervision and survival audit",
        "status": "APPROVED_FOR_ONE_IMMUTABLE_GSE_DEVELOPMENT_RELATION_ENDPOINT_SUPPORT_AUDIT_V1",
        "purpose": "Test on development worlds whether low per-identity supervision support systematically lowers relation-endpoint proposal and final-node survival.", "approval": approval,
        "source": {"teacher_run": teacher, "pair_cache_run": pair, "frozen_replay_run": replay, "endpoint_capacity_run": capacity, "license_or_allowed_use": "Local research use of project-generated procedural worlds and locally trained outputs."},
        "worlds": {"fit": "All 60 C01-C06 worlds", "selection": "All 20 C07-C08 worlds", "forbidden": "C09, C10 and M-TARE"},
        "trajectories": {"directed_traversals": 16078, "rule": "All sealed C01-C08 directed traversals; two known zero-sequence traversals remain explicit and are not fabricated."},
        "sampling": {"raw_frame_count": 252430, "effective_sample_count": 188126, "fit_observations": 142184, "selection_observations": 45942, "independent_units": "Fit: 36 relations/72 distinct endpoints. Selection: 13 relations/26 distinct endpoints.", "spatial_interval_m": 1.0, "temporal_context": "Existing five-frame causal observations.", "support_bins": ["1-3", "4-10", "11-30", "31+"]},
        "split": {"fit": "C01-C06; may justify a corrective.", "selection": "C07-C08; checks direction only and cannot set a numeric threshold.", "validation": "C09 is not read.", "strict_test": "C10 remains unread.", "leakage_audit": "Only sealed development Teacher, predictions and replay are read; zero model/threshold updates."},
        "teacher": {"source": "Corrected sealed C01-C08 causal Teacher identities/events.", "objective_relation": "Consecutive observed junction/terminal identities along a directed traversal.", "support": "Number of causal Teacher rows for each distinct relation endpoint identity."},
        "methods": {"main": "Reconstruct the frozen proposal/association/commit replay and join its sealed endpoint qualification/final objective mapping. Report mutually exclusive endpoint/relation stages by support bin, event and family.", "baseline": "The existing row-balanced action model and frozen endpoint-geometry capacity graph.", "fallback": "If low support does not show a pre-registered proposal-recall gap on fit without reversal on selection, reject identity-balanced training and inspect the next causal stage."},
        "acceptance": {"population": "Exactly 188126 observations; fit 36 relations/72 endpoints/3037 proposals/718 commits; selection 13/26/1022/234.", "attribution": "All 98 endpoints and 49 relations receive one stage and all support bins are reported without deletion.", "decision": "Identity-balanced corrective is justified only if fit has at least 8 endpoints with 1-3 rows, high-support(>=11) minus low-support proposal recall is >=0.10, and selection shows no reversal.", "isolation": "Zero training, inference, threshold selection, C09, C10 and M-TARE."},
        "estimated_cost": {"compute": "One CPU replay and aggregation.", "wall_time_hours": 0.1, "host_ram_gb": 4, "gpu_memory_gb": 0, "disk_gb": 0.1, "gpu": "none"},
        "retention": "Keep endpoint/relation tables, support statistics, PNG/PDF/SVG/source, logs, environment, RUN_STATE and seal.",
        "failure_policy": "Any source, alignment, population, replay or attribution mismatch fails closed; do not adapt from partial results.",
    }
    write_json(CARD, card)
    inputs = [
        f"{teacher}/RUN_STATE.json", f"{teacher}/artifacts/evidence_sha256.txt", f"{teacher}/artifacts/teacher_observations.jsonl",
        f"{pair}/RUN_STATE.json", f"{pair}/artifacts/evidence_sha256.txt", f"{pair}/artifacts/pair_cache/pairs.npz",
        f"{replay}/RUN_STATE.json", f"{replay}/artifacts/evidence_sha256.txt", f"{replay}/artifacts/replay/action_ensemble.npz", f"{replay}/artifacts/replay/association_pairs.npz", f"{replay}/artifacts/projection/spatial_center_ensemble_all_rows.npz",
        f"{capacity}/RUN_STATE.json", f"{capacity}/artifacts/evidence_sha256.txt", f"{capacity}/artifacts/capacity/summary.json", f"{capacity}/artifacts/capacity/fit_hypothesis_qualification.jsonl", f"{capacity}/artifacts/capacity/selection_hypothesis_qualification.jsonl", f"{capacity}/artifacts/capacity/fit_edge_audit.jsonl", f"{capacity}/artifacts/capacity/selection_edge_audit.jsonl",
    ]
    tools = {"data_card": str(CARD.relative_to(PROJECT_ROOT)), "executor": "tools/v3/execute_gse_development_relation_endpoint_support_audit_v1.py", "runner": "tools/v3/run_gse_development_relation_endpoint_support_audit_v1.py", "freezer": "tools/v3/freeze_gse_development_relation_endpoint_support_audit_spec_v1.py", "trigger_contract": "src/mtare_topo/evaluation/gse_causal_episode_metrics.py", "partial_incidence": "src/mtare_topo/evaluation/gse_partial_incidence_reliability.py", "relation_contract": "src/mtare_topo/evaluation/gse_trace_commit_failure_funnel.py", "trace_replay": "src/mtare_topo/topology/gse_trace_commit_replay.py", "governance": "src/mtare_topo/governance.py", "preflight": "tools/v3/preflight.py", "create_run": "tools/v3/create_run.py"}
    spec = {"schema_version": "v3_run_spec_v1", "gate": 3, "execution_phase": 3, "date": "20260828", "slug": "gse_development_relation_endpoint_support_audit_v1", "seed": 0, "operation": "audit", "question": "Does per-identity supervision scarcity explain relation-endpoint proposal/commit loss on C01-C08?", "method": card["methods"]["main"], "baseline": card["methods"]["baseline"], "fallback": card["methods"]["fallback"], "data_card": str(CARD.relative_to(PROJECT_ROOT)), "config_path": str(CARD.relative_to(PROJECT_ROOT)), "user_authorization": approval, "acceptance_criteria": list(card["acceptance"].values()), "expected_counts": {"development_worlds": 80, "causal_observations": 188126, "objective_relations": 49, "relation_endpoints": 98, "model_inference_frames": 0, "optimizer_steps": 0, "model_updates": 0, "threshold_selection_steps": 0, "c09_worlds_read": 0, "c10_worlds_read": 0, "mtare_worlds_read": 0}, "expected_evidence": ["98-row endpoint table, 49-row relation table, support-bin statistics, PNG/PDF/SVG/source, environment, logs, RUN_STATE and SHA-256 seal."], "estimated_cost": card["estimated_cost"], "frozen_inputs": {path: _sha(PROJECT_ROOT / path) for path in inputs}, "frozen_tools": {name: {"path": path, "sha256": _sha(PROJECT_ROOT / path)} for name, path in tools.items()}, "working_directory": str(PROJECT_ROOT), "command": ["/usr/bin/timeout", "--signal=INT", "--kill-after=30s", "660s", PYTHON, "tools/v3/run_gse_development_relation_endpoint_support_audit_v1.py", "--spec", str(SPEC), "--run-dir", str(PROJECT_ROOT / "results/gate3_semantics" / RUN_ID)]}
    write_json(SPEC, spec); print(CARD.relative_to(PROJECT_ROOT)); print(SPEC.relative_to(PROJECT_ROOT)); return 0


if __name__ == "__main__": raise SystemExit(main())
