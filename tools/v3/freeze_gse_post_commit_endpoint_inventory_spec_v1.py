#!/usr/bin/env python3
"""Freeze the post-commit execution-endpoint inventory run."""

from __future__ import annotations

import hashlib

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import write_json


RUN_ID = "gate3_20260828_gse_post_commit_endpoint_inventory_v1_seed0"
SPEC = PROJECT_ROOT / "configs/v3/gate3/gse_post_commit_endpoint_inventory_v1.json"
CARD = PROJECT_ROOT / "configs/v3/gate3/data_cards/gse_post_commit_endpoint_inventory_v1.json"
PYTHON = "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python"


def _sha(path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    if SPEC.exists() or CARD.exists():
        raise RuntimeError("post-commit endpoint card/spec exists; overwrite is forbidden")
    teacher_root = "results/gate2_representation/gate2_20260827_gse_corrected_causal_teacher_manifest_v1r_seed0/artifacts"
    replay_root = "results/gate3_semantics/gate3_20260828_gse_spatial_trace_commit_requalification_v1r2_seed0"
    approval = {
        "status": "APPROVED", "approved_by": "user-standing-authorization",
        "approved_at": "2026-08-28T18:05:00+08:00", "authorized_gates": [3],
        "authorized_operations": ["audit"],
        "scope": "One immutable C01-C08 read-only post-commit endpoint inventory; zero training/inference/C09/C10/M-TARE.",
        "confirmation_reference": "User instructed automatic best-choice execution without routine approval prompts.",
    }
    card = {
        "schema_version": "v3_data_card_v1", "card_id": "gse_post_commit_endpoint_inventory_v1",
        "title": "Execution-endpoint duplicate consolidation and residual edge-safety inventory",
        "status": "APPROVED_FOR_ONE_IMMUTABLE_GSE_POST_COMMIT_ENDPOINT_INVENTORY_V1",
        "approval": approval,
        "purpose": "Determine whether completed-traversal endpoint identity can safely consolidate duplicate committed nodes, and isolate any remaining graph-safety defect.",
        "operation": "read-only endpoint-identity Teacher inventory and no-training graph capacity replay",
        "worlds": {"fit": "60 worlds S01-S10 C01-C06", "selection": "20 worlds S01-S10 C07-C08", "forbidden": "C09, C10, strict test and M-TARE"},
        "sampling": {
            "raw_observations": 188126, "directed_traversals": 16078,
            "fit_observations": 142184, "selection_observations": 45942,
            "fit_selection_proposals": "3037/1022", "fit_selection_committed_hypotheses": "718/234",
            "independent_unit": "One committed hypothesis or one orientation-invariant completed physical-edge endpoint token",
        },
        "trajectories": {"source": "All sealed directed traversals in C01-C08", "spacing_m": 1.0, "temporal_context": "5 causal frames, no future"},
        "split_logic": "C01-C06 establishes token invariance and safety capacity. C07-C08 is a one-time read-only replication; it sets no threshold, feature or checkpoint.",
        "teacher": {
            "source": "TNG objective identity is read only after runtime replay; runtime endpoint token uses traversal direction, executed route arc and completed edge length.",
            "positive": "Two committed hypotheses with the same exact traversed-edge endpoint token and the same objective structure identity.",
            "negative": "Same endpoint-token candidate with different objective identities; absence is reported and forbids verifier training.",
            "runtime_isolation": "No objective identity, objective center or future traversal is consumed by consolidation.",
        },
        "methods": {
            "main": "Keep the 4m online union replay, require at least two physical edges for junctions, and union committed same-world/same-event hypotheses only on an exact learned-event endpoint token. A fit-only factorized hybrid additionally requires old-verifier support for two-edge junctions.",
            "baseline": "Frozen support-2 incidence graph without endpoint consolidation and the support-3 high-precision graph.",
            "fallback": "If endpoint tokens are ambiguous, candidates contain no hard negatives, or fit edge safety remains below 0.98, do not train a duplicate verifier and isolate the remaining edge mechanism.",
        },
        "acceptance": {
            "population": "Exact 188126 observations, 16078 traversals, 3037/1022 proposals and 718/234 committed hypotheses.",
            "invariant": "Exactly 2933/1013 fit/selection endpoint Teacher tokens and zero identity ambiguity.",
            "candidate_inventory": "Support-2 consolidation yields exactly 9/5 fit/selection positive components, with every negative/unknown candidate reported.",
            "decision": "Zero hard negatives must set training_authorized=false; residual graph failure must remain visible rather than being adapted away.",
            "isolation": "Zero optimizer/model inference/update and zero C09/C10/M-TARE reads.",
            "evidence": "Summary, candidate/edge JSONL, capacity CSV, PNG/PDF/SVG, provenance, logs and SHA-256 seal.",
        },
        "leakage_audit": {"future_frames": 0, "selection_used_for_training": 0, "c09_worlds": 0, "c10_worlds": 0, "mtare_worlds": 0},
        "estimated_cost": {"compute": "CPU read-only", "wall_time_hours": 0.25, "host_ram_gb": 4, "gpu_memory_gb": 0, "disk_gb": 0.1},
        "retention": "Retain compact inventory, all paper figure formats, provenance, metrics, logs and seal.",
    }
    write_json(CARD, card)
    inputs = [
        f"{teacher_root}/teacher_observations.jsonl", f"{teacher_root}/traversal_manifest.jsonl",
        "results/gate3_semantics/gate3_20260826_gse_exit_token_association_corrective_v2_seed0/artifacts/pair_cache/pairs.npz",
        "results/gate3_semantics/gate3_20260828_gse_event_center_offset_training_v1r3_seed0/artifacts/teacher/event_center_teacher.npz",
        f"{replay_root}/RUN_STATE.json", f"{replay_root}/artifacts/evidence_sha256.txt",
        f"{replay_root}/artifacts/replay/action_ensemble.npz", f"{replay_root}/artifacts/replay/association_pairs.npz",
        f"{replay_root}/artifacts/projection/spatial_center_ensemble_all_rows.npz",
        "results/gate3_semantics/gate3_20260828_gse_partial_incidence_teacher_inventory_v1_seed0/RUN_STATE.json",
        "results/gate3_semantics/gate3_20260828_gse_partial_incidence_teacher_inventory_v1_seed0/artifacts/evidence_sha256.txt",
    ]
    tools = {
        "data_card": str(CARD.relative_to(PROJECT_ROOT)),
        "consolidation_module": "src/mtare_topo/topology/gse_post_commit_consolidation.py",
        "incidence_module": "src/mtare_topo/evaluation/gse_partial_incidence_reliability.py",
        "objective_scorer": "src/mtare_topo/evaluation/gse_objective_spatial_graph_score.py",
        "executor": "tools/v3/execute_gse_post_commit_endpoint_inventory_v1.py",
        "runner": "tools/v3/run_gse_post_commit_endpoint_inventory_v1.py",
        "freezer": "tools/v3/freeze_gse_post_commit_endpoint_inventory_spec_v1.py",
        "unit_test": "tests/v3/unit/test_gse_post_commit_consolidation.py",
    }
    spec = {
        "schema_version": "v3_run_spec_v1", "gate": 3, "operation": "audit", "date": "20260828",
        "slug": "gse_post_commit_endpoint_inventory_v1", "seed": 0,
        "question": "Can accumulated execution endpoint identity consolidate duplicate committed nodes safely enough for objective graph qualification?",
        "method": card["methods"]["main"], "baseline": card["methods"]["baseline"],
        "estimated_cost": card["estimated_cost"], "config_path": str(CARD.relative_to(PROJECT_ROOT)),
        "data_card": str(CARD.relative_to(PROJECT_ROOT)), "user_authorization": approval,
        "estimated_resources": {"wall_time_minutes": 15, "ram_gb": 4, "disk_gb": .1, "gpu_count": 0},
        "acceptance_criteria": list(card["acceptance"].values()),
        "frozen_inputs": {value: _sha(PROJECT_ROOT / value) for value in inputs},
        "frozen_tools": {name: {"path": value, "sha256": _sha(PROJECT_ROOT / value)} for name, value in tools.items()},
        "expected_evidence": [
            "Exact route-endpoint invariance and duplicate candidate population.",
            "No-training endpoint and factorized-hybrid objective graph capacity.",
            "Machine-readable prohibition on invalid duplicate-verifier training and the residual edge-safety blocker.",
        ],
        "command": [
            "/usr/bin/timeout", "--signal=INT", "--kill-after=30s", "900s", PYTHON,
            "tools/v3/run_gse_post_commit_endpoint_inventory_v1.py", "--spec", str(SPEC),
            "--run-dir", str(PROJECT_ROOT / "results/gate3_semantics" / RUN_ID),
        ],
    }
    write_json(SPEC, spec)
    print(SPEC.relative_to(PROJECT_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
