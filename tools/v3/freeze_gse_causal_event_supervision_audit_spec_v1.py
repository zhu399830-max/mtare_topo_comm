#!/usr/bin/env python3
"""Freeze the causal event supervision audit card and one-shot spec."""

from __future__ import annotations

import hashlib
from pathlib import Path

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, write_json


RUN_ID = "gate3_20260827_gse_causal_event_supervision_audit_v1_seed0"
CARD_PATH = PROJECT_ROOT / "configs/v3/gate3/data_cards/gse_causal_event_supervision_audit_v1.json"
SPEC_PATH = PROJECT_ROOT / "configs/v3/gate3/gse_causal_event_supervision_audit_v1.json"
SOURCE_CARD = PROJECT_ROOT / "configs/v3/gate3/data_cards/gse_corrected_causal_event_training_v1.json"
TEACHER = "results/gate2_representation/gate2_20260827_gse_corrected_causal_teacher_manifest_v1r_seed0"
PROOF = "results/gate3_semantics/gate3_20260826_gse_causal_change_point_proof_v1_seed0"
RISK = "results/gate3_semantics/gate3_20260827_gse_causal_geometry_risk_conflict_audit_v1_seed0"
CAPACITY = "results/gate3_semantics/gate3_20260827_gse_geometry_conditioned_identity_risk_capacity_v1_seed0"
PYTHON = "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    source_card = load_json(SOURCE_CARD)
    capacity = load_json(PROJECT_ROOT / CAPACITY / "metrics/capacity_summary.json")
    if capacity.get("overall_status") != "FAIL_GSE_GEOMETRY_CONDITIONED_IDENTITY_RISK_CAPACITY_V1":
        raise RuntimeError("supervision audit requires the sealed capacity failure")
    card = {
        "schema_version": "v3_data_card_v1",
        "card_id": "gse_causal_event_supervision_audit_v1",
        "status": "APPROVED_FOR_ONE_IMMUTABLE_GSE_CAUSAL_EVENT_SUPERVISION_AUDIT_V1",
        "purpose": "Determine whether the persistent change-point target is naturally an episode-level causal detection problem rather than an independent frame-classification problem, and whether existing scans support the required history.",
        "approval": {
            "status": "APPROVED", "approved_by": "user", "approved_at": "2026-08-27T23:59:40+08:00",
            "authorized_gates": [3], "authorized_operations": ["audit"],
            "scope": "One immutable read-only C01-C08 frame-versus-episode supervision audit; zero model update/inference and zero C09/C10/M-TARE access.",
            "confirmation_reference": "User granted standing authorization to choose the optimal in-scope route and continue without routine approval prompts.",
        },
        "worlds": source_card["worlds"],
        "trajectories": source_card["trajectories"],
        "source": {
            **source_card["source"],
            "change_point_proof_run": PROOF,
            "risk_conflict_audit_run": RISK,
            "failed_capacity_run": CAPACITY,
        },
        "sampling": {
            "raw_frame_count": 252430,
            "raw_unique_lidar_frames": 252430,
            "effective_sample_count": 188126,
            "effective_observations": 188126,
            "effective_structure_event_count": 37162,
            "fit_worlds": 60,
            "selection_worlds": 20,
            "directed_traversals": 16078,
            "structural_identities": 1534,
            "geometry_change_identities": 76,
            "geometry_change_labels": 1031,
            "rule": "Use every sealed corrected C01-C08 Teacher observation and every emitted persistent bidirectional change-point proof record. Partition contiguous same-traversal/same-identity labels into episodes; compare each final change label to its objective first-detection, closed-confirmation and back-projected boundary.",
            "spatial_interval_m": 1.0,
            "independent_sampling_unit": "Persistent structural identity and its directed contiguous causal episodes; frames remain repeated within-unit observations and are never counted as independent events.",
        },
        "split": {
            "fit": "C01-C06 records are reported without fitting.",
            "selection": "C07-C08 records are reported without threshold or architecture selection.",
            "strict_test": "C09/C10 and every M-TARE world remain unread.",
            "world_disjoint": True,
            "trajectory_disjoint": True,
            "historical_pollution_audit": source_card["split"]["historical_pollution_audit"],
        },
        "teacher": {
            "source": "Sealed corrected causal Teacher V1R and sealed persistent bidirectional change-point proof.",
            "student_input": "No student inference is run. The audit checks the declared five-frame input support and fixed 5/8/10/12-frame past-history availability at causal confirmation.",
            "forbidden_input": "Identity, TNG, mesh and future evidence are audit-only metadata and are not proposed as deployment inputs.",
            "valid_mask": "Only exact final Teacher labels and unique causal proof matches are accepted; missing, ambiguous or duplicate matches fail closed.",
            "planner_consistency_plan": source_card["teacher"]["planner_consistency_plan"],
        },
        "leakage_audit": source_card["leakage_audit"],
        "method": {
            "main": "Exact label-timing and episode inventory: quantify proposal-versus-confirmation frames, boundary inclusion in the current five-frame history, all-event episode lengths, and fixed history availability at confirmation.",
            "baseline": "Current independent frame-classification target and five-frame causal temporal encoder.",
            "fallback": "If the mismatch is not material or 12-frame history is unavailable, revise the Teacher definition rather than adding classifier capacity. If confirmed, freeze a 12-frame past-only episode detector with multiple-instance supervision and back-projected node commit.",
        },
        "acceptance": {
            "audit_completion": "Exact 80 worlds, 188126 observations, 1534 identities, 76 change identities and 1031 final change labels; every final label maps once to the sealed proof.",
            "decision_rule": "Recommend episode supervision only if most change labels precede closed confirmation, most do not contain the boundary in five-frame history, all 152 directional confirmation episodes have 12-frame scan history, and the sealed long-history/capacity predecessors agree.",
        },
        "estimated_cost": {"compute": "CPU-only JSON audit; zero training/inference", "wall_time_hours": 0.1, "host_ram_gb": 4, "disk_gb": 0.125},
        "evidence": {
            "machine_metrics": "Episode inventory, 1031-row timing table, decision checks, recommendation, PNG/PDF/SVG, environment, command, raw log, RUN_STATE and SHA-256 seal.",
            "failure_policy": "Any source/hash/count/key/split drift, ambiguous mapping, forbidden read, resource overflow or predecessor mismatch seals FAIL; no label or threshold adaptation is allowed.",
        },
        "retention": "Keep all compact audit outputs and paper-ready mechanism figure; retain the failed capacity run as an ablation/failure result.",
    }
    write_json(CARD_PATH, card)

    inputs = []
    for run in (TEACHER, PROOF, RISK, CAPACITY):
        inputs.extend([f"{run}/RUN_STATE.json", f"{run}/metrics/summary.json", f"{run}/artifacts/evidence_sha256.txt"])
    inputs.extend([
        f"{TEACHER}/artifacts/teacher_observations.jsonl",
        f"{PROOF}/artifacts/bidirectional_change_points.jsonl",
        f"{PROOF}/artifacts/causal_change_point_labels.jsonl",
        f"{RISK}/metrics/risk_conflict_summary.json",
        f"{CAPACITY}/metrics/capacity_summary.json",
    ])
    tool_paths = {
        "data_card": str(CARD_PATH.relative_to(PROJECT_ROOT)),
        "audit_contract": "src/mtare_topo/evaluation/gse_causal_event_supervision.py",
        "executor": "tools/v3/execute_gse_causal_event_supervision_audit_v1.py",
        "runner": "tools/v3/run_gse_causal_event_supervision_audit_v1.py",
        "freezer": "tools/v3/freeze_gse_causal_event_supervision_audit_spec_v1.py",
        "evidence_integrity": "src/mtare_topo/evaluation/gse_evidence_integrity.py",
        "governance": "src/mtare_topo/governance.py",
        "preflight": "tools/v3/preflight.py",
        "create_run": "tools/v3/create_run.py",
    }
    spec = {
        "schema_version": "v3_run_spec_v1", "gate": 3, "execution_phase": 3,
        "date": "20260827", "slug": "gse_causal_event_supervision_audit_v1", "seed": 0,
        "operation": "audit",
        "question": "Is the persistent geometric change target an episode-level causal detection problem, and can existing LiDAR references support its confirmation history?",
        "method": card["method"]["main"], "baseline": card["method"]["baseline"], "fallback": card["method"]["fallback"],
        "audit_card": str(CARD_PATH.relative_to(PROJECT_ROOT)),
        "config_path": str(CARD_PATH.relative_to(PROJECT_ROOT)),
        "user_authorization": card["approval"],
        "acceptance_criteria": [
            "Exact C01-C08 80/16078/252430/188126 population and exact 1534 identities, 76 change identities, 1031 final change labels.",
            "Every final change label maps once to one emitted bidirectional point and one causal directional episode; report proposal/confirmation and five-frame boundary alignment.",
            "Audit fixed 5/8/10/12 scan histories without selecting a winning length; require all 152 directional confirmations to support 12 frames for the recommended fallback.",
            "Zero optimizer/model inference and zero C09/C10/M-TARE access; source unchanged and complete compact evidence seal.",
        ],
        "expected_counts": {
            "worlds": 80, "directed_traversals": 16078, "unique_frames": 252430,
            "causal_observations": 188126, "structural_identities": 1534,
            "change_identities": 76, "change_labels": 1031, "directional_change_episodes": 152,
            "optimizer_steps": 0, "model_inference_frames": 0, "c09_worlds_read": 0,
            "strict_test_worlds_read": 0, "mtare_worlds_read": 0,
        },
        "expected_evidence": ["Episode inventory, exact transition timing table, causal alignment checks, frozen recommendation, paper-ready figure, logs, RUN_STATE and SHA-256 seal."],
        "estimated_cost": card["estimated_cost"],
        "frozen_inputs": {path: _sha(PROJECT_ROOT / path) for path in inputs},
        "frozen_tools": {name: {"path": path, "sha256": _sha(PROJECT_ROOT / path)} for name, path in tool_paths.items()},
        "working_directory": str(PROJECT_ROOT),
        "command": [
            "/usr/bin/systemd-inhibit", "--what=sleep:shutdown", "--why=GSE causal event supervision audit", "--mode=block",
            "/usr/bin/timeout", "--signal=INT", "--kill-after=30s", "360s", PYTHON,
            "tools/v3/run_gse_causal_event_supervision_audit_v1.py", "--spec", str(SPEC_PATH),
            "--run-dir", str(PROJECT_ROOT / "results/gate3_semantics" / RUN_ID),
        ],
    }
    write_json(SPEC_PATH, spec)
    print(CARD_PATH.relative_to(PROJECT_ROOT))
    print(SPEC_PATH.relative_to(PROJECT_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
