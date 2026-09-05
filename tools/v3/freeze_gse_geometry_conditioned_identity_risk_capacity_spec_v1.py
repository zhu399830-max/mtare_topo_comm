#!/usr/bin/env python3
"""Freeze the geometry-conditioned event-risk Data Card and one-shot spec."""

from __future__ import annotations

import hashlib
from pathlib import Path

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, write_json


RUN_ID = "gate3_20260827_gse_geometry_conditioned_identity_risk_capacity_v1_seed0"
CARD_PATH = PROJECT_ROOT / "configs/v3/gate3/data_cards/gse_geometry_conditioned_identity_risk_capacity_v1.json"
SPEC_PATH = PROJECT_ROOT / "configs/v3/gate3/gse_geometry_conditioned_identity_risk_capacity_v1.json"
SOURCE_CARD = PROJECT_ROOT / "configs/v3/gate3/data_cards/gse_corrected_causal_event_training_v1.json"
TEACHER = "results/gate2_representation/gate2_20260827_gse_corrected_causal_teacher_manifest_v1r_seed0"
VERIFIER = "results/gate3_semantics/gate3_20260826_gse_exit_token_association_corrective_v2_seed0"
AUDIT = "results/gate3_semantics/gate3_20260827_gse_causal_geometry_risk_conflict_audit_v1_seed0"
PYTHON = "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    source_card = load_json(SOURCE_CARD)
    audit_summary = load_json(PROJECT_ROOT / AUDIT / "metrics/risk_conflict_summary.json")
    mechanism = audit_summary.get("mechanism_conclusion", {})
    if (
        mechanism.get("recommended_next")
        != "GEOMETRY_CONDITIONED_IDENTITY_LEVEL_RISK_CAPACITY_PROOF"
        or mechanism.get("longer_history_signal_material") is not True
        or mechanism.get("scalar_one_percent_risk_sufficient") is not False
    ):
        raise RuntimeError("sealed mechanism audit does not authorize risk capacity fitting")
    card = {
        "schema_version": "v3_data_card_v1",
        "card_id": "gse_geometry_conditioned_identity_risk_capacity_v1",
        "status": "APPROVED_FOR_ONE_IMMUTABLE_GSE_GEOMETRY_CONDITIONED_IDENTITY_RISK_CAPACITY_V1",
        "purpose": "Test whether a low-capacity multivariate causal geometry risk readout can convert transferable LiDAR geometry into safe structural node events before revising the sensor/Teacher contract.",
        "approval": {
            "status": "APPROVED",
            "approved_by": "user",
            "approved_at": "2026-08-27T23:58:00+08:00",
            "authorized_gates": [3],
            "authorized_operations": ["training"],
            "scope": "One deterministic C01-C06 fit/C07-C08 selection geometry-conditioned identity-risk capacity proof; zero upstream inference and zero C09/C10/M-TARE access.",
            "confirmation_reference": "User instructed Codex to automatically choose and execute the optimal in-scope evidence-supported route without requesting routine approvals.",
        },
        "worlds": source_card["worlds"],
        "trajectories": source_card["trajectories"],
        "source": {
            **source_card["source"],
            "mechanism_audit_run": AUDIT,
            "mechanism_audit_status": "PASS_GSE_CAUSAL_GEOMETRY_RISK_CONFLICT_AUDIT_V1",
            "mechanism_audit_seal_sha256": _sha(
                PROJECT_ROOT / AUDIT / "artifacts/evidence_sha256.txt"
            ),
        },
        "sampling": {
            "raw_frame_count": 252430,
            "raw_unique_lidar_frames": 252430,
            "effective_sample_count": 188126,
            "effective_observations": 188126,
            "effective_structure_event_count": 37162,
            "fit_worlds": 60,
            "fit_directed_traversals": 12106,
            "fit_unique_lidar_frames": 190600,
            "fit_observations": 142184,
            "selection_worlds": 20,
            "selection_directed_traversals": 3972,
            "selection_unique_lidar_frames": 61830,
            "selection_observations": 45942,
            "fit_event_counts": source_card["sampling"]["fit_event_counts"],
            "selection_event_counts": source_card["sampling"]["selection_event_counts"],
            "fit_event_identity_counts": source_card["sampling"]["fit_event_identity_counts"],
            "selection_event_identity_counts": source_card["sampling"]["selection_event_identity_counts"],
            "structure_event_counts": source_card["sampling"]["structure_event_counts"],
            "rule": "Use all sealed C01-C08 five-frame observations at 1 m traversal spacing. Build fixed causal deltas at 2/4/6/8/10/12 preceding sequence steps within the same traversal; represent missing history with a mask and never cross a traversal or split.",
            "spatial_interval_m": 1.0,
            "causal_lags_m": [2, 4, 6, 8, 10, 12],
            "independent_sampling_unit": "Persistent structural identity for non-corridor classes and parent world for corridor; each class receives equal total fit weight and each independent unit receives equal within-class mass.",
        },
        "split": {
            "fit": "C01-C06 only; determines weighted normalization and every linear softmax coefficient.",
            "selection": "C07-C08 only; selects the one non-vacuous open-set threshold and evaluates all unchanged event/identity gates.",
            "strict_test": "C09/C10 and all M-TARE worlds remain unread.",
            "world_disjoint": True,
            "trajectory_disjoint": True,
            "historical_pollution_audit": source_card["split"]["historical_pollution_audit"],
        },
        "teacher": {
            "source": "Sealed corrected causal Teacher V1R with persistent bidirectional structural identities.",
            "student_input": "Equal-weight mean of three sealed GSE predictions: signed/absolute width, height, slope and curvature deltas at fixed past lags; current structural/conditional event probabilities; current uncertainty; history availability bits.",
            "forbidden_input": "No pose, XYZ, parent/world code, identity value, edge/node/TNG field, mesh value or future frame enters the feature matrix.",
            "valid_mask": "A geometry lag is available only when current and past corrected Teacher geometry_valid are both true within the same traversal and development split. Missing lags use zero delta plus an explicit zero availability bit; no imputation from Teacher values enters inference features.",
            "planner_consistency_plan": source_card["teacher"]["planner_consistency_plan"],
        },
        "leakage_audit": source_card["leakage_audit"],
        "method": {
            "main": "One deterministic five-class linear softmax with corridor as fixed reference class, weighted fit-only normalization, L2=1e-3 and L-BFGS maximum 500 iterations; plus ten leave-family-out refits as non-selection diagnostics.",
            "baseline": "The unchanged equal-weight original three-seed event probabilities and the sealed old-directional macro-F1=0.6879041032.",
            "fallback": "If the capacity proof fails any event/identity/risk gate, stop classifier expansion and revise the causal input/Teacher contract; do not tune graph/planner thresholds.",
        },
        "acceptance": source_card["acceptance"],
        "estimated_cost": {
            "compute": "CPU-only deterministic convex fitting; 11 bounded L-BFGS fits, zero backbone inference/training.",
            "wall_time_hours": 2.0,
            "host_ram_gb": 4,
            "disk_gb": 0.25,
        },
        "evidence": {
            "machine_metrics": "Readout weights/normalization, fit weights, selection probabilities, five-class metrics, open-set gate, identity coverage, ten leave-family-out diagnostics, optimizer convergence, source integrity, environment, log and seal.",
            "failure_policy": "Any source/count/split/index/hash drift, nonconvergence, forbidden input/read, C09/C10/M-TARE access, resource overflow or unmet scientific gate seals FAIL. No retry, regularization search, lag selection or threshold weakening.",
        },
        "retention": "Keep the readout, selection outputs, fit contract, leave-family-out table, summaries, config, raw log and complete SHA-256 seal; publish figures only from sealed evidence.",
    }
    write_json(CARD_PATH, card)

    inputs = []
    for run in (TEACHER, VERIFIER, AUDIT):
        inputs.extend([f"{run}/RUN_STATE.json", f"{run}/metrics/summary.json", f"{run}/artifacts/evidence_sha256.txt"])
    inputs.extend(
        [
            f"{TEACHER}/artifacts/teacher_observations.jsonl",
            f"{VERIFIER}/artifacts/pair_cache/pairs.npz",
            f"{AUDIT}/metrics/risk_conflict_summary.json",
        ]
    )
    for seed in (0, 1, 2):
        inputs.append(f"{VERIFIER}/artifacts/models/seed{seed}/frozen_observation_features.npy")
    tool_paths = {
        "data_card": str(CARD_PATH.relative_to(PROJECT_ROOT)),
        "risk_contract": "src/mtare_topo/representation/gse_geometry_conditioned_risk.py",
        "event_gate": "src/mtare_topo/representation/gse_corrected_causal_event.py",
        "event_evaluator": "src/mtare_topo/representation/gse_rare_event_corrective.py",
        "open_set_contract": "src/mtare_topo/representation/gse_open_set_association.py",
        "executor": "tools/v3/execute_gse_geometry_conditioned_identity_risk_capacity_v1.py",
        "runner": "tools/v3/run_gse_geometry_conditioned_identity_risk_capacity_v1.py",
        "freezer": "tools/v3/freeze_gse_geometry_conditioned_identity_risk_capacity_spec_v1.py",
        "evidence_integrity": "src/mtare_topo/evaluation/gse_evidence_integrity.py",
        "governance": "src/mtare_topo/governance.py",
        "preflight": "tools/v3/preflight.py",
        "create_run": "tools/v3/create_run.py",
    }
    spec = {
        "schema_version": "v3_run_spec_v1",
        "gate": 3,
        "execution_phase": 3,
        "date": "20260827",
        "slug": "gse_geometry_conditioned_identity_risk_capacity_v1",
        "seed": 0,
        "operation": "training",
        "question": "Can fixed multi-lag LiDAR geometry and uncertainty support safe identity-level structural event decisions with a low-capacity readout?",
        "method": card["method"]["main"],
        "baseline": card["method"]["baseline"],
        "fallback": card["method"]["fallback"],
        "data_card": str(CARD_PATH.relative_to(PROJECT_ROOT)),
        "config_path": str(CARD_PATH.relative_to(PROJECT_ROOT)),
        "user_authorization": card["approval"],
        "acceptance_criteria": [
            "Exact C01-C06 60/142184 fit and C07-C08 20/45942 selection populations with 17 change identities; fit-only weighted normalization and parameters.",
            "Selection event macro-F1>=0.7379041032, structural precision>=0.98, false acceptance<=0.01 and recall>=0.40.",
            "Correct-class identity coverage: junction>=0.90, terminal>=0.90, turn>=0.40 and change-point>=7/17; every topology family has a nonempty safe accepted set.",
            "Ten leave-family-out fit diagnostics, deterministic convergence, zero upstream inference and zero C09/C10/M-TARE access.",
        ],
        "expected_counts": {
            "fit_worlds": 60,
            "fit_observations": 142184,
            "selection_worlds": 20,
            "selection_observations": 45942,
            "selection_change_identities": 17,
            "features": 60,
            "leave_family_out_fits": 10,
            "risk_readout_scored_observations": 330310,
            "upstream_model_inference_frames": 0,
            "c09_worlds_read": 0,
            "strict_test_worlds_read": 0,
            "mtare_worlds_read": 0,
        },
        "expected_evidence": [
            "Frozen linear readout and normalization, fit weights, selection outputs, full five-class/open-set/identity metrics, ten leave-family-out rows, environment, raw log, RUN_STATE and complete SHA-256 seal."
        ],
        "estimated_cost": card["estimated_cost"],
        "frozen_inputs": {path: _sha(PROJECT_ROOT / path) for path in inputs},
        "frozen_tools": {
            name: {"path": path, "sha256": _sha(PROJECT_ROOT / path)}
            for name, path in tool_paths.items()
        },
        "working_directory": str(PROJECT_ROOT),
        "command": [
            "/usr/bin/systemd-inhibit",
            "--what=sleep:shutdown",
            "--why=GSE geometry conditioned identity risk capacity",
            "--mode=block",
            "/usr/bin/timeout",
            "--signal=INT",
            "--kill-after=30s",
            "7260s",
            PYTHON,
            "tools/v3/run_gse_geometry_conditioned_identity_risk_capacity_v1.py",
            "--spec",
            str(SPEC_PATH),
            "--run-dir",
            str(PROJECT_ROOT / "results/gate3_semantics" / RUN_ID),
        ],
    }
    write_json(SPEC_PATH, spec)
    print(CARD_PATH.relative_to(PROJECT_ROOT))
    print(SPEC_PATH.relative_to(PROJECT_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
