#!/usr/bin/env python3
"""Freeze the dual-Composer observability Data Card and one-shot run spec."""

from __future__ import annotations

import hashlib
from pathlib import Path

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import (
    load_json,
    validate_data_card,
    validate_run_spec,
    write_json,
)


RUN_ID = "gate3_20260829_gse_composer_observability_v1_seed0"
CARD = PROJECT_ROOT / "configs/v3/gate3/data_cards/gse_composer_observability_v1.json"
SPEC = PROJECT_ROOT / "configs/v3/gate3/gse_composer_observability_v1.json"
TEACHER = "results/gate2_representation/gate2_20260827_gse_corrected_causal_teacher_manifest_v1r_seed0"
BASELINE = "results/gate3_semantics/gate3_20260829_gse_sparse_circular_relation_transport_three_seed_training_v2r5_seed0"
PYTHON = "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python"
SOURCE_CARD = PROJECT_ROOT / "configs/v3/gate3/data_cards/gse_sparse_circular_relation_transport_three_seed_training_v2r5.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    state = load_json(PROJECT_ROOT / BASELINE / "RUN_STATE.json")
    if state.get("state") != "COMPLETED" or state.get("error") is not None:
        raise RuntimeError("V2R5 must be completed without a system error before freezing this audit")
    summary = load_json(PROJECT_ROOT / BASELINE / "metrics/summary.json")
    if (
        summary.get("optimizer_steps") != 36690
        or not (PROJECT_ROOT / BASELINE / "artifacts/evidence_sha256.txt").is_file()
    ):
        raise RuntimeError("V2R5 must be completely sealed before freezing this audit")

    source_card = load_json(SOURCE_CARD)
    c07_worlds = sorted(
        world for world in source_card["worlds"]["validation"] if world.endswith("_C07")
    )
    c08_worlds = sorted(
        world for world in source_card["worlds"]["validation"] if world.endswith("_C08")
    )
    trajectories = []
    for source in source_card["trajectories"]:
        if source["world"] not in set(c07_worlds + c08_worlds):
            continue
        record = dict(source)
        record["split"] = "train" if record["world"].endswith("_C07") else "validation"
        trajectories.append(record)
    if len(c07_worlds) != 10 or len(c08_worlds) != 10 or len(trajectories) != 20:
        raise RuntimeError("source Data Card C07/C08 population drift")

    observability = {
        "method": {
            "main": "Three per-seed 642-D explicit-state probes plus a score-level equal-weight ensemble, together with train-free physical scores, evaluated C07-to-C08 for four corrected events.",
            "baseline": "The V2R5 independent hidden-context event head scored against the same corrected Teacher; it is comparison-only and cannot enter the explicit probe.",
            "fallback": "If novelty-essential turn/transition observability fails, stop dual-Composer training and execute RouteGeometryProfile visibility/Teacher uniqueness proof.",
        },
        "acceptance": {
            "explicit_ensemble_c08_auc": {"junction": 0.90, "terminal": 0.90, "turn": 0.70, "geometry_transition": 0.65},
            "turn_and_transition_ap_over_prevalence_min": 2.0,
            "fit_transfer_auc_gap": "diagnostic_only_because_C07_fit_AUC_is_in_sample",
            "seed_transition_auc_min": 0.60,
            "seed_transition_minimum_count": 2,
            "provenance": "The transition AUC 0.65 gate predates this audit in the sealed causal geometry-delta observability protocol; other values require nontrivial transferable signal and do not relax later method-performance gates.",
        },
        "event_frames": {
            "C07": {"corridor": 17113, "junction": 3189, "terminal": 900, "turn": 279, "geometry_transition": 67},
            "C08": {"corridor": 19234, "junction": 3676, "terminal": 1074, "turn": 237, "geometry_transition": 173},
        },
        "event_identities": {
            "C07": {"junction": 69, "terminal": 59, "turn": 43, "geometry_transition": 5},
            "C08": {"junction": 77, "terminal": 69, "turn": 52, "geometry_transition": 12},
        },
        "diversity_risk": "C07 contains only five transition identities. Threshold-free C08 AUC/AP and identity coverage are mandatory; no final deployment threshold is frozen here.",
        "forbidden_inputs": "Encoder context, old event labels/logits as probe inputs, token/place descriptors, GT identity, pose, world, TNG and future frames.",
        "failure_policy": "Any source/count/index/split/environment drift, forbidden input, missing seed, C08 adaptation, resource excess or unmet observability gate seals FAIL. A scientific FAIL triggers only RouteGeometryProfile proof; it never authorizes graph or planner compensation.",
        "retention": "Keep CSV/JSON, PNG/PDF/SVG paper diagnostic, exact command/environment/logs, RUN_STATE and SHA-256 seal.",
    }
    card = {
        "schema_version": "v3_data_card_v1",
        "card_id": "gse_composer_observability_v1",
        "status": "APPROVED_FOR_ONE_IMMUTABLE_GSE_COMPOSER_OBSERVABILITY_V1",
        "purpose": "Decide whether sealed V2R5 explicit token, transport and metric-geometry outputs contain enough corrected causal structure information to justify the two GSE Composers without hidden-context bypass.",
        "approval": {
            "status": "APPROVED",
            "approved_by": "user-standing-authorization",
            "approved_at": "2026-08-29T18:30:00+08:00",
            "scope": "One immutable CPU-only C07-to-C08 observability audit over three sealed V2R5 seeds; fixed diagnostic probes only; no checkpoint update, C09/C10, graph, planner or M-TARE.",
            "authorized_operations": ["audit"],
            "authorized_gates": [3],
            "confirmation_reference": "User instructed autonomous evidence-optimal continuation without routine approval prompts.",
        },
        "source": {
            "raw_sources": [
                "Sealed V2R5 C07/C08 per-seed development prediction archives.",
                "Sealed corrected causal Teacher V1R observation manifest.",
                "Sealed V2R5 source Data Card trajectories and world isolation contract.",
            ],
            "teacher_run": TEACHER,
            "prediction_run": BASELINE,
            "license_or_allowed_use": "Local project-generated procedural worlds, objective Teacher and model predictions.",
        },
        "worlds": {
            "train": c07_worlds,
            "validation": c08_worlds,
            "ssl": [],
            "normalization": [],
            "teacher_calibration": [],
            "threshold_calibration": [],
            "augmentation_tuning": [],
            "checkpoint_selection": [],
            "strict_test": source_card["worlds"]["strict_test"],
            "audit_scope": "C07 diagnostic probe fit and one untouched C08 transfer; C09/C10/M-TARE remain unread.",
        },
        "trajectories": trajectories,
        "sampling": {
            "raw_frame_count": 45942,
            "effective_sample_count": 45942,
            "effective_structure_event_count": 45942,
            "structure_event_counts": {
                "c07_corridor": 17113, "c07_junction": 3189, "c07_terminal": 900, "c07_turn": 279, "c07_geometry_transition": 67,
                "c08_corridor": 19234, "c08_junction": 3676, "c08_terminal": 1074, "c08_turn": 237, "c08_geometry_transition": 173
            },
            "spatial_interval_m": 1.0,
            "temporal_window_frames": 5,
            "rule": "Use every sealed C07/C08 causal observation exactly once; C07 fits fixed diagnostic probes and C08 is transferred once without adaptation.",
        },
        "teacher": {
            "source": "Sealed corrected causal Teacher V1R using objective TNG identities and persistent bidirectionally consistent past-confirmed change points.",
            "valid_mask": "Every prediction row must have a unique corrected Teacher global identity; causal geometry history is masked before traversal start and never crosses traversal.",
            "planner_consistency_plan": "Identity is used only after scoring for coverage. No graph, planner or deployment checkpoint is produced.",
        },
        "split": {
            "world_disjoint": True,
            "trajectory_disjoint": True,
            "historical_pollution_audit": "C07 and C08 are disjoint procedural worlds. C07 alone fits the diagnostic probe/reporting threshold; C08 is opened once after all probe settings freeze. C09/C10/M-TARE remain excluded.",
            "selection": "Fixed C=1 class-balanced linear probe and 0.98-precision reporting threshold on C07 only.",
            "transfer": "Exactly one C08 score evaluation with no feature, probe or threshold adaptation.",
        },
        "leakage_audit": dict(source_card["leakage_audit"]),
        "estimated_cost": {"compute": "CPU-only, 12 fixed diagnostic linear probes and one score-level three-seed ensemble", "wall_time_hours": 1.0, "host_ram_gb": 4, "disk_gb": 0.5, "gpu": 0},
        "observability_contract": observability,
    }
    card_report = validate_data_card(card)
    if not card_report.passed:
        raise RuntimeError(
            "generated Composer observability Data Card is invalid: "
            + "; ".join(card_report.errors)
        )
    write_json(CARD, card)

    inputs = [
        f"{TEACHER}/RUN_STATE.json",
        f"{TEACHER}/metrics/summary.json",
        f"{TEACHER}/artifacts/evidence_sha256.txt",
        f"{TEACHER}/artifacts/teacher_observations.jsonl",
        f"{BASELINE}/RUN_STATE.json",
        f"{BASELINE}/metrics/summary.json",
        f"{BASELINE}/artifacts/evidence_sha256.txt",
    ]
    for seed in range(3):
        root = f"{BASELINE}/artifacts/models/seed{seed}"
        inputs.extend((f"{root}/best.pt", f"{root}/summary.json", f"{root}/history.json"))
        predictions = sorted((PROJECT_ROOT / root / "development_predictions").glob("*.npz"))
        if len(predictions) != 20:
            raise RuntimeError(f"seed{seed} prediction population is incomplete")
        inputs.extend(str(path.relative_to(PROJECT_ROOT)) for path in predictions)
    tools = {
        "audit_card": str(CARD.relative_to(PROJECT_ROOT)),
        "audit_contract": "src/mtare_topo/evaluation/gse_composer_observability.py",
        "audit_tests": "tests/v3/unit/test_gse_composer_observability.py",
        "evaluator_tests": "tests/v3/unit/test_evaluate_gse_composer_observability_v1.py",
        "evaluator": "tools/v3/evaluate_gse_composer_observability_v1.py",
        "runner": "tools/v3/run_gse_composer_observability_v1.py",
        "freezer": "tools/v3/freeze_gse_composer_observability_spec_v1.py",
        "governance": "src/mtare_topo/governance.py",
        "preflight": "tools/v3/preflight.py",
        "create_run": "tools/v3/create_run.py",
    }
    spec = {
        "schema_version": "v3_run_spec_v1",
        "gate": 3,
        "execution_phase": 3,
        "date": "20260829",
        "slug": "gse_composer_observability_v1",
        "seed": 0,
        "operation": "audit",
        "question": "Can V2R5 explicit outputs support both the action-set and novelty-essential metric-change Composers on corrected causal events without hidden-context bypass?",
        "method": observability["method"]["main"],
        "baseline": observability["method"]["baseline"],
        "fallback": observability["method"]["fallback"],
        "data_card": str(CARD.relative_to(PROJECT_ROOT)),
        "config_path": str(CARD.relative_to(PROJECT_ROOT)),
        "user_authorization": card["approval"],
        "acceptance_criteria": [
            "Exact C07/C08 21548/24394 rows, 10/10 worlds and corrected event/identity populations.",
            "Explicit ensemble C08 AUC junction/terminal/turn/transition >=0.90/0.90/0.70/0.65.",
            "Turn and transition C08 AP each >=2x prevalence; C07 in-sample fit to C08 AUC gaps are reported but are not a pass gate.",
            "At least two seeds have transition C08 AUC>=0.60.",
            "Zero main-model optimizer/checkpoint update, C09/C10/M-TARE/graph; complete paper figure and seal.",
        ],
        "expected_counts": {
            "c07_worlds": 10,
            "c08_worlds": 10,
            "c07_observations": 21548,
            "c08_observations": 24394,
            "diagnostic_linear_probe_fits": 12,
            "main_model_optimizer_steps": 0,
            "checkpoint_updates": 0,
            "c09_worlds_read": 0,
            "c10_worlds_read": 0,
            "mtare_worlds_read": 0,
            "graph_replays": 0,
        },
        "expected_evidence": ["Per-seed/ensemble train-free, independent-head and explicit-probe AUC/AP/transfer threshold/identity coverage; PNG/PDF/SVG; environment, commands, logs, RUN_STATE and full SHA-256 seal."],
        "estimated_cost": card["estimated_cost"],
        "frozen_inputs": {path: _sha256(PROJECT_ROOT / path) for path in inputs},
        "frozen_tools": {name: {"path": path, "sha256": _sha256(PROJECT_ROOT / path)} for name, path in tools.items()},
        "working_directory": str(PROJECT_ROOT),
        "command": [
            "/usr/bin/timeout",
            "3600s",
            PYTHON,
            "tools/v3/run_gse_composer_observability_v1.py",
            "--spec",
            str(SPEC),
            "--run-dir",
            str(PROJECT_ROOT / "results/gate3_semantics" / RUN_ID),
        ],
    }
    spec_report = validate_run_spec(spec)
    if not spec_report.passed:
        raise RuntimeError(
            "generated Composer observability run spec is invalid: "
            + "; ".join(spec_report.errors)
        )
    write_json(SPEC, spec)
    print(CARD.relative_to(PROJECT_ROOT))
    print(SPEC.relative_to(PROJECT_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
