#!/usr/bin/env python3
"""Freeze the typed dual-Composer readiness Data Card and run spec."""

from __future__ import annotations

import hashlib
from pathlib import Path

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, validate_data_card, validate_run_spec, write_json


RUN_ID = "gate3_20260829_gse_typed_composer_readiness_v1_seed0"
CARD = PROJECT_ROOT / "configs/v3/gate3/data_cards/gse_typed_composer_readiness_v1.json"
SPEC = PROJECT_ROOT / "configs/v3/gate3/gse_typed_composer_readiness_v1.json"
TEACHER = "results/gate2_representation/gate2_20260827_gse_corrected_causal_teacher_manifest_v1r_seed0"
BASELINE = "results/gate3_semantics/gate3_20260829_gse_sparse_circular_relation_transport_three_seed_training_v2r5_seed0"
SOURCE_CARD = PROJECT_ROOT / "configs/v3/gate3/data_cards/gse_sparse_circular_relation_transport_three_seed_training_v2r5.json"
PYTHON = "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    observability = load_json(PROJECT_ROOT / "results/gate3_semantics/gate3_20260829_gse_composer_observability_v1_seed0/metrics/summary.json")
    if (
        observability.get("overall_status") != "PASS_GSE_COMPOSER_OBSERVABILITY_V1"
        or observability.get("scientific_pass") is not True
    ):
        raise RuntimeError("formal explicit Composer observability must pass before readiness")
    baseline_state = load_json(PROJECT_ROOT / BASELINE / "RUN_STATE.json")
    if baseline_state.get("state") != "COMPLETED" or baseline_state.get("error") is not None:
        raise RuntimeError("sealed V2R5 baseline is not complete")
    source = load_json(SOURCE_CARD)
    c07 = sorted(world for world in source["worlds"]["validation"] if world.endswith("_C07"))
    c08 = sorted(world for world in source["worlds"]["validation"] if world.endswith("_C08"))
    trajectories = [
        {**record, "split": "train"}
        for record in source["trajectories"] if record["world"] in set(c07)
    ]
    if len(c07) != 10 or len(c08) != 10 or len(trajectories) != 10:
        raise RuntimeError("C07 readiness population drift")
    method = {
        "main": "Two independently typed small neural Composers: unordered exit-token/transport composition for junction-terminal decisions and causal metric-sequence composition for turn-transition events plus bounded backprojection.",
        "baseline": "Historical ActionSetNodeDetector and CausalGeometryDeltaEventHead are interface baselines only because they consume descriptors, three seeds jointly, hidden context or old event logits.",
        "fallback": "Any readiness failure stops Composer training and is corrected only at the explicit interface or causal implementation level; no graph, threshold or planner compensation.",
    }
    card = {
        "schema_version": "v3_data_card_v1",
        "card_id": "gse_typed_composer_readiness_v1",
        "status": "APPROVED_FOR_ONE_IMMUTABLE_GSE_TYPED_COMPOSER_READINESS_V1",
        "purpose": "Prove before training that both GSE structural Composers consume only explicit five-frame causal geometry, are permutation/rotation consistent, deterministic, reject malformed inputs, support bounded backprojection and have complete finite gradients.",
        "approval": {
            "status": "APPROVED", "approved_by": "user-standing-authorization",
            "approved_at": "2026-08-29T20:30:00+08:00",
            "scope": "One immutable CPU-only zero-training readiness run on ten fixed C07 samples plus exact 21,548-row provenance; no C08/C09/C10, checkpoint, graph, planner or M-TARE.",
            "authorized_operations": ["audit"], "authorized_gates": [3],
            "confirmation_reference": "User explicitly authorized autonomous optimal continuation without further approval prompts.",
        },
        "source": {
            "raw_sources": [
                "Sealed V2R5 seed0 C07 explicit token/transport/metric-geometry predictions.",
                "Sealed corrected causal Teacher V1R used only to select two fixed examples per event and reconstruct traversal-bounded past indices.",
            ],
            "prediction_run": BASELINE, "teacher_run": TEACHER,
            "license_or_allowed_use": "Local project-generated procedural worlds, objective Teacher and frozen model predictions.",
        },
        "worlds": {
            "train": c07, "validation": c08, "ssl": [], "normalization": [],
            "teacher_calibration": [], "threshold_calibration": [], "augmentation_tuning": [],
            "checkpoint_selection": [], "strict_test": source["worlds"]["strict_test"],
            "readiness": c07,
            "audit_scope": "C07 only for interface readiness; C08/C09/C10 and M-TARE remain unread.",
        },
        "trajectories": trajectories,
        "sampling": {
            "raw_frame_count": 21548, "effective_sample_count": 10,
            "effective_structure_event_count": 10,
            "structure_event_counts": {
                "corridor": 17113, "junction": 3189, "terminal": 900,
                "turn": 279, "geometry_transition": 67,
            },
            "independent_sampling_units": "10 disjoint C07 procedural worlds; the ten forward samples are deterministic interface fixtures, not statistical performance samples.",
            "spatial_interval_m": 1.0, "temporal_window_frames": 5,
            "rule": "Verify all ten C07 archive counts, then select the first two corrected causal rows of each of five event types; reconstruct no more than four strictly earlier rows within the same traversal for metric history.",
        },
        "teacher": {
            "source": "Sealed corrected causal Teacher V1R from objective TNG, splines and persistent bidirectionally consistent past-confirmed change points.",
            "valid_mask": "Teacher traversal and sequence indices construct a left-padded suffix only; identity is retained solely in the evidence record and never enters either Composer.",
            "planner_consistency_plan": "No planner or graph is invoked. A later adapter def-use audit is mandatory after scientific training gates pass.",
        },
        "split": {
            "world_disjoint": True, "trajectory_disjoint": True,
            "historical_pollution_audit": "Readiness uses C07 only and performs no optimization, normalization, calibration, threshold selection or checkpoint selection. C08 and strict tests remain closed.",
            "selection": "Deterministic two-row-per-event interface fixture only.",
            "transfer": "None; readiness establishes implementation properties, not generalization.",
        },
        "leakage_audit": dict(source["leakage_audit"]),
        "method": method,
        "estimated_cost": {"compute": "CPU-only unit and ten-sample readiness", "wall_time_hours": 0.05, "host_ram_gb": 1, "disk_gb": 0.1, "gpu": 0},
        "failure_policy": "Any forbidden input, source/count/index/environment drift, future/cross-traversal history, equivariance/determinism/gradient failure or resource excess seals FAIL and stops training.",
    }
    report = validate_data_card(card)
    if not report.passed:
        raise RuntimeError("generated readiness Data Card invalid: " + "; ".join(report.errors))
    write_json(CARD, card)

    inputs = [
        str(SOURCE_CARD.relative_to(PROJECT_ROOT)),
        f"{TEACHER}/RUN_STATE.json", f"{TEACHER}/metrics/summary.json",
        f"{TEACHER}/artifacts/evidence_sha256.txt", f"{TEACHER}/artifacts/teacher_observations.jsonl",
        f"{BASELINE}/RUN_STATE.json", f"{BASELINE}/metrics/summary.json",
        f"{BASELINE}/artifacts/evidence_sha256.txt",
        "results/gate3_semantics/gate3_20260829_gse_composer_observability_v1_seed0/RUN_STATE.json",
        "results/gate3_semantics/gate3_20260829_gse_composer_observability_v1_seed0/metrics/summary.json",
        "results/gate3_semantics/gate3_20260829_gse_composer_observability_v1_seed0/artifacts/evidence_sha256.txt",
        "configs/v3/gate2/environments/phase3_torch290_cu129_zarr2187_v1r2.json",
    ]
    prediction_root = PROJECT_ROOT / BASELINE / "artifacts/models/seed0/development_predictions"
    predictions = sorted(prediction_root.glob("*_C07.npz"))
    if len(predictions) != 10:
        raise RuntimeError("expected ten frozen C07 prediction archives")
    inputs.extend(str(path.relative_to(PROJECT_ROOT)) for path in predictions)
    tools = {
        "data_card": str(CARD.relative_to(PROJECT_ROOT)),
        "composer_module": "src/mtare_topo/representation/gse_typed_composers.py",
        "unit_tests": "tests/v3/unit/test_gse_typed_composers.py",
        "evaluator": "tools/v3/evaluate_gse_typed_composer_readiness_v1.py",
        "runner": "tools/v3/run_gse_typed_composer_readiness_v1.py",
        "freezer": "tools/v3/freeze_gse_typed_composer_readiness_spec_v1.py",
        "governance": "src/mtare_topo/governance.py",
        "preflight": "tools/v3/preflight.py", "create_run": "tools/v3/create_run.py",
    }
    spec = {
        "schema_version": "v3_run_spec_v1", "gate": 3, "execution_phase": 3,
        "date": "20260829", "slug": "gse_typed_composer_readiness_v1", "seed": 0,
        "operation": "audit",
        "question": "Can both GSE Composers enforce an explicit causal geometry-only interface and satisfy readiness before any training?",
        "method": method["main"], "baseline": method["baseline"], "fallback": method["fallback"],
        "data_card": str(CARD.relative_to(PROJECT_ROOT)), "config_path": str(CARD.relative_to(PROJECT_ROOT)),
        "user_authorization": card["approval"],
        "acceptance_criteria": [
            "Exact ten C07 worlds, 21,548 prediction/Teacher rows and ten fixed samples spanning all five events.",
            "Typed forward interfaces expose no context, old event output, descriptor, pose, world, TNG, identity or future frame.",
            "Action Composer is token-permutation and global-rotation invariant; both Composers are batch-consistent and exactly deterministic.",
            "Metric backprojection has support only on the valid five-frame past/current suffix.",
            "Every trainable parameter receives a finite gradient; Action/Metric parameter counts remain <=250k/100k.",
            "Zero optimizer/checkpoint/C08/C09/C10/M-TARE/graph/planner and complete evidence seal.",
        ],
        "expected_counts": {
            "c07_worlds_read": 10, "c07_observations_provenance": 21548,
            "readiness_forward_samples": 10, "event_types_represented": 5,
            "optimizer_steps": 0, "checkpoints_created": 0,
            "c08_worlds_read": 0, "c09_worlds_read": 0, "c10_worlds_read": 0,
            "mtare_worlds_read": 0, "graph_replays": 0,
        },
        "expected_evidence": [
            "Typed contract, real sample manifest, numerical invariance/repeat errors, finite-gradient report, parameter counts, 15 checks, PNG/PDF/SVG, logs, environment, RUN_STATE and SHA-256 seal.",
        ],
        "estimated_cost": card["estimated_cost"],
        "frozen_inputs": {path: sha256(PROJECT_ROOT / path) for path in inputs},
        "frozen_tools": {name: {"path": path, "sha256": sha256(PROJECT_ROOT / path)} for name, path in tools.items()},
        "working_directory": str(PROJECT_ROOT),
        "command": [
            "/usr/bin/timeout", "600s", PYTHON,
            "tools/v3/run_gse_typed_composer_readiness_v1.py", "--spec", str(SPEC),
            "--run-dir", str(PROJECT_ROOT / "results/gate3_semantics" / RUN_ID),
        ],
    }
    spec_report = validate_run_spec(spec)
    if not spec_report.passed:
        raise RuntimeError("generated readiness run spec invalid: " + "; ".join(spec_report.errors))
    write_json(SPEC, spec)
    print(CARD.relative_to(PROJECT_ROOT)); print(SPEC.relative_to(PROJECT_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
