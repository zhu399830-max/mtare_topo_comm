#!/usr/bin/env python3
"""Freeze the one corrected-slope C09 perception qualification."""

from __future__ import annotations

import hashlib
from pathlib import Path

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, validate_data_card, write_json


OLD_CARD = PROJECT_ROOT / "configs/v3/gate3/data_cards/gse_perception_validation_v1.json"
CARD = PROJECT_ROOT / "configs/v3/gate3/data_cards/gse_corrected_perception_validation_v1.json"
SPEC = PROJECT_ROOT / "configs/v3/gate3/gse_corrected_perception_validation_v1.json"
DATASET = PROJECT_ROOT / "results/gate2_representation/gate2_20260824_gse_deduplicated_dataset_export_v1_seed0"
MAIN_TRAINING = PROJECT_ROOT / "results/gate2_representation/gate2_20260824_gse_graph_three_seed_training_v1r_seed0"
CORRECTIVE = PROJECT_ROOT / "results/gate2_representation/gate2_20260824_gse_slope_corrective_three_seed_training_v1r_seed0"
ORIGINAL_C09 = PROJECT_ROOT / "results/gate3_semantics/gate3_20260824_gse_perception_validation_v1_seed0"
RUN_ID = "gate3_20260824_gse_corrected_perception_validation_v1_seed0"
PYTHON = "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _record(relative: str) -> dict[str, str]:
    return {"path": relative, "sha256": _sha256(PROJECT_ROOT / relative)}


def freeze() -> dict:
    if CARD.exists() or SPEC.exists():
        raise RuntimeError("corrected perception Data Card/spec already exists; refusing overwrite")
    old = load_json(OLD_CARD)
    card = dict(old)
    card["card_id"] = "gse_corrected_perception_validation_v1"
    card["status"] = "APPROVED_FOR_ONE_IMMUTABLE_GSE_CORRECTED_PERCEPTION_VALIDATION"
    card["purpose"] = (
        "Evaluate the frozen three-seed physics-guided slope corrective on every C09 causal sequence, replace only slope_deg in the immutable original perception gate, and require both the original full gate and independent learned-over-five-frame-prior gain."
    )
    card["approval"] = dict(old["approval"])
    card["approval"]["approved_at"] = "2026-08-24T16:55:00+08:00"
    card["approval"]["scope"] = (
        "One immutable Gate-3 C09 evaluation over exactly 10 worlds, 32678 unique frames and 24462 causal sequences. Reuse sealed original event/exit/association/other-geometry evidence, infer three frozen corrective seeds, zero updates and zero C10/M-TARE reads."
    )
    card["approval"]["confirmation_reference"] = (
        "The user granted continuous execution for the fixed GSE-Graph paper program and requested no repeated approval prompts; the new method, data and gates were reported before this run."
    )
    card["source"] = dict(old["source"])
    card["source"]["raw_sources"] = list(old["source"]["raw_sources"]) + [
        "Sealed PASS slope corrective V1R with three frozen checkpoints, fit-only normalization, exact 115-file seal and train actual-path audit.",
        "Sealed original C09 perception FAIL; its event, exit, association, axis, width, height, curvature, baselines and calibration remain immutable and are reused exactly.",
    ]
    card["split"] = dict(old["split"])
    card["split"]["historical_pollution_audit"] += (
        " The corrective was fit/selected only on C01-C06/C07-C08 deterministic scan geometry. Its actual cache source paths contain exactly 80 C01-C08 train shards and no C09/C10. C09 is read here only after all corrective checkpoints and gates are frozen."
    )
    card["methods"] = {
        "unchanged_outputs": "Reuse the exact sealed original C09 event, axis, width, height, curvature, place/exit descriptors, exit tokens, calibration decisions and M1D/nonlearning baselines.",
        "corrected_slope": "For each unique C09 frame compute the same six deterministic geometry features; assemble five causal rows, apply C01-C06-only normalization and each frozen GRU residual checkpoint; replace only per-seed slope_deg for the gate.",
        "forbidden": "No model update, threshold fit, checkpoint selection, output blending, parent selection, C10/M-TARE read or change to any non-slope result.",
    }
    card["metrics_and_pre_registered_gates"] = {
        "original_complete_gate": "Event gain, four-field geometry improvement/no-field-regression and non-vacuous >=0.98 precision <=0.01 false-accept place/exit association all remain unchanged.",
        "learned_slope_gate": "Three-seed mean corrected slope improvement over five-frame analytic prior >=5%; at least two seeds >=5%; no seed overall regression; no C09 world three-seed-mean regression beyond 5%.",
        "identity": "All 24462 global sequence indices, parents, targets, current/prior/corrected slopes and predicted error scales are preserved.",
    }
    card["estimated_cost"] = {
        "disk_gb": 2.0,
        "wall_time_hours": 1.0,
        "compute": "CPU deterministic geometry over 32678 unique C09 frames plus three small frozen GPU residual passes; host RSS <=8 GiB, output <=2 GiB, zero optimizer/model updates.",
    }
    card["evidence"] = {
        "machine_metrics": "Ten per-world feature archives, three complete prediction archives and metrics, corrected complete gate, source-path/read audit, source seals before/after, environment, logs, state and final seal.",
        "complete_visual_review": "Only after sealed PASS, publish the corrected perception and slope-ablation paper bundle in PNG/PDF/SVG plus CSV/JSON/provenance/hash.",
        "failure_policy": "Any source/tool/checkpoint/normalization drift, incomplete sequence/world, forbidden read/update, original gate regression, learned slope gate failure or resource overrun is FAIL and stops topology.",
    }
    report = validate_data_card(card)
    if not report.passed:
        raise RuntimeError(f"corrected perception Data Card invalid: {report.errors}")
    write_json(CARD, card)
    tools = {
        "data_card": str(CARD.relative_to(PROJECT_ROOT)),
        "runner": "tools/v3/run_gse_corrected_perception_validation_v1.py",
        "slope_evaluator": "tools/v3/evaluate_gse_slope_corrective_c09_v1.py",
        "gate_summarizer": "tools/v3/summarize_gse_corrected_perception_gate_v1.py",
        "corrected_gate_logic": "src/mtare_topo/evaluation/gse_slope_corrective_gate.py",
        "original_gate_logic": "src/mtare_topo/evaluation/gse_perception_gate.py",
        "corrective_dataset": "src/mtare_topo/data/gse_slope_corrective_dataset.py",
        "corrective_model": "src/mtare_topo/representation/gse_slope_corrective.py",
        "range_geometry_baseline": "src/mtare_topo/semantics/range_geometry_baseline.py",
        "sensor_contract": "src/mtare_topo/data/cano_sensor_smoke.py",
        "governance": "src/mtare_topo/governance.py",
        "preflight": "tools/v3/preflight.py",
    }
    inputs = [
        DATASET / "RUN_STATE.json", DATASET / "metrics/summary.json", DATASET / "artifacts/evidence_sha256.txt",
        MAIN_TRAINING / "RUN_STATE.json", MAIN_TRAINING / "metrics/summary.json", MAIN_TRAINING / "artifacts/evidence_sha256.txt",
        CORRECTIVE / "RUN_STATE.json", CORRECTIVE / "metrics/summary.json", CORRECTIVE / "artifacts/evidence_sha256.txt",
        CORRECTIVE / "artifacts/slope_corrective_cache/normalization.json", CORRECTIVE / "artifacts/slope_corrective_cache/manifest.json",
        ORIGINAL_C09 / "RUN_STATE.json", ORIGINAL_C09 / "metrics/summary.json", ORIGINAL_C09 / "metrics/perception_gate.json", ORIGINAL_C09 / "artifacts/evidence_sha256.txt",
    ] + [CORRECTIVE / f"artifacts/models/seed{seed}/best.pt" for seed in (0, 1, 2)]
    spec = {
        "schema_version": "v3_run_spec_v1",
        "gate": 3,
        "execution_phase": 3,
        "date": "20260824",
        "slug": "gse_corrected_perception_validation_v1",
        "operation": "threshold_calibration",
        "question": "Does the frozen physics-guided slope residual repair the only failed C09 field while preserving every original GSE event, geometry, association and no-leakage gate and showing genuine learned gain beyond five-frame analytic averaging?",
        "method": "Read all 32678 unique C09 frames once, compute the frozen six-feature causal geometry sequence, infer corrective seeds 0/1/2, replace only slope_deg in the exact sealed original gate, and additionally apply a learned-versus-five-frame-prior and per-world safety gate. No calibration or model update is performed.",
        "baseline": "Primary corrective ablation is the same five-frame analytic slope mean with zero residual; secondary diagnostic is current-frame analytic slope. All original M1D, event, association and four-field nonlearning comparisons are reused byte-for-byte from the sealed original C09 run.",
        "user_authorization": card["approval"],
        "seed": 0,
        "data_card": str(CARD.relative_to(PROJECT_ROOT)),
        "config_path": str(CARD.relative_to(PROJECT_ROOT)),
        "working_directory": str(PROJECT_ROOT),
        "command": [
            "/usr/bin/systemd-inhibit", "--what=sleep:shutdown", "--why=GSE corrected C09 perception validation", "--mode=block",
            "/usr/bin/timeout", "--signal=INT", "--kill-after=300s", "3900s", PYTHON,
            "tools/v3/run_gse_corrected_perception_validation_v1.py", "--spec", str(SPEC), "--run-dir", str(PROJECT_ROOT / "results/gate3_semantics" / RUN_ID),
        ],
        "acceptance_criteria": [
            "Exactly ten C09 worlds, 32678 unique frames and 24462 sequences are evaluated for all three frozen corrective seeds with zero updates and zero C10/M-TARE reads.",
            "The original event, width, height, curvature, axis, exit and place/exit association evidence is reused from the exact 49-file sealed original run; only slope_deg changes.",
            "The complete original perception gate passes after corrected slope replacement: mean four-field improvement >=10% and no field regression beyond 5%, alongside unchanged event and association gates.",
            "Corrected slope improves over the five-frame analytic prior by >=5% on three-seed mean, at least two seeds individually, no seed regresses, and no C09 world three-seed mean regresses beyond 5%.",
            "All four source seals verify before/after; output <=2 GiB, child RSS <=8 GiB and wall time <=1 hour.",
        ],
        "stop_conditions": [
            "Any source, seal, tool, checkpoint, normalization, world, sequence, target, identity or frozen original metric drift.",
            "Any C10/M-TARE read, optimizer/model update, threshold/checkpoint selection, non-slope replacement, gate failure or resource overrun.",
        ],
        "expected_evidence": [
            "Actual corrective-training source-path audit explicitly proving C09=0 and C10=0 during fit/selection, plus C09 evaluation and C10/M-TARE zero counters.",
            "Ten feature archives, three complete slope outputs/metrics, corrected complete gate, source verification before/after, raw logs, environment, RUN_STATE and SHA-256 seal.",
            "After PASS, a corrected perception/slope-ablation paper bundle retained in PNG/PDF/SVG with source CSV/JSON, generator, provenance and hashes.",
        ],
        "estimated_cost": card["estimated_cost"],
        "frozen_tools": {name: _record(relative) for name, relative in tools.items()},
        "frozen_inputs": {str(path.relative_to(PROJECT_ROOT)): _sha256(path) for path in inputs},
        "expected_counts": {"validation_worlds": 10, "validation_unique_frames": 32678, "validation_sequences": 24462, "seeds": 3, "c10_worlds_read": 0, "mtare_worlds_read": 0, "optimizer_steps": 0, "model_updates": 0},
    }
    write_json(SPEC, spec)
    return {"data_card": str(CARD.relative_to(PROJECT_ROOT)), "data_card_sha256": _sha256(CARD), "spec": str(SPEC.relative_to(PROJECT_ROOT)), "spec_sha256": _sha256(SPEC)}


if __name__ == "__main__":
    print(freeze())
