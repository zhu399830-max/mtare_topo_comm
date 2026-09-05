#!/usr/bin/env python3
"""Freeze the exact formal perception-validation spec after training PASS."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, validate_data_card, write_json


TRAINING = PROJECT_ROOT / "results/gate2_representation/gate2_20260824_gse_graph_three_seed_training_v1r_seed0"
DATASET = PROJECT_ROOT / "results/gate2_representation/gate2_20260824_gse_deduplicated_dataset_export_v1_seed0"
CARD = PROJECT_ROOT / "configs/v3/gate3/data_cards/gse_perception_validation_v1.json"
OUTPUT = PROJECT_ROOT / "configs/v3/gate3/gse_perception_validation_v1.json"
RUN_ID = "gate3_20260824_gse_perception_validation_v1_seed0"
PYTHON = "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python"
FROZEN_TOOLS = {
    "data_card": "configs/v3/gate3/data_cards/gse_perception_validation_v1.json",
    "runner": "tools/v3/run_gse_perception_validation_v1.py",
    "exit_only_evaluator": "tools/v3/evaluate_gse_exit_only_baseline_v1.py",
    "nonlearning_geometry_evaluator": "tools/v3/evaluate_gse_nonlearning_geometry_v1.py",
    "validation_calibration": "tools/v3/evaluate_gse_validation_outputs_v1.py",
    "perception_gate": "tools/v3/summarize_gse_perception_gate_v1.py",
    "paper_publisher": "tools/v3/publish_gse_perception_figure_v1.py",
    "rejection_analysis_publisher": "tools/v3/publish_gse_rejection_analysis_figure_v1.py",
    "uncertainty_analysis_publisher": "tools/v3/publish_gse_uncertainty_analysis_figure_v1.py",
    "per_world_perception_publisher": "tools/v3/publish_gse_per_world_perception_figure_v1.py",
    "exit_token_validation_publisher": "tools/v3/publish_gse_exit_token_validation_figure_v1.py",
    "event_class_table_publisher": "tools/v3/publish_gse_event_class_table_v1.py",
    "complete_evidence_publisher": "tools/v3/publish_gse_perception_evidence_v1.py",
    "dataset_reader": "src/mtare_topo/data/gse_training_dataset.py",
    "historical_model": "src/mtare_topo/representation/phase3_structural_semantics.py",
    "gse_model_contract": "src/mtare_topo/representation/gse_graph.py",
    "gse_semantic_types": "src/mtare_topo/semantics/geometric_semantics.py",
    "nonlearning_geometry": "src/mtare_topo/semantics/range_geometry_baseline.py",
    "calibration_primitives": "src/mtare_topo/evaluation/gse_validation_calibration.py",
    "validation_evaluator": "src/mtare_topo/evaluation/gse_validation_evaluator.py",
    "source_evidence_integrity": "src/mtare_topo/evaluation/gse_evidence_integrity.py",
    "exit_token_metrics": "src/mtare_topo/evaluation/gse_exit_token_metrics.py",
    "shared_direction_metrics": "src/mtare_topo/evaluation/phase3_semantic_metrics.py",
    "gate_logic": "src/mtare_topo/evaluation/gse_perception_gate.py",
    "governance": "src/mtare_topo/governance.py",
    "preflight": "tools/v3/preflight.py",
}


def _zero_forbidden_reads(summary: dict) -> bool:
    return summary.get("strict_test_worlds_read") == 0 and summary.get("mtare_worlds_read") == 0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _record(relative: str) -> dict[str, str]:
    return {"path": relative, "sha256": _sha256(PROJECT_ROOT / relative)}


def freeze(output: Path = OUTPUT) -> dict:
    output = output.resolve()
    output.relative_to(PROJECT_ROOT.resolve())
    if output.exists():
        raise RuntimeError("perception run spec already exists; refusing overwrite")
    required_source_files = (
        TRAINING / "RUN_STATE.json",
        TRAINING / "metrics/summary.json",
        TRAINING / "artifacts/evidence_sha256.txt",
        DATASET / "RUN_STATE.json",
        DATASET / "metrics/summary.json",
        DATASET / "artifacts/evidence_sha256.txt",
    )
    if not all(path.is_file() for path in required_source_files):
        raise RuntimeError("cannot freeze perception spec before training and dataset evidence is complete")
    training_state = load_json(TRAINING / "RUN_STATE.json")
    training_summary = load_json(TRAINING / "metrics/summary.json")
    dataset_state = load_json(DATASET / "RUN_STATE.json")
    dataset_summary = load_json(DATASET / "metrics/summary.json")
    if (
        training_state.get("state") != "COMPLETED"
        or training_state.get("overall_status") != "PASS_GSE_GRAPH_THREE_SEED_TRAINING_V1R"
        or training_summary.get("overall_status") != "PASS_GSE_GRAPH_THREE_SEED_TRAINING_V1R"
        or dataset_state.get("state") != "COMPLETED"
        or dataset_state.get("overall_status") != "PASS_GSE_DEDUPLICATED_DATASET_EXPORT_V1"
        or dataset_summary.get("overall_status") != "PASS_GSE_DEDUPLICATED_DATASET_EXPORT_V1"
        or not _zero_forbidden_reads(training_summary)
        or not _zero_forbidden_reads(dataset_summary)
    ):
        raise RuntimeError("cannot freeze perception spec before training and dataset are sealed PASS")
    card = load_json(CARD)
    report = validate_data_card(card)
    if not report.passed or card.get("status") != "APPROVED_FOR_ONE_IMMUTABLE_GSE_PERCEPTION_VALIDATION":
        raise RuntimeError(f"perception Data Card is invalid: {report.errors}")
    tools = FROZEN_TOOLS
    input_paths = [
        TRAINING / "RUN_STATE.json",
        TRAINING / "metrics/summary.json",
        TRAINING / "artifacts/evidence_sha256.txt",
        DATASET / "RUN_STATE.json",
        DATASET / "metrics/summary.json",
        DATASET / "artifacts/evidence_sha256.txt",
    ]
    checkpoints = (
        "results/gate2_representation/gate2_20260812_cano_phase3_masking_corrective_ray_dropout_v1r3_seed0/artifacts/models/m1d_seed0/best.pt",
        "results/gate2_representation/gate2_20260812_cano_phase3_masking_corrective_ray_dropout_v1r3_seed0/artifacts/models/m1d_seed1/best.pt",
        "results/gate2_representation/gate2_20260813_cano_phase3_masking_corrective_recovery_seed2_v1_seed2/artifacts/models/m1d_seed2/best.pt",
    )
    input_paths.extend(PROJECT_ROOT / relative for relative in checkpoints)
    spec = {
        "schema_version": "v3_run_spec_v1",
        "gate": 3,
        "execution_phase": 3,
        "date": "20260824",
        "slug": "gse_perception_validation_v1",
        "operation": "threshold_calibration",
        "question": "Do the three frozen GSE seeds significantly outperform the frozen exit-only model and deterministic geometry estimator while satisfying non-vacuous association safety on the ten C09 validation parents?",
        "method": "Evaluate all 24,462 validation sequences, fit only pre-registered validation temperatures/rejection/association thresholds, apply the fixed event/geometry/association gates, and retain complete curves without model updates.",
        "baseline": "Historical frozen M1D seeds 0/1/2 on only the current fifth frame and a deterministic current-frame range geometry estimator; no checkpoint or baseline threshold selection is changed.",
        "user_authorization": {
            "status": "APPROVED",
            "approved_by": "user",
            "approved_at": "2026-08-24T05:28:00+08:00",
            "scope": "One immutable GSE C09 validation-only perception/calibration run after training PASS, with zero C10/M-TARE reads or model updates.",
            "confirmation_reference": "The user granted standing authorization for the fixed GSE-Graph paper program and asked not to request repeated approvals.",
        },
        "seed": 0,
        "data_card": str(CARD.relative_to(PROJECT_ROOT)),
        "config_path": str(CARD.relative_to(PROJECT_ROOT)),
        "working_directory": str(PROJECT_ROOT),
        "command": [
            "/usr/bin/systemd-inhibit",
            "--what=sleep:shutdown",
            "--why=GSE perception validation",
            "--mode=block",
            "/usr/bin/timeout",
            "--signal=INT",
            "--kill-after=300s",
            "7800s",
            PYTHON,
            "tools/v3/run_gse_perception_validation_v1.py",
            "--spec",
            str(output),
            "--run-dir",
            str(PROJECT_ROOT / "results/gate3_semantics" / RUN_ID),
        ],
        "acceptance_criteria": [
            "All exact 10 C09 validation parents, 24,462 sequences and three frozen seeds are evaluated with zero model updates and zero C10/M-TARE reads.",
            "Mean event macro-F1 gain is at least 0.05, at least two seeds gain 0.05 and no seed regresses against paired exit-only M1D.",
            "Four-field relative geometry MAE improvement averages at least 0.10 and no field regresses by more than 0.05.",
            "Every seed has nonzero place and exit association acceptance, precision at least 0.98 and false-accept rate at most 0.01.",
        ],
        "expected_evidence": [
            "Complete three-seed M1D outputs, deterministic geometry predictions, calibration/rejection curves and per-seed summaries.",
            "One pre-registered perception gate summary, raw logs, environment, RUN_STATE and SHA-256 seal.",
            "After sealed PASS, six atomically published paper evidence bundles plus one aggregate index: aggregate and all-parent diagnostics in PNG/PDF/SVG/CSV/Markdown/LaTeX/JSON/provenance and SHA-256 manifests.",
        ],
        "estimated_cost": {
            "disk_gb": 2.0,
            "wall_time_hours": 2.0,
            "compute": "One RTX 5090 for 73,386 frozen M1D current-frame inferences plus CPU calibration/non-learning geometry; zero GSE training.",
        },
        "frozen_tools": {name: _record(relative) for name, relative in tools.items()},
        "frozen_inputs": {
            str(path.relative_to(PROJECT_ROOT)): _sha256(path)
            for path in input_paths
        },
        "expected_counts": {
            "validation_worlds": 10,
            "validation_sequences": 24462,
            "validation_unique_frames": 32678,
            "directed_traversals": 2054,
            "seeds": 3,
            "strict_test_worlds_read": 0,
            "mtare_worlds_read": 0,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, spec)
    return {"spec": str(output.relative_to(PROJECT_ROOT)), "frozen_tools": len(tools), "frozen_inputs": len(input_paths)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    print(freeze(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
