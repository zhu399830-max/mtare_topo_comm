#!/usr/bin/env python3
"""Freeze offline topology V2 with risk-calibrated perception inputs."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from _bootstrap import PROJECT_ROOT
from mtare_topo.evaluation.gse_evidence_integrity import (
    verify_complete_run_seal,
    verify_failed_component_run_seal,
)
from mtare_topo.governance import load_json, validate_data_card, write_json


TRAINING = PROJECT_ROOT / "results/gate2_representation/gate2_20260824_gse_graph_three_seed_training_v1r_seed0"
PERCEPTION_COMPONENT = PROJECT_ROOT / "results/gate3_semantics/gate3_20260824_gse_perception_validation_v1_seed0"
CORRECTED_PERCEPTION = PROJECT_ROOT / "results/gate3_semantics/gate3_20260825_gse_risk_calibrated_perception_validation_v2_seed0"
DATASET = PROJECT_ROOT / "results/gate2_representation/gate2_20260824_gse_deduplicated_dataset_export_v1_seed0"
TEACHER = PROJECT_ROOT / "results/gate2_representation/gate2_20260824_gse_teacher_manifest_v1_seed0"
CARD = PROJECT_ROOT / "configs/v3/gate4/data_cards/gse_offline_topology_validation_v2.json"
OUTPUT = PROJECT_ROOT / "configs/v3/gate4/gse_offline_topology_validation_v2.json"
RUN_ID = "gate4_20260825_gse_offline_topology_validation_v2_seed0"
PYTHON = "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python"
SOURCE_STATUS = {
    TRAINING: "PASS_GSE_GRAPH_THREE_SEED_TRAINING_V1R",
    CORRECTED_PERCEPTION: "PASS_GSE_RISK_CALIBRATED_PERCEPTION_VALIDATION_V2",
    DATASET: "PASS_GSE_DEDUPLICATED_DATASET_EXPORT_V1",
    TEACHER: "PASS_GSE_TEACHER_MANIFEST_V1",
}
FROZEN_TOOLS = {
    "data_card": "configs/v3/gate4/data_cards/gse_offline_topology_validation_v2.json",
    "runner": "tools/v3/run_gse_offline_topology_validation_v2.py",
    "evaluator": "tools/v3/evaluate_gse_offline_topology_v2.py",
    "paper_publisher": "tools/v3/publish_gse_offline_topology_figure_v2.py",
    "topology_example_publisher": "tools/v3/publish_gse_topology_examples_v2.py",
    "complete_evidence_publisher": "tools/v3/publish_gse_offline_topology_evidence_v2.py",
    "replay_dataset": "src/mtare_topo/data/gse_replay_dataset.py",
    "training_dataset": "src/mtare_topo/data/gse_training_dataset.py",
    "topology_replay": "src/mtare_topo/evaluation/gse_topology_replay.py",
    "topology_metrics": "src/mtare_topo/evaluation/gse_metrics.py",
    "source_evidence_integrity": "src/mtare_topo/evaluation/gse_evidence_integrity.py",
    "learned_graph": "src/mtare_topo/topology/gse_graph.py",
    "rule_graph": "src/mtare_topo/topology/gse_rule_graph.py",
    "gse_adapter": "src/mtare_topo/semantics/gse_observation_adapter.py",
    "exit_only_adapter": "src/mtare_topo/semantics/exit_only_geometry_observation.py",
    "nonlearning_adapter": "src/mtare_topo/semantics/nonlearning_geometry_observation.py",
    "typed_observation": "src/mtare_topo/semantics/geometric_semantics.py",
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
        raise RuntimeError("offline topology run spec already exists; refusing overwrite")
    source_files = []
    for run, expected in SOURCE_STATUS.items():
        required = (run / "RUN_STATE.json", run / "metrics/summary.json", run / "artifacts/evidence_sha256.txt")
        if not all(path.is_file() for path in required):
            raise RuntimeError("cannot freeze offline topology spec before every source evidence bundle is complete")
        verify_complete_run_seal(PROJECT_ROOT, run, expected)
        source_files.extend(required)
    verify_failed_component_run_seal(
        PROJECT_ROOT, PERCEPTION_COMPONENT, "FAIL_GSE_PERCEPTION_VALIDATION_V1"
    )
    source_files.extend(
        (
            PERCEPTION_COMPONENT / "RUN_STATE.json",
            PERCEPTION_COMPONENT / "metrics/summary.json",
            PERCEPTION_COMPONENT / "artifacts/evidence_sha256.txt",
        )
    )
    card = load_json(CARD)
    report = validate_data_card(card)
    if not report.passed or card.get("status") != "APPROVED_FOR_ONE_IMMUTABLE_GSE_OFFLINE_TOPOLOGY_VALIDATION_V2":
        raise RuntimeError(f"offline topology Data Card is invalid: {report.errors}")
    tools = FROZEN_TOOLS
    spec = {
        "schema_version": "v3_run_spec_v1",
        "gate": 4,
        "execution_phase": 3,
        "date": "20260825",
        "slug": "gse_offline_topology_validation_v2",
        "operation": "topology_replay",
        "question": "Do risk-calibrated GSE semantic observations produce more accurate causal structural nodes and trace-verified edges than exit-only and deterministic geometry rule graphs on all ten C09 development-validation parents?",
        "method": "Overlay only the sealed risk-calibrated slope onto identity-aligned frozen GSE outputs, then replay every directed traversal in deterministic Euler order for four fixed methods over the same 243 structural settings, retaining all sweeps and applying the unchanged node/edge, association and invariant gates.",
        "baseline": "Frozen M1D exit-only observations from only the fifth frame plus rule association, and deterministic non-learning geometry events from the same five-frame causal history as GSE plus rule association; GSE plus rule association is a required ablation and GT-TNG is diagnostic only.",
        "user_authorization": {
            "status": "APPROVED",
            "approved_by": "user",
            "approved_at": "2026-08-25T00:35:00+08:00",
            "scope": "One immutable 10-world C09 offline topology V2 validation after risk-calibrated perception PASS, with fixed methods/grid and zero model updates, C10 or M-TARE reads.",
            "confirmation_reference": "The user granted standing authorization for the fixed GSE-Graph paper program and asked not to request repeated approvals.",
        },
        "seed": 0,
        "data_card": str(CARD.relative_to(PROJECT_ROOT)),
        "config_path": str(CARD.relative_to(PROJECT_ROOT)),
        "working_directory": str(PROJECT_ROOT),
        "command": [
            "/usr/bin/systemd-inhibit",
            "--what=sleep:shutdown",
            "--why=GSE offline topology validation",
            "--mode=block",
            "/usr/bin/timeout",
            "--signal=INT",
            "--kill-after=300s",
            "29400s",
            PYTHON,
            "tools/v3/run_gse_offline_topology_validation_v2.py",
            "--spec",
            str(output),
            "--run-dir",
            str(PROJECT_ROOT / "results/gate4_topology" / RUN_ID),
        ],
        "acceptance_criteria": [
            "All ten C09 worlds, 2,054 directed traversals and 24,462 sequence observations are consumed once per selected world/seed replay; all four methods use exactly 243 structural configurations.",
            "GSE node F1 and edge F1 each exceed the stronger corresponding deployable baseline by at least 0.05.",
            "GSE association attempts are nonzero with precision at least 0.98 and false-loop rate at most 0.01.",
            "Absolute mean signed connected-component and cycle-rank errors are each at most 0.25; C10/M-TARE reads and model updates remain zero.",
        ],
        "expected_evidence": [
            "Complete parameter sweeps and unique selected configuration for all four methods.",
            "Per selected world/seed nodes, execution-verified edges, decision traces, association metrics and topology invariants plus GT-TNG diagnostic.",
            "Raw log, environment, source-before/after verification, RUN_STATE and SHA-256 seal; after PASS, complete paper PNG/PDF/SVG/CSV/JSON/provenance bundle.",
        ],
        "estimated_cost": {
            "disk_gb": 5.0,
            "wall_time_hours": 8.0,
            "compute": "CPU-only replay of 59,442,660 typed observation updates across 24,300 world/config/seed runs; zero inference or training.",
        },
        "frozen_tools": {name: _record(relative) for name, relative in tools.items()},
        "frozen_inputs": {
            str(path.relative_to(PROJECT_ROOT)): _sha256(path)
            for path in source_files
        },
        "expected_counts": {
            "validation_worlds": 10,
            "validation_sequences": 24462,
            "directed_traversals": 2054,
            "parameter_groups_per_method": 243,
            "world_config_seed_replays": 24300,
            "typed_observation_updates": 59442660,
            "strict_test_worlds_read": 0,
            "c10_worlds_read": 0,
            "mtare_worlds_read": 0,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, spec)
    return {"spec": str(output.relative_to(PROJECT_ROOT)), "frozen_tools": len(tools), "frozen_inputs": len(source_files)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    print(freeze(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
