#!/usr/bin/env python3
"""Freeze the one-time C09 distance-aware ensemble topology run spec."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from _bootstrap import PROJECT_ROOT


RUN_ID = "gate4_20260826_gse_offline_topology_validation_v3_seed0"
SPEC = PROJECT_ROOT / "configs/v3/gate4/gse_offline_topology_validation_v3.json"
CARD = "configs/v3/gate4/data_cards/gse_offline_topology_validation_v3.json"
SOURCE_RUNS = (
    "results/gate2_representation/gate2_20260824_gse_graph_three_seed_training_v1r_seed0",
    "results/gate3_semantics/gate3_20260824_gse_perception_validation_v1_seed0",
    "results/gate3_semantics/gate3_20260825_gse_risk_calibrated_perception_validation_v2_seed0",
    "results/gate2_representation/gate2_20260824_gse_deduplicated_dataset_export_v1_seed0",
    "results/gate2_representation/gate2_20260824_gse_teacher_manifest_v1_seed0",
    "results/gate3_semantics/gate3_20260826_gse_exit_token_association_corrective_v2_seed0",
    "results/gate3_semantics/gate3_20260826_gse_distance_aware_ensemble_calibration_v1_seed0",
)
TOOLS = {
    "data_card": CARD,
    "evaluator": "tools/v3/evaluate_gse_offline_topology_v3.py",
    "legacy_evaluator_helpers": "tools/v3/evaluate_gse_offline_topology_v2.py",
    "runner": "tools/v3/run_gse_offline_topology_validation_v3.py",
    "ensemble_runtime": "src/mtare_topo/representation/gse_exit_token_ensemble.py",
    "exit_token_verifier": "src/mtare_topo/representation/gse_exit_token_association.py",
    "open_set_features": "src/mtare_topo/representation/gse_open_set_association.py",
    "learned_graph": "src/mtare_topo/topology/gse_graph.py",
    "rule_graph": "src/mtare_topo/topology/gse_rule_graph.py",
    "topology_replay": "src/mtare_topo/evaluation/gse_topology_replay.py",
    "topology_metrics": "src/mtare_topo/evaluation/gse_metrics.py",
    "gse_adapter": "src/mtare_topo/semantics/gse_observation_adapter.py",
    "exit_only_adapter": "src/mtare_topo/semantics/exit_only_geometry_observation.py",
    "nonlearning_adapter": "src/mtare_topo/semantics/nonlearning_geometry_observation.py",
    "replay_dataset": "src/mtare_topo/data/gse_replay_dataset.py",
    "training_dataset": "src/mtare_topo/data/gse_training_dataset.py",
    "typed_observation": "src/mtare_topo/semantics/geometric_semantics.py",
    "source_evidence_integrity": "src/mtare_topo/evaluation/gse_evidence_integrity.py",
    "governance": "src/mtare_topo/governance.py",
    "preflight": "tools/v3/preflight.py",
    "create_run": "tools/v3/create_run.py",
}


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    if SPEC.exists():
        raise RuntimeError("C09 V3 run spec already exists")
    frozen_inputs = {}
    for run in SOURCE_RUNS:
        for relative in ("RUN_STATE.json", "artifacts/evidence_sha256.txt", "metrics/summary.json"):
            path = PROJECT_ROOT / run / relative
            frozen_inputs[str(path.relative_to(PROJECT_ROOT))] = _sha(path)
    ensemble_summary = PROJECT_ROOT / SOURCE_RUNS[-1] / "artifacts/calibration/summary.json"
    frozen_inputs[str(ensemble_summary.relative_to(PROJECT_ROOT))] = _sha(ensemble_summary)
    verifier = PROJECT_ROOT / SOURCE_RUNS[-2] / "artifacts/models"
    for seed in (0, 1, 2):
        for name in ("best.pt", "normalization.npz"):
            path = verifier / f"seed{seed}" / name
            frozen_inputs[str(path.relative_to(PROJECT_ROOT))] = _sha(path)
    frozen_tools = {
        name: {"path": relative, "sha256": _sha(PROJECT_ROOT / relative)}
        for name, relative in TOOLS.items()
    }
    run_dir = PROJECT_ROOT / "results/gate4_topology" / RUN_ID
    command = [
        "/usr/bin/systemd-inhibit", "--what=sleep:shutdown",
        "--why=GSE C09 frozen ensemble topology validation", "--mode=block",
        "/usr/bin/timeout", "--signal=INT", "--kill-after=300s", "29400s",
        "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python",
        "tools/v3/run_gse_offline_topology_validation_v3.py",
        "--spec", str(SPEC), "--run-dir", str(run_dir),
    ]
    spec = {
        "schema_version": "v3_run_spec_v1",
        "slug": "gse_offline_topology_validation_v3",
        "date": "20260826",
        "gate": 4,
        "execution_phase": 3,
        "operation": "topology_replay",
        "question": "Does the frozen distance-aware three-seed exit-token ensemble make GSE semantic nodes and trace-verified edges at least five F1 points better than the strongest deployable baseline on all ten C09 worlds?",
        "method": "Use frozen risk-calibrated GSE observations for event/geometry prediction and the frozen equal-weight V2 exit-token verifier ensemble for node association. Pair scores are accepted only within 16 m at threshold 0.9431912302970886; the unchanged predeclared 243 graph grid is selected on C09. Edges require physical traversal.",
        "baseline": "GSE observations plus rule association, frozen exit-only plus rule graph, deterministic non-learning geometry plus rule graph; GT-TNG remains evaluator-only diagnostic.",
        "data_card": CARD,
        "config_path": CARD,
        "seed": 0,
        "working_directory": str(PROJECT_ROOT),
        "estimated_cost": {
            "compute": "Three frozen verifier inferences per unique <=16 m C09 candidate pair plus 59,442,660 typed graph updates; zero optimizer/model updates.",
            "wall_time_hours": 8.0,
            "disk_gb": 5.0,
        },
        "expected_counts": {
            "validation_worlds": 10,
            "directed_traversals": 2054,
            "validation_sequences": 24462,
            "parameter_groups_per_method": 243,
            "world_config_seed_replays": 24300,
            "typed_observation_updates": 59442660,
            "strict_test_worlds_read": 0,
            "c10_worlds_read": 0,
            "mtare_worlds_read": 0,
            "optimizer_steps": 0,
            "model_updates": 0,
        },
        "acceptance_criteria": [
            "All ten C09 worlds, 2,054 directed traversals and 24,462 sequences are consumed once per selected world/seed replay; all four methods retain exactly 243 graph settings.",
            "GSE node F1 and edge F1 each exceed the strongest corresponding deployable baseline by at least 0.05.",
            "GSE merge attempts are nonzero, association precision is at least 0.98 and false-loop merge rate is at most 0.01.",
            "Absolute mean signed connected-component and cycle-rank errors are each at most 0.25; C10/M-TARE reads and model updates remain zero.",
        ],
        "expected_evidence": [
            "Complete 243-row sweeps and unique selected graph configuration for all four methods.",
            "Per selected world/seed nodes, trace-verified edges, decision traces and frozen verifier candidate decisions.",
            "Topology/association metrics, invariant biases, candidate-pair counts, environment, source-before/after verification, raw log, RUN_STATE and SHA-256 seal.",
        ],
        "frozen_inputs": frozen_inputs,
        "frozen_tools": frozen_tools,
        "command": command,
        "user_authorization": {
            "status": "APPROVED",
            "approved_by": "user",
            "approved_at": "2026-08-26T00:00:00+08:00",
            "scope": "One immutable C09 V3 topology validation with the frozen association operating point and unchanged comparison contract.",
            "confirmation_reference": "Standing authorization for the full GSE-Graph paper workflow; user explicitly instructed autonomous optimal execution without repeated approval requests.",
        },
    }
    SPEC.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(SPEC.relative_to(PROJECT_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
