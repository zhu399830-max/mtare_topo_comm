from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import numpy as np

from mtare_topo.evaluation.gse_topology_replay import rule_graph_parameter_grid
from mtare_topo.governance import validate_data_card


ROOT = Path(__file__).resolve().parents[3]


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_offline_topology_data_card_is_exact_and_governance_valid() -> None:
    card = json.loads(
        (ROOT / "configs/v3/gate4/data_cards/gse_offline_topology_validation_v1.json").read_text(
            encoding="utf-8"
        )
    )
    report = validate_data_card(card)
    assert report.passed, report.errors
    assert len(card["worlds"]["train"]) == 80
    assert len(card["worlds"]["validation"]) == 10
    assert len(card["trajectories"]) == 10
    assert sum(row["sequence_count"] for row in card["trajectories"]) == 24462
    assert sum(row["directed_traversal_count"] for row in card["trajectories"]) == 2054
    assert card["sampling"]["structure_event_counts"] == {
        "corridor": 17451,
        "junction": 3216,
        "terminal": 869,
        "turn": 310,
        "geometry_transition": 2616,
    }


def test_rule_association_ablation_can_preserve_calibrated_uncertainty() -> None:
    grid = rule_graph_parameter_grid(event_probability_threshold=0.43, maximum_uncertainty=0.27)
    assert len(grid) == 243
    assert all(config.event_probability_threshold == 0.43 for config in grid)
    assert all(config.maximum_uncertainty == 0.27 for config in grid)
    assert all(config.descriptor_minimum_similarity == -1.0 for config in grid)


def test_v2_offline_observation_overlay_changes_only_aligned_slope() -> None:
    evaluator = _load("gse_offline_evaluator_v2_overlay", "tools/v3/evaluate_gse_offline_topology_v2.py")
    base = {
        "global_sequence_index": np.asarray([4, 9], dtype=np.int64),
        "slope_deg": np.asarray([8.0, 7.0], dtype=np.float32),
        "width_m": np.asarray([2.0, 3.0], dtype=np.float32),
    }
    corrected = {
        "global_sequence_index": np.asarray([9, 4], dtype=np.int64),
        "corrected_slope_deg": np.asarray([-2.0, 1.0], dtype=np.float32),
    }
    output = evaluator._overlay_risk_calibrated_slope(base, corrected, expected_count=2)
    assert np.array_equal(output["slope_deg"], [1.0, -2.0])
    assert output["width_m"] is base["width_m"]
    assert np.array_equal(base["slope_deg"], [8.0, 7.0])
    with pytest.raises(RuntimeError, match="identity drift"):
        evaluator._overlay_risk_calibrated_slope(
            base,
            {**corrected, "global_sequence_index": np.asarray([9, 8])},
            expected_count=2,
        )


def test_run_sweep_writes_exact_243_rows_replay_count_and_selected_file_set(
    tmp_path: Path,
    monkeypatch,
) -> None:
    evaluator = _load("gse_offline_evaluator_sweep_contract", "tools/v3/evaluate_gse_offline_topology_v1.py")
    calls = 0

    def replay(**_kwargs):
        nonlocal calls
        calls += 1
        return {
            "nodes": [],
            "edges": [],
            "decision_trace": [
                {
                    "evaluator_association_opportunity": True,
                    "evaluator_has_existing_true_node": True,
                }
            ],
            "association_metrics": {
                "merge_attempts": 1,
                "correct_merges": 1,
                "false_merges": 0,
            },
            "topology_metrics": {
                "node_true_positive": 1,
                "predicted_node_count": 1,
                "teacher_node_count": 1,
                "edge_true_positive": 1,
                "predicted_edge_count": 1,
                "teacher_edge_count": 1,
                "connected_component_error": 0,
                "cycle_rank_error": 0,
            },
        }

    monkeypatch.setattr(evaluator, "replay_typed_observation_world", replay)
    grids = {seed: rule_graph_parameter_grid(event_probability_threshold=0.5) for seed in (0, 1)}
    worlds = {
        "C09_world": SimpleNamespace(
            traversal_records=(),
            teacher_observations=(),
            pose_by_sequence_index={},
        )
    }
    result = evaluator._run_sweep(
        method_id="synthetic_method",
        worlds=worlds,
        observations_by_seed={0: {}, 1: {}},
        grids_by_seed=grids,
        output_dir=tmp_path,
        association_mode="rule",
        selector=lambda rows: rows[7],
    )

    assert result["parameter_sweep_rows"] == 243
    assert result["world_seed_replays_in_sweep"] == 486
    assert result["selected_evidence_files"] == 10
    assert calls == 486 + 2
    sweep = tmp_path / "synthetic_method/parameter_sweep.jsonl"
    assert len(sweep.read_text(encoding="utf-8").splitlines()) == 243
    actual = {path.relative_to(tmp_path / "synthetic_method") for path in (tmp_path / "synthetic_method").rglob("*") if path.is_file()}
    assert len(actual) == 12


def test_offline_runner_and_publisher_bind_one_fixed_identity() -> None:
    runner = _load("gse_offline_runner", "tools/v3/run_gse_offline_topology_validation_v1.py")
    publisher = _load("gse_offline_publisher", "tools/v3/publish_gse_offline_topology_figure_v1.py")
    assert runner.RUN_ID == "gate4_20260824_gse_offline_topology_validation_v1_seed0"
    assert publisher.EXPECTED_RUN_ID == runner.RUN_ID
    assert publisher.EXPECTED_STATUS == runner.PASS_STATUS
    assert publisher.METHOD_ORDER == (
        "exit_only_rule_graph",
        "nonlearning_geometry_rule_graph",
        "gse_rule_association_ablation",
        "gse_learned_association",
    )


def test_perception_runner_uses_governance_legal_threshold_calibration_identity() -> None:
    runner = _load("gse_perception_runner", "tools/v3/run_gse_perception_validation_v1.py")
    publisher = _load("gse_perception_publisher", "tools/v3/publish_gse_perception_figure_v1.py")
    assert runner.RUN_ID == "gate3_20260824_gse_perception_validation_v1_seed0"
    assert publisher.EXPECTED_RUN_ID == runner.RUN_ID


def test_spec_freezers_bind_the_same_downstream_run_identities() -> None:
    perception = _load("gse_perception_spec_freezer", "tools/v3/freeze_gse_perception_validation_spec_v1.py")
    topology = _load("gse_topology_spec_freezer", "tools/v3/freeze_gse_offline_topology_spec_v1.py")
    assert perception.RUN_ID == "gate3_20260824_gse_perception_validation_v1_seed0"
    assert topology.RUN_ID == "gate4_20260824_gse_offline_topology_validation_v1_seed0"
    assert topology.PERCEPTION.name == perception.RUN_ID
    assert all((ROOT / relative).is_file() for relative in perception.FROZEN_TOOLS.values())
    assert all((ROOT / relative).is_file() for relative in topology.FROZEN_TOOLS.values())
    assert perception._zero_forbidden_reads({"strict_test_worlds_read": 0, "mtare_worlds_read": 0})
    assert not perception._zero_forbidden_reads({"strict_test_worlds_read": 0, "mtare_worlds_read": 1})
    assert topology._zero_forbidden_reads({"strict_test_worlds_read": 0, "mtare_worlds_read": 0})
    assert not topology._zero_forbidden_reads({"strict_test_worlds_read": 1, "mtare_worlds_read": 0})


def test_offline_runner_rejects_source_with_mtare_read(tmp_path: Path, monkeypatch) -> None:
    runner = _load("gse_offline_runner_forbidden_source", "tools/v3/run_gse_offline_topology_validation_v1.py")
    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)
    source = tmp_path / "results/source"
    (source / "metrics").mkdir(parents=True)
    (source / "RUN_STATE.json").write_text(
        json.dumps({"state": "COMPLETED", "overall_status": "PASS_SOURCE"}), encoding="utf-8"
    )
    (source / "metrics/summary.json").write_text(
        json.dumps(
            {
                "overall_status": "PASS_SOURCE",
                "strict_test_worlds_read": 0,
                "mtare_worlds_read": 1,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="leakage-free PASS"):
        runner._verify_sealed_source(source, "PASS_SOURCE")


def test_offline_publisher_materializes_complete_vector_and_source_bundle(tmp_path: Path, monkeypatch) -> None:
    publisher = _load("gse_offline_publisher_synthetic", "tools/v3/publish_gse_offline_topology_figure_v1.py")
    fake_generator = tmp_path / "tools/v3/publish.py"
    fake_generator.parent.mkdir(parents=True)
    fake_generator.write_text("# frozen synthetic generator identity\n", encoding="utf-8")
    monkeypatch.setattr(publisher, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(publisher, "__file__", str(fake_generator))
    run = tmp_path / "results/gate4_topology/gate4_20260824_gse_offline_topology_validation_v1_seed0"
    topology = run / "artifacts/offline_topology"
    topology.mkdir(parents=True)
    (run / "metrics").mkdir()
    state = {"state": "COMPLETED", "overall_status": publisher.EXPECTED_STATUS}
    runner_summary = {"overall_status": publisher.EXPECTED_STATUS}

    def aggregate(offset: float) -> dict:
        return {
            "world_seed_replays": 30,
            "node": {"f1": 0.60 + offset},
            "edge": {"f1": 0.55 + offset},
            "association": {"precision": 0.985, "false_loop_merge_rate": 0.005},
            "invariants": {
                "connected_component_mean_signed_error": 0.10,
                "cycle_rank_mean_signed_error": -0.10,
            },
        }

    methods = {}
    for index, method in enumerate(publisher.METHOD_ORDER):
        method_dir = topology / method
        method_dir.mkdir()
        summary = {
            "selected_grid_index": index,
            "selected_aggregate": aggregate(index * 0.03),
        }
        methods[method] = summary
        (method_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    source = {
        "overall_status": publisher.EXPECTED_STATUS,
        "scientific_gate": {"passed": True},
        "strict_test_worlds_read": 0,
        "mtare_worlds_read": 0,
        "methods": methods,
    }
    files = {
        run / "RUN_STATE.json": state,
        run / "metrics/summary.json": runner_summary,
        topology / "summary.json": source,
    }
    for path, value in files.items():
        path.write_text(json.dumps(value), encoding="utf-8")
    seal = run / "artifacts/evidence_sha256.txt"
    evidence = list(files) + [topology / method / "summary.json" for method in publisher.METHOD_ORDER]
    seal.write_text(
        "".join(
            f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(tmp_path)}\n"
            for path in evidence
        ),
        encoding="utf-8",
    )
    destination = tmp_path / "docs/figures/gse_graph"
    result = publisher.publish(run, destination)
    assert result["published_files"] == 7
    for suffix in (".png", ".pdf", ".svg", ".csv", "_source.json", "_provenance.json", "_sha256.txt"):
        name = f"gse_offline_topology{suffix}"
        assert (destination / name).is_file()
