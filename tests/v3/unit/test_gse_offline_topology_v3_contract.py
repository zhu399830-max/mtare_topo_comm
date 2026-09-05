from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from mtare_topo.governance import validate_data_card
from mtare_topo.representation.gse_exit_token_ensemble import FROZEN_ENSEMBLE_THRESHOLD


ROOT = Path(__file__).resolve().parents[3]


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v3_data_card_binds_exact_worlds_samples_and_association_selection_domain() -> None:
    card = json.loads(
        (ROOT / "configs/v3/gate4/data_cards/gse_offline_topology_validation_v3.json").read_text(
            encoding="utf-8"
        )
    )
    report = validate_data_card(card)
    assert report.passed, report.errors
    assert card["status"] == "APPROVED_FOR_ONE_IMMUTABLE_GSE_OFFLINE_TOPOLOGY_VALIDATION_V3"
    assert len(card["worlds"]["validation"]) == 10
    assert all(name.endswith("_C09") for name in card["worlds"]["validation"])
    assert len(card["worlds"]["threshold_calibration"]) == 20
    assert all(name.endswith(("_C07", "_C08")) for name in card["worlds"]["threshold_calibration"])
    assert sum(row["sequence_count"] for row in card["trajectories"]) == 24462
    assert sum(row["directed_traversal_count"] for row in card["trajectories"]) == 2054
    assert card["sampling"]["association_selection_pairs_within_16m"] == 31469
    assert card["sampling"]["association_operating_point"]["threshold"] == FROZEN_ENSEMBLE_THRESHOLD


def test_v3_runner_and_evaluator_freeze_one_method_identity() -> None:
    runner = _load("gse_topology_v3_runner", "tools/v3/run_gse_offline_topology_validation_v3.py")
    evaluator = _load("gse_topology_v3_evaluator", "tools/v3/evaluate_gse_offline_topology_v3.py")
    freezer = _load("gse_topology_v3_freezer", "tools/v3/freeze_gse_offline_topology_spec_v3.py")
    assert runner.RUN_ID == freezer.RUN_ID
    assert runner.PASS_STATUS == evaluator.PASS_STATUS
    assert runner.FAIL_STATUS == evaluator.FAIL_STATUS
    assert evaluator.FROZEN_ENSEMBLE_THRESHOLD == FROZEN_ENSEMBLE_THRESHOLD
    assert "gse_exit_token_ensemble.py" in freezer.TOOLS["ensemble_runtime"]


def test_v3_evaluator_uses_frozen_backend_without_old_similarity_prefilter() -> None:
    graph_source = (ROOT / "src/mtare_topo/topology/gse_graph.py").read_text(encoding="utf-8")
    evaluator_source = (ROOT / "tools/v3/evaluate_gse_offline_topology_v3.py").read_text(encoding="utf-8")
    assert '"frozen_exit_token_ensemble"' in evaluator_source
    frozen_branch = graph_source.split("if self.association_backend is not None:", 2)[2].split(
        "query = _unit(descriptor)", 1
    )[0]
    assert "descriptor_minimum_similarity" not in frozen_branch
    assert "_exit_token_match" not in frozen_branch
    assert "ensemble_score" in frozen_branch


def test_v3_spec_declares_exact_replay_counts_after_freeze() -> None:
    path = ROOT / "configs/v3/gate4/gse_offline_topology_validation_v3.json"
    if not path.exists():
        return
    spec = json.loads(path.read_text(encoding="utf-8"))
    assert spec["expected_counts"]["validation_sequences"] == 24462
    assert spec["expected_counts"]["world_config_seed_replays"] == 24300
    assert spec["expected_counts"]["typed_observation_updates"] == 59442660
    assert spec["expected_counts"]["c10_worlds_read"] == 0
    assert spec["user_authorization"]["status"] == "APPROVED"
