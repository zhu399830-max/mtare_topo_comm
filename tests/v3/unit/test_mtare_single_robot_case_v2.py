from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]


def load_runner():
    tools = str(ROOT / "tools/v3")
    if tools not in sys.path:
        sys.path.insert(0, tools)
    spec = importlib.util.spec_from_file_location(
        "run_mtare_single_robot_case_v2",
        ROOT / "tools/v3/run_mtare_single_robot_case_v2.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def arguments(**overrides):
    values = {
        "method_family": "m1d_topology",
        "checkpoint": "results/checkpoint.pt",
        "checkpoint_sha256": "a" * 64,
        "world": "tunnel",
        "environment_seed": 11,
        "complete_map": None,
        "complete_map_sha256": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_v2_case_runner_selects_only_v4_for_m1d() -> None:
    runner = load_runner()
    command = runner.method_command(arguments(), Path("/case/planner"))
    assert "semantic_topology_global_node_v4.py" in command
    assert "semantic_topology_global_node_v3.py" not in command
    assert "--publish-period-sec 1.0" in command


def test_v2_case_runner_preserves_original_and_oracle_commands() -> None:
    runner = load_runner()
    original = runner.method_command(arguments(method_family="original_mtare"), Path("/case/planner"))
    oracle = runner.method_command(
        arguments(
            method_family="layered_gt_map_oracle",
            complete_map="/map.ply",
            complete_map_sha256="b" * 64,
        ),
        Path("/case/planner"),
    )
    assert "explore_seeded.launch" in original
    assert "layered_gt_map_global_node_v1.py" in oracle


def test_v2_case_runner_dispatch_remains_nonrecursive_while_base_is_patched() -> None:
    runner = load_runner()
    original = runner.base.method_command
    runner.base.method_command = runner.method_command
    try:
        command = runner.method_command(arguments(method_family="original_mtare"), Path("/case/planner"))
    finally:
        runner.base.method_command = original
    assert "explore_seeded.launch" in command


def test_v2_case_runner_rejects_missing_checkpoint_identity() -> None:
    runner = load_runner()
    with pytest.raises(ValueError, match="checkpoint identity"):
        runner.method_command(arguments(checkpoint=None), Path("/case/planner"))
