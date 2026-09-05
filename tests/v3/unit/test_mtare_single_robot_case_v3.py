from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def _load():
    tools = str(ROOT / "tools/v3")
    if tools not in sys.path:
        sys.path.insert(0, tools)
    spec = importlib.util.spec_from_file_location(
        "run_mtare_single_robot_case_v3",
        ROOT / "tools/v3/run_mtare_single_robot_case_v3.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v3_case_wrapper_selects_v5_only_for_m1d():
    module = _load()
    args = argparse.Namespace(
        method_family="m1d_topology", checkpoint="checkpoint.pt",
        checkpoint_sha256="a" * 64,
    )
    command = module.method_command(args, Path("/evidence/planner"))
    assert "semantic_topology_global_node_v5.py" in command
    assert "checkpoint.pt" in command


def test_v3_case_wrapper_source_does_not_change_case_contract():
    source = (ROOT / "tools/v3/run_mtare_single_robot_case_v3.py").read_text()
    assert "import run_mtare_single_robot_case_v1 as base" in source
    assert "retry_penalty" not in source
    assert "runtime-sec" not in source
