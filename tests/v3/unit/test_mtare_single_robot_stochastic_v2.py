from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from mtare_topo.evaluation.closed_loop_matrix import enumerate_stochastic_cases, load_stochastic_matrix


ROOT = Path(__file__).resolve().parents[3]


def _load():
    tools = str(ROOT / "tools/v3")
    if tools not in sys.path:
        sys.path.insert(0, tools)
    spec = importlib.util.spec_from_file_location(
        "run_mtare_single_robot_stochastic_v2", ROOT / "tools/v3/run_mtare_single_robot_stochastic_v2.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v2_case_command_uses_host_archive_without_changing_case_identity(tmp_path: Path) -> None:
    runner = _load()
    matrix = load_stochastic_matrix(ROOT / "configs/v3/gate6/mtare_single_robot_stochastic_matrix_v1.json")
    case = enumerate_stochastic_cases(matrix)[0]
    command, name = runner.case_command(tmp_path, case, matrix)
    shell = command[-1]
    assert name == f"mtare-single-v2-{case.index:03d}"
    for token in (case.case_id, f"--runtime-sec {case.runtime_sec}", "--archive-mode host", "--host-uid 1000", "--host-gid 1000"):
        assert token in shell
    assert command[command.index("--hostname") + 1] == "localhost"
    assert "--topic-contract /workspace/configs/v3/gate5/closed_loop_recording_topics_v1.json" in shell


def test_m1d_method_identity_records_adapted_fallback_rate(tmp_path: Path) -> None:
    runner = _load()
    case = tmp_path / "case"
    (case / "planner").mkdir(parents=True)
    snapshot = {
        "checkpoint": {"mode": "M1D_AEE_HEAD_ADAPTED_V1", "source_seed": 1},
        "learned_empty_cycles": 7,
        "fallback_cycles": 7,
        "post_warmup_cycles": 100,
        "post_warmup_fallback_cycles": 4,
    }
    (case / "planner/topology_snapshot.json").write_text(json.dumps(snapshot))
    summary = {"case": {"method_family": "m1d_topology"}, "planner_evidence": {"topology_snapshot": "planner/topology_snapshot.json"}}
    evidence = runner.method_identity_evidence(case, summary)
    assert evidence["post_warmup_fallback_rate"] == 0.04
    assert evidence["checkpoint_mode"] == "M1D_AEE_HEAD_ADAPTED_V1"
    snapshot["checkpoint"]["mode"] = "M1D"
    (case / "planner/topology_snapshot.json").write_text(json.dumps(snapshot))
    with pytest.raises(RuntimeError, match="expected checkpoint"):
        runner.method_identity_evidence(case, summary)


def test_nonlearned_family_has_no_fake_fallback_metric(tmp_path: Path) -> None:
    runner = _load()
    assert runner.method_identity_evidence(tmp_path, {"case": {"method_family": "original_mtare"}}) == {
        "method_family": "original_mtare",
        "semantic_fallback_applicable": False,
    }
