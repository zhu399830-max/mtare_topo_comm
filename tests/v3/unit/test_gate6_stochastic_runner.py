from __future__ import annotations

import importlib.util
from pathlib import Path

from mtare_topo.evaluation.closed_loop_matrix import (
    enumerate_stochastic_cases,
    load_stochastic_matrix,
)


ROOT = Path(__file__).resolve().parents[3]
MATRIX_PATH = ROOT / "configs/v3/gate6/mtare_single_robot_stochastic_matrix_v1.json"


def load_runner():
    import sys

    tools = ROOT / "tools/v3"
    sys.path.insert(0, str(tools))
    spec = importlib.util.spec_from_file_location(
        "run_mtare_single_robot_stochastic_v1",
        tools / "run_mtare_single_robot_stochastic_v1.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_runner_identity_is_gate6_closed_loop() -> None:
    runner = load_runner()
    source = (ROOT / "tools/v3/run_mtare_single_robot_stochastic_v1.py").read_text(encoding="utf-8")
    assert runner.RUN_ID == "gate6_20260820_mtare_single_robot_stochastic_v1_seed20260820"
    assert 'spec.get("gate") != 6' in source
    assert 'spec.get("operation") != "closed_loop_single"' in source
    assert "approved Gate-5 infrastructure" not in source


def test_every_case_command_binds_schedule_identity_and_no_retry() -> None:
    runner = load_runner()
    matrix = load_stochastic_matrix(MATRIX_PATH)
    run_dir = ROOT / "results/gate6_single_robot/gate6_20260820_mtare_single_robot_stochastic_v1_seed20260820"
    cases = enumerate_stochastic_cases(matrix)
    for case in cases:
        command, container_name = runner.case_command(run_dir, case, matrix)
        rendered = " ".join(command)
        assert container_name.endswith(f"{case.index:03d}")
        assert f"--case-id {case.case_id}" in rendered
        assert f"--block-id {case.block_id}" in rendered
        assert f"--environment-seed {case.environment_seed}" in rendered
        assert f"--runtime-sec {case.runtime_sec}" in rendered
        assert "--execution-repeat " + str(case.execution_repeat) in rendered
        if case.method_family == "m1d_topology":
            assert f"--checkpoint-seed {case.checkpoint_seed}" in rendered
            assert "--checkpoint " in rendered
        else:
            assert "--checkpoint-seed" not in rendered
    source = (ROOT / "tools/v3/run_mtare_single_robot_stochastic_v1.py").read_text(encoding="utf-8")
    assert "retry" not in source.lower()


def test_runner_enforces_serial_cases_disk_floor_and_exact_counts() -> None:
    runner = load_runner()
    source = (ROOT / "tools/v3/run_mtare_single_robot_stochastic_v1.py").read_text(encoding="utf-8")
    assert runner.MINIMUM_FREE_BYTES == 150 * 1024**3
    assert "for case in cases:" in source
    assert "completed_case_count\": 90" in source
    assert "simulated_runtime_sec\": 54000" in source
    assert "analyze_stochastic_cases(summaries)" in source
    assert "c09_reads\": 0" in source
    assert "c10_reads\": 0" in source
