from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from mtare_topo.evaluation.closed_loop_matrix import (
    enumerate_stochastic_cases,
    load_stochastic_matrix,
)


ROOT = Path(__file__).resolve().parents[3]


def _load():
    tools = str(ROOT / "tools/v3")
    if tools not in sys.path:
        sys.path.insert(0, tools)
    spec = importlib.util.spec_from_file_location(
        "run_aee_composite_v9_reanchor_stochastic_v1",
        ROOT / "tools/v3/run_aee_composite_v9_reanchor_stochastic_v1.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _matrix_and_cases():
    matrix = load_stochastic_matrix(
        ROOT / "configs/v3/gate6/mtare_single_robot_stochastic_matrix_v3.json"
    )
    return matrix, list(enumerate_stochastic_cases(matrix))


def test_v4_selection_preserves_exact_thirty_m1d_cases_and_ten_blocks() -> None:
    runner = _load()
    _, cases = _matrix_and_cases()
    selected = runner.select_v4_cases(cases)
    assert len(selected) == 30
    assert [case.index for case in selected] == [
        case.index for case in cases if case.method_family == "m1d_topology"
    ]
    assert {case.checkpoint_seed for case in selected} == {0, 1, 2}
    assert len({case.block_id for case in selected}) == 10


def test_v4_case_command_uses_reanchor_case_wrapper(tmp_path: Path) -> None:
    runner = _load()
    matrix, cases = _matrix_and_cases()
    case = runner.select_v4_cases(cases)[0]
    command, name = runner.case_command(tmp_path, case, matrix)
    assert name.startswith("mtare-v9-v4-")
    assert command[command.index("--name") + 1] == name
    assert "/workspace/tools/v3/run_mtare_single_robot_case_v2.py" in command[-1]
    assert "/workspace/tools/v3/run_mtare_single_robot_case_v1.py" not in command[-1]
    assert case.case_id in command[-1]


def test_v4_schedule_is_bound_to_exact_sealed_source(tmp_path: Path) -> None:
    runner = _load()
    _, all_cases = _matrix_and_cases()
    selected = runner.select_v4_cases(all_cases)
    source = tmp_path / "source"
    (source / "config").mkdir(parents=True)
    (source / "metrics").mkdir()
    (source / "artifacts").mkdir()
    state_path = source / "RUN_STATE.json"
    summary_path = source / "metrics/summary.json"
    schedule_path = source / "config/case_schedule.json"
    state_path.write_text(
        json.dumps({"state": "FAILED", "overall_status": "FAIL_EXPECTED"}) + "\n",
        encoding="utf-8",
    )
    summary_path.write_text(json.dumps({"completed_case_count": 90}) + "\n", encoding="utf-8")
    schedule_path.write_text(
        json.dumps({"cases": [case.to_dict() for case in all_cases]}) + "\n",
        encoding="utf-8",
    )
    seal_path = source / "artifacts/evidence_sha256.txt"
    bound = (state_path, summary_path, schedule_path)
    seal_path.write_text(
        "".join(
            f"{runner.matrix_base.sha256(path)}  {path.relative_to(tmp_path).as_posix()}\n"
            for path in bound
        ),
        encoding="utf-8",
    )
    spec = {
        "paired_source_run": "source",
        "paired_source_status": "FAIL_EXPECTED",
        "paired_source_seal_sha256": runner.matrix_base.sha256(seal_path),
        "paired_source_schedule_file_sha256": runner.matrix_base.sha256(schedule_path),
        "paired_source_schedule_content_sha256": runner.schedule_content_sha256(
            [case.to_dict() for case in all_cases]
        ),
    }
    evidence = runner.validate_paired_source_schedule(spec, selected, project_root=tmp_path)
    assert evidence["matched_m1d_case_count"] == 30
    assert evidence["exact_order_and_identity_match"] is True
    assert evidence["source_schedule_file_sha256"] == runner.matrix_base.sha256(
        schedule_path
    )
