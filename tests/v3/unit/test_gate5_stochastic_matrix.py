from __future__ import annotations

from collections import Counter
from pathlib import Path

from mtare_topo.evaluation.closed_loop_matrix import (
    enumerate_stochastic_cases,
    load_stochastic_matrix,
    stochastic_case_audit,
)


ROOT = Path(__file__).resolve().parents[3]
MATRIX = ROOT / "configs/v3/gate6/mtare_single_robot_stochastic_matrix_v1.json"


def test_stochastic_matrix_has_exact_approved_units() -> None:
    matrix = load_stochastic_matrix(MATRIX)
    cases = enumerate_stochastic_cases(matrix)
    audit = stochastic_case_audit(cases)
    assert audit["case_count"] == 90
    assert audit["block_count"] == 10
    assert audit["cases_per_block"] == 9
    assert audit["family_case_counts"] == {
        "original_mtare": 30,
        "m1d_topology": 30,
        "layered_gt_map_oracle": 30,
    }
    assert audit["all_blocks_complete"]
    assert audit["total_simulated_runtime_sec"] == 54000


def test_stochastic_schedule_is_deterministic_unique_and_randomized() -> None:
    matrix = load_stochastic_matrix(MATRIX)
    first = enumerate_stochastic_cases(matrix)
    second = enumerate_stochastic_cases(matrix)
    assert [case.to_dict() for case in first] == [case.to_dict() for case in second]
    assert len({case.case_id for case in first}) == 90
    families = [case.method_family for case in first]
    assert families != sorted(families)
    assert len({case.block_id for case in first[:9]}) > 1


def test_every_block_balances_family_replicates_without_inflating_worlds() -> None:
    cases = enumerate_stochastic_cases(load_stochastic_matrix(MATRIX))
    for block_id in sorted({case.block_id for case in cases}):
        block = [case for case in cases if case.block_id == block_id]
        assert Counter(case.method_family for case in block) == {
            "original_mtare": 3,
            "m1d_topology": 3,
            "layered_gt_map_oracle": 3,
        }
        assert sorted(case.execution_repeat for case in block if case.method_family == "original_mtare") == [0, 1, 2]
        assert sorted(case.checkpoint_seed for case in block if case.method_family == "m1d_topology") == [0, 1, 2]
        assert sorted(case.execution_repeat for case in block if case.method_family == "layered_gt_map_oracle") == [0, 1, 2]


def test_stochastic_matrix_excludes_c09_c10_and_freezes_checkpoints() -> None:
    matrix = load_stochastic_matrix(MATRIX)
    cases = enumerate_stochastic_cases(matrix)
    assert not any("c09" in case.case_id.lower() or "c10" in case.case_id.lower() for case in cases)
    for checkpoint in matrix["m1d_checkpoints"]:
        path = ROOT / checkpoint["path"]
        assert path.is_file()
        import hashlib
        assert hashlib.sha256(path.read_bytes()).hexdigest() == checkpoint["sha256"]
