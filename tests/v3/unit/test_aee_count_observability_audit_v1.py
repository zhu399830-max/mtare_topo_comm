from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT=Path(__file__).resolve().parents[3]; TOOLS=ROOT/"tools/v3"
if str(TOOLS) not in sys.path: sys.path.insert(0,str(TOOLS))
spec=importlib.util.spec_from_file_location("audit",TOOLS/"run_aee_count_observability_audit_v1.py"); assert spec and spec.loader
audit=importlib.util.module_from_spec(spec); spec.loader.exec_module(audit)


def test_matching_is_circular_and_one_flag_per_teacher_exit() -> None:
    assert audit.matched_flags((359.0, 90.0, 200.0), [1.0, 110.0]) == [True, True, False]


def test_matching_tolerance_is_inclusive() -> None:
    assert audit.matched_flags((0.0,), [20.0]) == [True]
    assert audit.matched_flags((0.0,), [20.1]) == [False]
