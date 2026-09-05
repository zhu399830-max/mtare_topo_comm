from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def test_v1r2_binds_existing_v1_case_executor():
    tools = str(ROOT / "tools/v3")
    if tools not in sys.path:
        sys.path.insert(0, tools)
    spec = importlib.util.spec_from_file_location("v1r2", ROOT / "tools/v3/run_aee_composite_v9_combined_correction_stochastic_v1r2.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.base.base.matrix_base.run_case is not None
    assert not hasattr(module.base.archive_base, "run_case")
    assert module.FAILED_REASON.endswith("has no attribute 'run_case'")
