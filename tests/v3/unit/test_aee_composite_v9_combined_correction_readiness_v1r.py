from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def test_v1r_is_startup_only_replacement():
    tools = str(ROOT / "tools/v3")
    if tools not in sys.path:
        sys.path.insert(0, tools)
    spec = importlib.util.spec_from_file_location(
        "run_aee_composite_v9_combined_correction_readiness_v1r",
        ROOT / "tools/v3/run_aee_composite_v9_combined_correction_readiness_v1r.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.RUN_ID.endswith("readiness_v1r_seed20260823")
    assert module.STATUS_PASS.endswith("READINESS_V1R")
    assert module.base.EXPECTED_PROBE_STATUS.endswith("PROBE_V1R2")
    assert module.base.PROJECT_ROOT == ROOT
