from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def _load():
    tools = str(ROOT / "tools/v3")
    if tools not in sys.path:
        sys.path.insert(0, tools)
    spec = importlib.util.spec_from_file_location(
        "run_aee_composite_v9_stochastic_v3",
        ROOT / "tools/v3/run_aee_composite_v9_stochastic_v3.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v9_wrapper_freezes_expected_runtime_contract() -> None:
    runner = _load()
    assert runner.RUN_ID.endswith("aee_composite_v9_stochastic_v3_seed20260820")
    assert runner.STATUS_PASS == "PASS_AEE_COMPOSITE_V9_STOCHASTIC_V3"
    source = (ROOT / "tools/v3/run_aee_composite_v9_stochastic_v3.py").read_text()
    assert 'base.EXPECTED_READINESS_STATUS = "PASS_AEE_COMPOSITE_V9_READINESS_V1R3"' in source
    assert 'base.TOPIC_CONTRACT = "configs/v3/gate6/closed_loop_recording_topics_aee_v2.json"' in source
