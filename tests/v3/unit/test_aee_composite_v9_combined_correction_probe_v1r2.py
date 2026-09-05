from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def _load():
    tools = str(ROOT / "tools/v3")
    if tools not in sys.path:
        sys.path.insert(0, tools)
    spec = importlib.util.spec_from_file_location(
        "run_aee_composite_v9_combined_correction_probe_v1r2_test",
        ROOT / "tools/v3/run_aee_composite_v9_combined_correction_probe_v1r2.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v1r2_changes_only_run_identity_while_retaining_v1r_audit_contract():
    module = _load()
    assert module.RUN_ID.endswith("combined_correction_probe_v1r2_seed20260823")
    assert module.STATUS_PASS.endswith("COMBINED_CORRECTION_PROBE_V1R2")
    assert module.AUDIT_STATUS_PASS == "PASS_COMPOSED_FRONTIER_ATTEMPT_AUDIT_V1R"


def test_python38_compatibility_uses_exact_binary64_ulp_at_one():
    runtime = (ROOT / "src/mtare_topo/topology/frontier_execution_feedback.py").read_text()
    evidence = (ROOT / "src/mtare_topo/evaluation/topology_frontier_attempt_evidence.py").read_text()
    assert "math.ulp" not in "\n".join(
        line for line in runtime.splitlines() if not line.lstrip().startswith("#")
    )
    assert "math.ulp" not in "\n".join(
        line for line in evidence.splitlines() if not line.lstrip().startswith("#")
    )
    assert "sys.float_info.epsilon" in runtime
    assert "sys.float_info.epsilon" in evidence
    assert sys.float_info.epsilon == math.ulp(1.0)
