from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
AUDIT = ROOT / (
    "results/gate6_single_robot/"
    "gate6_20260823_aee_composite_v9_composed_frontier_attempt_audit_v1r_seed20260820"
)


def _load():
    tools = str(ROOT / "tools/v3")
    if tools not in sys.path:
        sys.path.insert(0, tools)
    spec = importlib.util.spec_from_file_location(
        "run_aee_composite_v9_combined_correction_probe_v1r_test",
        ROOT / "tools/v3/run_aee_composite_v9_combined_correction_probe_v1r.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v1r_probe_accepts_exact_audit_and_selects_case052():
    module = _load()
    module.base.AUDIT_STATUS_PASS = module.AUDIT_STATUS_PASS
    spec = {
        "mechanism_audit_run": AUDIT.relative_to(ROOT).as_posix(),
        "mechanism_audit_status": module.AUDIT_STATUS_PASS,
        "mechanism_audit_seal_sha256": module.base.sha256(
            AUDIT / "artifacts/evidence_sha256.txt"
        ),
    }
    source = module.base.validate_composed_audit_source(spec)
    assert source["selected_source_case"] == {
        "case_id": "052_tunnel_env11_m1d_seed0_repeat0",
        "world": "tunnel",
        "environment_seed": 11,
        "checkpoint_seed": 0,
        "first_reanchor_proxy_frame": 144,
        "first_nonmatching_frontier_event_frame": 119,
        "both_mechanisms_observed_by_frame": 144,
    }
    selected = source["selected_source_case"]
    case = {
        "case_id": "tunnel_env11_m1d_seed0_combined_v5_probe",
        "source_case_id": selected["case_id"],
        "world": "tunnel",
        "environment_seed": 11,
        "checkpoint_seed": 0,
        "runtime_sec": 180.0,
    }
    assert module.base.validate_probe_case(case, selected) == case


def test_v1r_wrapper_sets_new_run_and_status_contract():
    module = _load()
    assert module.RUN_ID.endswith("combined_correction_probe_v1r_seed20260823")
    assert module.STATUS_PASS.endswith("COMBINED_CORRECTION_PROBE_V1R")
    assert module.AUDIT_STATUS_PASS == "PASS_COMPOSED_FRONTIER_ATTEMPT_AUDIT_V1R"
