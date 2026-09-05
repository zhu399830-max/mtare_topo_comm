from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / (
    "results/gate6_single_robot/"
    "gate6_20260823_aee_composite_v9_stochastic_v3r_seed20260820"
)
PROPOSAL = ROOT / (
    "configs/v3/gate6/"
    "aee_composite_v9_composed_frontier_attempt_audit_v1.proposal.json"
)


def _load():
    tools = str(ROOT / "tools/v3")
    if tools not in sys.path:
        sys.path.insert(0, tools)
    spec = importlib.util.spec_from_file_location(
        "run_composed_frontier_attempt_audit_v1r_test",
        ROOT / "tools/v3/run_composed_frontier_attempt_audit_v1r.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _bound_spec(module) -> dict:
    spec = json.loads(PROPOSAL.read_text(encoding="utf-8"))
    spec["source_seal_sha256"] = module.base.sha256(
        SOURCE / "artifacts/evidence_sha256.txt"
    )
    return spec


def test_v1r_accepts_only_exact_sealed_aggregate_failure_and_composes_90_cases():
    module = _load()
    summaries, roots, seals, source_specs, source = module.load_composed_source(
        _bound_spec(module)
    )
    assert len(summaries) == len(roots) == 90
    assert source["manifest"]["source_counts"] == {
        "failed_source_run": 69,
        "recovery_run": 21,
    }
    assert source["manifest"]["family_case_counts"] == {
        "layered_gt_map_oracle": 30,
        "m1d_topology": 30,
        "original_mtare": 30,
    }
    assert source["provenance"]["source_status_promoted_to_pass"] is False
    assert source["provenance"]["source_failure_reason"] == (
        "analysis cannot include a failed case"
    )
    assert set(seals) == set(source_specs) == {
        "failed_source_run",
        "recovery_run",
    }


def test_v1r_rejects_any_recovery_failure_reason_drift(monkeypatch):
    module = _load()
    original = module.base._verify_root_identity

    def altered(*args, **kwargs):
        summary, spec, bound = original(*args, **kwargs)
        root = args[0]
        if root == SOURCE:
            summary = copy.deepcopy(summary)
            summary["failure_reason"] = "different aggregate failure"
        return summary, spec, bound

    monkeypatch.setattr(module.base, "_verify_root_identity", altered)
    with pytest.raises(RuntimeError, match="aggregate-only failure reason drift"):
        module.load_composed_source(_bound_spec(module))


def test_v1r_full_real_source_statistics_and_probe_dataflow():
    module = _load()
    spec = _bound_spec(module)
    summaries, roots, seals, source_specs, source = module.load_composed_source(spec)
    generator = module.validate_composed_generator_identity(spec, source_specs)
    statistics = module.base.analyze_finalized_v2_cases(summaries)
    evidence = module.audit_composed_frontier_attempts(
        summaries, roots, seals, source["manifest"]
    )
    assert set(generator["direct_local_hashes_verified"]) == {
        "causal_graph",
        "frontier_planner",
        "online_runtime",
        "v9_node",
    }
    assert statistics["case_count"] == 90
    assert statistics["block_count"] == 10
    assert evidence["case_count"] == 30
    assert evidence["verified_trace_snapshot_file_count"] == 60
    assert evidence["combined_correction_probe_selection"]["selected_case"][
        "case_id"
    ] == "052_tunnel_env11_m1d_seed0_repeat0"
