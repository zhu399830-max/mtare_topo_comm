"""Zero-data contracts for the AEE corrective topology candidate runner."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
TOOLS = ROOT / "tools/v3"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
spec = importlib.util.spec_from_file_location(
    "run_aee_corrective_topology_candidate_audit_v1",
    TOOLS / "run_aee_corrective_topology_candidate_audit_v1.py",
)
assert spec is not None and spec.loader is not None
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_frozen_scope_and_candidate_registry_are_exact() -> None:
    proposal = json.loads(
        (ROOT / "configs/v3/gate2/aee_corrective_topology_candidate_audit_v1.proposal.json").read_text()
    )
    run_spec = json.loads(
        (
            ROOT
            / "configs/v3/gate2/aee_corrective_topology_candidate_audit_v1.json"
        ).read_text()
    )
    scope = proposal["frozen_candidate_scope"]
    registry = runner.build_arithmetic_candidate_registry(
        scope["strata"],
        first_candidate_index=scope["first_candidate_index"],
        candidates_per_stratum=scope["candidates_per_stratum"],
        topology_seed_base=scope["topology_seed_base"],
        geometry_seed_base=scope["reserved_geometry_seed_base"],
    )
    assert len(registry) == 120
    assert registry[0]["candidate_id"] == "S01_flat_tree_small_C13"
    assert registry[-1]["candidate_id"] == "S10_3d_complex_C24"
    assert runner.canonical_json_hash({"candidates": registry}) == run_spec["candidate_registry_sha256"]
    assert run_spec["data_scope"] == runner.EXPECTED_SCOPE


def test_normalized_freeze_hash_is_order_independent_but_content_strict() -> None:
    first = "b==2\na==1\n"
    second = "a==1\n\nb==2\n"
    assert runner._normalized_freeze_sha256(first) == runner._normalized_freeze_sha256(second)
    assert runner._normalized_freeze_sha256(first) != runner._normalized_freeze_sha256("a==1\nb==3\n")


def test_persistent_environment_and_historical_sources_pass_preentry() -> None:
    run_spec = json.loads(
        (
            ROOT
            / "configs/v3/gate2/aee_corrective_topology_candidate_audit_v1.json"
        ).read_text()
    )
    audit = runner._verify_environment_and_sources(run_spec)
    assert audit["live_normalized_freeze_sha256"] == audit["historical_normalized_freeze_sha256"]
    assert audit["pip_check"] == "No broken requirements found."
    assert audit["upstream_clean"] is True
    assert audit["upstream_commit"] == "b6c77621187404b4dfab1249c7a1b40f63ad9ab3"


def test_formal_output_directory_lifecycle_is_valid() -> None:
    run_dir = ROOT / "results/gate2_representation" / runner.RUN_ID
    if not run_dir.exists():
        return
    state = json.loads((run_dir / "RUN_STATE.json").read_text())
    assert state == {
        "overall_status": "PASS_AEE_CORRECTIVE_TOPOLOGY_CANDIDATE_AUDIT_V1",
        "run_id": runner.RUN_ID,
        "schema_version": "v3_run_state_v1",
        "state": "COMPLETED",
    }
    assert (run_dir / "artifacts/evidence_sha256.txt").is_file()
