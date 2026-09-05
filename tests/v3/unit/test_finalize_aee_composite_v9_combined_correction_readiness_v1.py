from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]


def _modules():
    tools = str(ROOT / "tools/v3")
    if tools not in sys.path:
        sys.path.insert(0, tools)
    loaded = []
    for name in (
        "run_aee_composite_v9_combined_correction_readiness_v1",
        "finalize_aee_composite_v9_combined_correction_readiness_v1",
    ):
        spec = importlib.util.spec_from_file_location(name, ROOT / f"tools/v3/{name}.py")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        loaded.append(module)
    return loaded


def _probe(tmp_path: Path, readiness):
    source = tmp_path / "probe"
    (source / "metrics").mkdir(parents=True)
    case_dir = source / "artifacts/cases/probe_case"
    case_dir.mkdir(parents=True)
    (source / "config").mkdir()
    selected = {"case_id": "source_case", "world": "tunnel", "environment_seed": 11, "checkpoint_seed": 0}
    paths = [
        source / "RUN_STATE.json", source / "metrics/summary.json",
        case_dir / "summary.json", source / "config/mechanism_audit_source.json",
    ]
    paths[0].write_text(json.dumps({
        "state": "COMPLETED", "overall_status": readiness.EXPECTED_PROBE_STATUS,
    }) + "\n")
    paths[1].write_text(json.dumps({
        "overall_status": readiness.EXPECTED_PROBE_STATUS, "completed_cases": 1,
        "case_result": {"case_id": "probe_case"}, "source_selected_case": selected,
        "mechanism_audit": {
            "verified_reanchor_count": 1,
            "frontier_execution_rejection_count": 0,
            "post_correction_route_growth": True, "uses_evaluator_gt": False,
        },
    }) + "\n")
    paths[2].write_text(json.dumps({"status": "PASS_SINGLE_ROBOT_CASE_V2"}) + "\n")
    paths[3].write_text(json.dumps({
        "selected_source_case": selected, "performance_outcomes_used_for_selection": [],
    }) + "\n")
    seal_path = source / "artifacts/evidence_sha256.txt"
    seal_path.write_text("".join(
        f"{readiness.sha256(path)}  {path.relative_to(tmp_path).as_posix()}\n" for path in paths
    ))
    return readiness.sha256(seal_path)


def test_finalizer_binds_completed_probe_and_materializes_once(tmp_path: Path):
    readiness, finalizer = _modules()
    seal_hash = _probe(tmp_path, readiness)
    configs = tmp_path / "configs"
    configs.mkdir()
    tool = tmp_path / "tool.py"
    tool.write_text("pass\n")
    card_proposal = configs / "card.proposal.json"
    spec_proposal = configs / "spec.proposal.json"
    card_final = configs / "card.json"
    spec_final = configs / "spec.json"
    card_proposal.write_text(json.dumps({
        "status": "PENDING_COMBINED_PROBE_SEAL_NOT_EXECUTABLE",
        "approval": {"status": "APPROVED_AFTER_EXACT_SOURCE_BINDING"},
        "sources": {},
    }) + "\n")
    spec_proposal.write_text(json.dumps({
        "status": "PENDING_COMBINED_PROBE_SEAL_NOT_EXECUTABLE",
        "combined_probe_run": "probe",
        "combined_probe_status": readiness.EXPECTED_PROBE_STATUS,
        "combined_probe_case_id": "PENDING", "combined_probe_seal_sha256": "PENDING",
        "user_authorization": {"status": "APPROVED_AFTER_EXACT_SOURCE_BINDING"},
        "frozen_tools": {
            "data_card_proposal": {
                "path": "configs/card.proposal.json",
                "sha256": readiness.sha256(card_proposal),
            },
            "tool": {"path": "tool.py", "sha256": readiness.sha256(tool)},
        },
    }) + "\n")
    card_path, spec_path = finalizer.finalize(
        proposal_spec_path=spec_proposal.relative_to(tmp_path),
        proposal_card_path=card_proposal.relative_to(tmp_path),
        final_spec_path=spec_final.relative_to(tmp_path),
        final_card_path=card_final.relative_to(tmp_path),
        project_root=tmp_path,
    )
    spec = json.loads(spec_path.read_text())
    card = json.loads(card_path.read_text())
    assert spec["combined_probe_case_id"] == "probe_case"
    assert spec["combined_probe_seal_sha256"] == seal_hash
    assert spec["user_authorization"]["status"] == "APPROVED"
    assert card["sources"]["combined_probe_case_id"] == "probe_case"
    with pytest.raises(RuntimeError, match="already exists"):
        finalizer.finalize(
            proposal_spec_path=spec_proposal.relative_to(tmp_path),
            proposal_card_path=card_proposal.relative_to(tmp_path),
            final_spec_path=spec_final.relative_to(tmp_path),
            final_card_path=card_final.relative_to(tmp_path),
            project_root=tmp_path,
        )
