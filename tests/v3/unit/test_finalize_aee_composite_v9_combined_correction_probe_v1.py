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
        "run_aee_composite_v9_combined_correction_probe_v1",
        "finalize_aee_composite_v9_combined_correction_probe_v1",
    ):
        spec = importlib.util.spec_from_file_location(name, ROOT / f"tools/v3/{name}.py")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        loaded.append(module)
    return loaded


def _audit_source(tmp_path: Path, probe):
    source = tmp_path / "audit"
    (source / "metrics").mkdir(parents=True)
    (source / "config").mkdir()
    (source / "artifacts").mkdir()
    selected = {
        "case_id": "052_tunnel_env11_m1d_seed0_repeat0",
        "world": "tunnel", "environment_seed": 11, "checkpoint_seed": 0,
        "first_reanchor_proxy_frame": 144,
        "first_nonmatching_frontier_event_frame": 119,
        "both_mechanisms_observed_by_frame": 144,
    }
    paths = [
        source / "RUN_STATE.json", source / "metrics/summary.json",
        source / "metrics/frontier_attempt_outcomes.json",
        source / "config/source_manifest.json", source / "config/source_provenance.json",
    ]
    paths[0].write_text(json.dumps({"state": "COMPLETED", "overall_status": probe.AUDIT_STATUS_PASS}) + "\n")
    paths[1].write_text(json.dumps({
        "overall_status": probe.AUDIT_STATUS_PASS, "source_case_count": 90, "v9_case_count": 30,
    }) + "\n")
    cases = [{
        "case_id": selected["case_id"], "world": "tunnel",
        "environment_seed": 11, "checkpoint_seed": 0,
    }]
    cases.extend({
        "case_id": f"case_{index}", "world": "garage",
        "environment_seed": index, "checkpoint_seed": index % 3,
    } for index in range(29))
    paths[2].write_text(json.dumps({
        "case_count": 30, "raw_bag_reads": 0, "gt_or_map_reads": 0, "cases": cases,
        "combined_correction_probe_selection": {
            "selection_rule": "minimize_max_first_mechanism_frame_then_case_id",
            "performance_outcomes_used_for_selection": [],
            "eligible_case_count": 1, "selected_case": selected,
            "ranked_candidates": [selected],
        },
    }) + "\n")
    paths[3].write_text(json.dumps({
        "case_count": 90,
        "family_case_counts": {
            "layered_gt_map_oracle": 30, "m1d_topology": 30, "original_mtare": 30,
        },
    }) + "\n")
    paths[4].write_text(json.dumps({"source_mutation_permitted": False}) + "\n")
    seal_path = source / "artifacts/evidence_sha256.txt"
    seal_path.write_text("".join(
        f"{probe.sha256(path)}  {path.relative_to(tmp_path).as_posix()}\n" for path in paths
    ))
    return selected, probe.sha256(seal_path)


def test_finalizer_binds_exact_audit_selection_and_materializes_once(tmp_path: Path):
    probe, finalizer = _modules()
    selected, seal_hash = _audit_source(tmp_path, probe)
    configs = tmp_path / "configs"
    configs.mkdir()
    tool = tmp_path / "tool.py"
    tool.write_text("pass\n")
    proposal_card = configs / "card.proposal.json"
    proposal_spec = configs / "proposal.json"
    final_card = configs / "card.json"
    final_spec = configs / "spec.json"
    card = {
        "status": "PENDING_MECHANISM_AUDIT_SEAL_NOT_EXECUTABLE",
        "approval": {"status": "APPROVED_AFTER_EXACT_SOURCE_BINDING"},
        "sources": {"mechanism_audit_seal_sha256": "PENDING"},
        "sampling": {},
    }
    proposal_card.write_text(json.dumps(card) + "\n")
    proposal = {
        "status": "PENDING_MECHANISM_AUDIT_SEAL_NOT_EXECUTABLE",
        "mechanism_audit_run": "audit",
        "mechanism_audit_status": probe.AUDIT_STATUS_PASS,
        "mechanism_audit_seal_sha256": "PENDING",
        "probe_case": "PENDING",
        "user_authorization": {"status": "APPROVED_AFTER_EXACT_SOURCE_BINDING"},
        "frozen_tools": {
            "data_card_proposal": {
                "path": "configs/card.proposal.json",
                "sha256": probe.sha256(proposal_card),
            },
            "tool": {"path": "tool.py", "sha256": probe.sha256(tool)},
        },
    }
    proposal_spec.write_text(json.dumps(proposal) + "\n")
    card_path, spec_path = finalizer.finalize(
        proposal_spec_path=proposal_spec.relative_to(tmp_path),
        proposal_card_path=proposal_card.relative_to(tmp_path),
        final_spec_path=final_spec.relative_to(tmp_path),
        final_card_path=final_card.relative_to(tmp_path),
        project_root=tmp_path,
    )
    final = json.loads(spec_path.read_text())
    finalized_card = json.loads(card_path.read_text())
    assert final["mechanism_audit_seal_sha256"] == seal_hash
    assert final["probe_case"]["source_case_id"] == selected["case_id"]
    assert final["probe_case"]["runtime_sec"] == 180.0
    assert final["user_authorization"]["status"] == "APPROVED"
    assert finalized_card["sampling"]["selected_source_case"] == selected
    with pytest.raises(RuntimeError, match="already exists"):
        finalizer.finalize(
            proposal_spec_path=proposal_spec.relative_to(tmp_path),
            proposal_card_path=proposal_card.relative_to(tmp_path),
            final_spec_path=final_spec.relative_to(tmp_path),
            final_card_path=final_card.relative_to(tmp_path),
            project_root=tmp_path,
        )


def test_finalizer_rejects_missing_audit_seal(tmp_path: Path):
    _, finalizer = _modules()
    configs = tmp_path / "configs"
    configs.mkdir()
    (tmp_path / "audit").mkdir()
    card = configs / "card.proposal.json"
    spec = configs / "proposal.json"
    card.write_text(json.dumps({
        "status": "PENDING_MECHANISM_AUDIT_SEAL_NOT_EXECUTABLE",
        "approval": {}, "sources": {}, "sampling": {},
    }) + "\n")
    spec.write_text(json.dumps({
        "status": "PENDING_MECHANISM_AUDIT_SEAL_NOT_EXECUTABLE",
        "mechanism_audit_run": "audit", "frozen_tools": {},
    }) + "\n")
    with pytest.raises(RuntimeError, match="no final seal"):
        finalizer.finalize(
            proposal_spec_path=spec.relative_to(tmp_path),
            proposal_card_path=card.relative_to(tmp_path),
            final_spec_path=Path("configs/final.json"),
            final_card_path=Path("configs/card.json"),
            project_root=tmp_path,
        )
