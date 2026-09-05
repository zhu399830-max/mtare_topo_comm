from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]


def _load():
    tools = str(ROOT / "tools/v3")
    if tools not in sys.path:
        sys.path.insert(0, tools)
    spec = importlib.util.spec_from_file_location(
        "finalize_aee_composite_v9_combined_correction_stochastic_v1",
        ROOT / "tools/v3/finalize_aee_composite_v9_combined_correction_stochastic_v1.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _seal(module, root: Path, run: Path, paths: list[Path]) -> str:
    target = run / "artifacts/evidence_sha256.txt"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "".join(
            f"{module.sha256(path)}  {path.relative_to(root).as_posix()}\n"
            for path in paths
        ),
        encoding="utf-8",
    )
    return module.sha256(target)


def _fixture(tmp_path: Path, *, wrong_quota: bool = False):
    module = _load()
    audit = tmp_path / "audit"
    audit_state = audit / "RUN_STATE.json"
    audit_summary = audit / "metrics/summary.json"
    audit_manifest = audit / "config/source_manifest.json"
    audit_provenance = audit / "config/source_provenance.json"
    _write(audit_state, {
        "state": "COMPLETED", "overall_status": module.AUDIT_STATUS,
    })
    _write(audit_summary, {
        "overall_status": module.AUDIT_STATUS,
        "source_case_count": 90, "v9_case_count": 30,
    })
    rows = []
    for index in range(90):
        is_m1d = index < 30
        source = "failed_source_run"
        if is_m1d and not wrong_quota and index >= 22:
            source = "recovery_run"
        rows.append({
            "index": index,
            "case_id": f"case_{index:03d}",
            "method_family": "m1d_topology" if is_m1d else "original_mtare",
            "source": source,
        })
    _write(audit_manifest, {"case_count": 90, "cases": rows})
    _write(audit_provenance, {"source_mutation_permitted": False})
    audit_seal = _seal(module, tmp_path, audit, [
        audit_state, audit_summary, audit_manifest, audit_provenance,
    ])

    readiness = tmp_path / "readiness"
    readiness_state = readiness / "RUN_STATE.json"
    readiness_summary = readiness / "metrics/summary.json"
    _write(readiness_state, {
        "state": "COMPLETED", "overall_status": module.READINESS_STATUS,
    })
    _write(readiness_summary, {
        "overall_status": module.READINESS_STATUS,
        "completed_cases": 6,
        "all_case_gates_passed": True,
        "maximum_post_warmup_fallback_rate": 0.0,
    })
    readiness_seal = _seal(
        module, tmp_path, readiness, [readiness_state, readiness_summary]
    )

    tool = tmp_path / "tool.py"
    tool.write_text("pass\n", encoding="utf-8")
    proposal_path = tmp_path / "proposal.json"
    card_path = tmp_path / "card.proposal.json"
    final_spec = tmp_path / "final.json"
    final_card = tmp_path / "card.json"
    _write(proposal_path, {
        "status": module.PENDING_STATUS,
        "user_authorization": {"status": "APPROVED_AFTER_EXACT_SOURCE_BINDING"},
        "paired_source_audit_run": "audit",
        "pending_paired_source_audit_seal_sha256": "PENDING",
        "readiness_run": "readiness",
        "pending_readiness_seal_sha256": "PENDING",
        "config_path": "final.json",
        "data_card": "card.json",
        "frozen_tools": {
            "fixture": {"path": "tool.py", "sha256": module.sha256(tool)},
            "data_card_proposal": {
                "path": "card.proposal.json", "sha256": "PENDING",
            },
        },
    })
    _write(card_path, {
        "status": module.PENDING_STATUS,
        "approval": {"status": "APPROVED_AFTER_EXACT_SOURCE_BINDING"},
        "sources": {},
    })
    proposal = json.loads(proposal_path.read_text())
    proposal["frozen_tools"]["data_card_proposal"]["sha256"] = module.sha256(card_path)
    _write(proposal_path, proposal)
    return module, proposal_path, card_path, final_spec, final_card, audit_seal, readiness_seal


def test_finalizer_binds_both_exact_source_seals(tmp_path: Path):
    module, proposal, card, final_spec, final_card, audit_seal, readiness_seal = _fixture(
        tmp_path
    )
    module.finalize(
        proposal_spec_path=proposal,
        proposal_card_path=card,
        final_spec_path=final_spec,
        final_card_path=final_card,
        project_root=tmp_path,
    )
    spec = json.loads(final_spec.read_text())
    data_card = json.loads(final_card.read_text())
    assert spec["user_authorization"]["status"] == "APPROVED"
    assert spec["paired_source_audit_seal_sha256"] == audit_seal
    assert spec["readiness_seal_sha256"] == readiness_seal
    assert spec["frozen_tools"]["data_card"]["sha256"] == module.sha256(final_card)
    assert data_card["status"] == "APPROVED_FOR_ONE_FORMAL_EXECUTION"
    assert data_card["sources"]["paired_source_audit"]["m1d_source_counts"] == {
        "failed_source_run": 22, "recovery_run": 8,
    }


def test_finalizer_rejects_wrong_m1d_source_quota(tmp_path: Path):
    module, proposal, card, final_spec, final_card, _, _ = _fixture(
        tmp_path, wrong_quota=True
    )
    with pytest.raises(RuntimeError, match=r"22\+8"):
        module.finalize(
            proposal_spec_path=proposal,
            proposal_card_path=card,
            final_spec_path=final_spec,
            final_card_path=final_card,
            project_root=tmp_path,
        )
    assert not final_spec.exists()
    assert not final_card.exists()
