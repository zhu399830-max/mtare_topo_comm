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
        "finalize_aee_composite_v9_compatibility_audit_v1",
        ROOT / "tools/v3/finalize_aee_composite_v9_compatibility_audit_v1.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _source(runner, root: Path, *, state: str = "FAILED") -> tuple[Path, list[dict]]:
    source = root / "source"
    (source / "metrics").mkdir(parents=True)
    (source / "config").mkdir()
    (source / "artifacts").mkdir()
    state_path = source / "RUN_STATE.json"
    summary_path = source / "metrics/summary.json"
    schedule_path = source / "config/case_schedule.json"
    cases = [{"index": index, "case_id": f"{index:03d}"} for index in range(90)]
    state_path.write_text(
        json.dumps({"state": state, "overall_status": "FAIL_SOURCE"}) + "\n",
        encoding="utf-8",
    )
    summary_path.write_text(
        json.dumps({"completed_case_count": 90}) + "\n", encoding="utf-8"
    )
    schedule_path.write_text(json.dumps({"cases": cases}) + "\n", encoding="utf-8")
    seal_path = source / "artifacts/evidence_sha256.txt"
    seal_path.write_text(
        "".join(
            f"{runner.sha256(path)}  {path.relative_to(root).as_posix()}\n"
            for path in (state_path, summary_path, schedule_path)
        ),
        encoding="utf-8",
    )
    return source, cases


def _proposals(runner, root: Path, source: Path, cases: list[dict]) -> tuple[Path, Path]:
    tool = root / "tool.py"
    tool.write_text("pass\n", encoding="utf-8")
    spec_path = root / "proposal_spec.json"
    card_path = root / "proposal_card.json"
    schedule = source / "config/case_schedule.json"
    spec_path.write_text(json.dumps({
        "status": "PENDING_FINAL_SOURCE_SEAL_NOT_EXECUTABLE",
        "source_run": "source",
        "expected_source_state": "FAILED",
        "expected_source_status": "FAIL_SOURCE",
        "source_seal_sha256": "PENDING_FINAL_SOURCE_SEAL",
        "source_schedule_file_sha256": runner.sha256(schedule),
        "source_schedule_content_sha256": runner.schedule_content_sha256(cases),
        "config_path": "final_spec.json",
        "data_card": "final_card.json",
        "user_authorization": {"status": "APPROVED_AFTER_EXACT_SOURCE_SEAL_BINDING"},
        "frozen_tools": {
            "dummy": {"path": "tool.py", "sha256": runner.sha256(tool)},
            "data_card_proposal": {
                "path": "proposal_card.json",
                "sha256": "PENDING_CARD_HASH",
            },
        },
    }) + "\n", encoding="utf-8")
    card_path.write_text(json.dumps({
        "status": "PENDING_FINAL_SOURCE_SEAL_NOT_EXECUTABLE",
        "approval": {"status": "APPROVED_IN_SCOPE_AFTER_EXACT_SOURCE_SEAL_BINDING"},
        "source": {"source_seal_sha256": "PENDING_FINAL_SOURCE_SEAL"},
    }) + "\n", encoding="utf-8")
    proposal = json.loads(spec_path.read_text())
    proposal["frozen_tools"]["data_card_proposal"]["sha256"] = runner.sha256(card_path)
    spec_path.write_text(json.dumps(proposal) + "\n", encoding="utf-8")
    return spec_path, card_path


def test_finalizer_binds_exact_source_seal_and_writes_once(tmp_path: Path) -> None:
    runner = _load()
    source, cases = _source(runner, tmp_path)
    proposal_spec, proposal_card = _proposals(runner, tmp_path, source, cases)
    final_spec, final_card = tmp_path / "final_spec.json", tmp_path / "final_card.json"
    card_path, spec_path = runner.finalize(
        proposal_spec_path=proposal_spec.relative_to(tmp_path),
        proposal_card_path=proposal_card.relative_to(tmp_path),
        final_spec_path=final_spec.relative_to(tmp_path),
        final_card_path=final_card.relative_to(tmp_path),
        project_root=tmp_path,
    )
    card, spec = json.loads(card_path.read_text()), json.loads(spec_path.read_text())
    source_seal = runner.sha256(source / "artifacts/evidence_sha256.txt")
    assert card["status"] == "APPROVED_FOR_ONE_FORMAL_EXECUTION"
    assert card["source"]["source_seal_sha256"] == source_seal
    assert spec["user_authorization"]["status"] == "APPROVED"
    assert spec["source_seal_sha256"] == source_seal
    assert "data_card_proposal" not in spec["frozen_tools"]
    assert spec["frozen_tools"]["data_card"]["sha256"] == runner.sha256(final_card)
    with pytest.raises(RuntimeError, match="already exists"):
        runner.finalize(
            proposal_spec_path=proposal_spec.relative_to(tmp_path),
            proposal_card_path=proposal_card.relative_to(tmp_path),
            final_spec_path=final_spec.relative_to(tmp_path),
            final_card_path=final_card.relative_to(tmp_path),
            project_root=tmp_path,
        )


def test_finalizer_rejects_unfinished_source(tmp_path: Path) -> None:
    runner = _load()
    source, cases = _source(runner, tmp_path, state="RUNNING")
    proposal_spec, proposal_card = _proposals(runner, tmp_path, source, cases)
    with pytest.raises(RuntimeError, match="finalized FAIL"):
        runner.finalize(
            proposal_spec_path=proposal_spec.relative_to(tmp_path),
            proposal_card_path=proposal_card.relative_to(tmp_path),
            final_spec_path=Path("final_spec.json"),
            final_card_path=Path("final_card.json"),
            project_root=tmp_path,
        )
