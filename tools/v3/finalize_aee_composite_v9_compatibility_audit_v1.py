#!/home/zeng-workstation/anaconda3/bin/python
"""Materialize the compatibility-audit card/spec after the 90-case source seals."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, write_json
from run_stochastic_v2_compatibility_audit_v1 import (
    _parse_seal,
    schedule_content_sha256,
    sha256,
)


PROPOSAL_SPEC = Path(
    "configs/v3/gate6/aee_composite_v9_stochastic_v2_compatibility_audit_v1.proposal.json"
)
PROPOSAL_CARD = Path(
    "configs/v3/gate6/data_cards/"
    "aee_composite_v9_stochastic_v2_compatibility_audit_v1.proposal.json"
)
FINAL_SPEC = Path(
    "configs/v3/gate6/aee_composite_v9_stochastic_v2_compatibility_audit_v1.json"
)
FINAL_CARD = Path(
    "configs/v3/gate6/data_cards/"
    "aee_composite_v9_stochastic_v2_compatibility_audit_v1.json"
)


def _within(root: Path, relative: str | Path) -> Path:
    path = (root / relative).resolve()
    path.relative_to(root.resolve())
    return path


def validate_final_source(spec: dict[str, Any], *, project_root: Path) -> str:
    source = _within(project_root, spec["source_run"])
    state_path = source / "RUN_STATE.json"
    summary_path = source / "metrics/summary.json"
    schedule_path = source / "config/case_schedule.json"
    seal_path = source / "artifacts/evidence_sha256.txt"
    state = load_json(state_path)
    if (
        state.get("state") != spec["expected_source_state"]
        or state.get("overall_status") != spec["expected_source_status"]
    ):
        raise RuntimeError("compatibility source is not the expected finalized FAIL")
    summary = load_json(summary_path)
    if summary.get("completed_case_count") != 90:
        raise RuntimeError("compatibility source did not complete exactly 90 cases")
    schedule_file = sha256(schedule_path)
    if schedule_file != spec["source_schedule_file_sha256"]:
        raise RuntimeError("compatibility source schedule file drift")
    cases = load_json(schedule_path).get("cases")
    if not isinstance(cases, list) or len(cases) != 90:
        raise RuntimeError("compatibility source schedule is not exactly 90 cases")
    if schedule_content_sha256(cases) != spec["source_schedule_content_sha256"]:
        raise RuntimeError("compatibility source schedule content drift")
    seal_digest = sha256(seal_path)
    entries = _parse_seal(source, seal_digest)
    for path in (state_path, summary_path, schedule_path):
        relative = path.relative_to(project_root.resolve()).as_posix()
        if entries.get(relative) != sha256(path):
            raise RuntimeError(f"compatibility source bound-file drift: {relative}")
    return seal_digest


def finalize(
    *,
    proposal_spec_path: Path,
    proposal_card_path: Path,
    final_spec_path: Path,
    final_card_path: Path,
    project_root: Path = PROJECT_ROOT,
) -> tuple[Path, Path]:
    root = project_root.resolve()
    proposal_spec_path = _within(root, proposal_spec_path)
    proposal_card_path = _within(root, proposal_card_path)
    final_spec_path = _within(root, final_spec_path)
    final_card_path = _within(root, final_card_path)
    if final_spec_path.exists() or final_card_path.exists():
        raise RuntimeError("final compatibility card/spec already exists")
    proposal_spec, proposal_card = (
        load_json(proposal_spec_path), load_json(proposal_card_path)
    )
    if proposal_spec.get("status") != "PENDING_FINAL_SOURCE_SEAL_NOT_EXECUTABLE":
        raise RuntimeError("compatibility spec is not the pending proposal")
    if proposal_card.get("status") != "PENDING_FINAL_SOURCE_SEAL_NOT_EXECUTABLE":
        raise RuntimeError("compatibility card is not the pending proposal")
    for name, item in proposal_spec["frozen_tools"].items():
        if sha256(_within(root, item["path"])) != item["sha256"]:
            raise RuntimeError(f"compatibility proposal frozen-tool drift: {name}")
    source_seal = validate_final_source(proposal_spec, project_root=root)

    card = copy.deepcopy(proposal_card)
    card["status"] = "APPROVED_FOR_ONE_FORMAL_EXECUTION"
    card["approval"]["status"] = "APPROVED"
    card["source"]["source_seal_sha256"] = source_seal
    final_card_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(final_card_path, card)

    spec = copy.deepcopy(proposal_spec)
    spec.pop("status", None)
    spec["user_authorization"]["status"] = "APPROVED"
    spec["source_seal_sha256"] = source_seal
    spec["config_path"] = final_spec_path.relative_to(root).as_posix()
    spec["data_card"] = final_card_path.relative_to(root).as_posix()
    spec["frozen_tools"].pop("data_card_proposal")
    spec["frozen_tools"]["data_card"] = {
        "path": final_card_path.relative_to(root).as_posix(),
        "sha256": sha256(final_card_path),
    }
    finalizer_path = Path(__file__).resolve()
    try:
        finalizer_relative = finalizer_path.relative_to(root).as_posix()
    except ValueError:
        finalizer_relative = None
    if finalizer_relative is not None:
        spec["frozen_tools"]["proposal_finalizer"] = {
            "path": finalizer_relative,
            "sha256": sha256(finalizer_path),
        }
    write_json(final_spec_path, spec)
    return final_card_path, final_spec_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposal-spec", type=Path, default=PROPOSAL_SPEC)
    parser.add_argument("--proposal-card", type=Path, default=PROPOSAL_CARD)
    parser.add_argument("--final-spec", type=Path, default=FINAL_SPEC)
    parser.add_argument("--final-card", type=Path, default=FINAL_CARD)
    args = parser.parse_args()
    card, spec = finalize(
        proposal_spec_path=args.proposal_spec,
        proposal_card_path=args.proposal_card,
        final_spec_path=args.final_spec,
        final_card_path=args.final_card,
    )
    print(card.relative_to(PROJECT_ROOT))
    print(spec.relative_to(PROJECT_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
