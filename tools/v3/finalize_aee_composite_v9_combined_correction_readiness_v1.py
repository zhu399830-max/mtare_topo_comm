#!/home/zeng-workstation/anaconda3/bin/python
"""Bind V5 six-case readiness to the completed source-qualified probe."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, write_json
import run_aee_composite_v9_combined_correction_readiness_v1 as readiness
from run_mtare_single_robot_stochastic_v1 import sha256


PROPOSAL_SPEC = Path("configs/v3/gate6/aee_composite_v9_combined_correction_readiness_v1.proposal.json")
PROPOSAL_CARD = Path("configs/v3/gate6/data_cards/aee_composite_v9_combined_correction_readiness_v1.proposal.json")
FINAL_SPEC = Path("configs/v3/gate6/aee_composite_v9_combined_correction_readiness_v1.json")
FINAL_CARD = Path("configs/v3/gate6/data_cards/aee_composite_v9_combined_correction_readiness_v1.json")


def _within(root: Path, relative: str | Path) -> Path:
    value = (root / relative).resolve()
    value.relative_to(root.resolve())
    return value


def finalize(
    *, proposal_spec_path: Path = PROPOSAL_SPEC,
    proposal_card_path: Path = PROPOSAL_CARD,
    final_spec_path: Path = FINAL_SPEC,
    final_card_path: Path = FINAL_CARD,
    project_root: Path = PROJECT_ROOT,
) -> tuple[Path, Path]:
    root = project_root.resolve()
    paths = [
        _within(root, proposal_spec_path), _within(root, proposal_card_path),
        _within(root, final_spec_path), _within(root, final_card_path),
    ]
    proposal_spec_path, proposal_card_path, final_spec_path, final_card_path = paths
    if final_spec_path.exists() or final_card_path.exists():
        raise RuntimeError("final combined readiness card/spec already exists")
    proposal, card_proposal = load_json(proposal_spec_path), load_json(proposal_card_path)
    expected_pending = "PENDING_COMBINED_PROBE_SEAL_NOT_EXECUTABLE"
    if proposal.get("status") != expected_pending or card_proposal.get("status") != expected_pending:
        raise RuntimeError("combined readiness proposal is not pending exact probe binding")
    for name, item in proposal["frozen_tools"].items():
        if sha256(_within(root, item["path"])) != item["sha256"]:
            raise RuntimeError(f"combined readiness proposal tool drift: {name}")
    probe_root = _within(root, proposal["combined_probe_run"])
    seal_path = probe_root / "artifacts/evidence_sha256.txt"
    summary_path = probe_root / "metrics/summary.json"
    if not seal_path.is_file() or not summary_path.is_file():
        raise RuntimeError("combined readiness source probe has no final evidence")
    summary = load_json(summary_path)
    case_id = summary.get("case_result", {}).get("case_id")
    if not isinstance(case_id, str) or not case_id:
        raise RuntimeError("combined readiness source probe case identity is missing")
    probe_seal = sha256(seal_path)
    bound = copy.deepcopy(proposal)
    bound["combined_probe_case_id"] = case_id
    bound["combined_probe_seal_sha256"] = probe_seal
    source = readiness.validate_probe_source(bound, project_root=root)

    card = copy.deepcopy(card_proposal)
    card["status"] = "APPROVED_FOR_ONE_FORMAL_EXECUTION"
    card["approval"]["status"] = "APPROVED"
    card["sources"]["combined_probe_seal_sha256"] = probe_seal
    card["sources"]["combined_probe_case_id"] = case_id
    card["sources"]["combined_probe_evidence"] = source
    write_json(final_card_path, card)

    final = copy.deepcopy(proposal)
    final.pop("status", None)
    final["user_authorization"]["status"] = "APPROVED"
    final["combined_probe_case_id"] = case_id
    final["combined_probe_seal_sha256"] = probe_seal
    final["config_path"] = final_spec_path.relative_to(root).as_posix()
    final["data_card"] = final_card_path.relative_to(root).as_posix()
    final["frozen_tools"].pop("data_card_proposal")
    final["frozen_tools"]["data_card"] = {
        "path": final_card_path.relative_to(root).as_posix(),
        "sha256": sha256(final_card_path),
    }
    finalizer = Path(__file__).resolve()
    try:
        relative = finalizer.relative_to(root).as_posix()
    except ValueError:
        relative = None
    if relative is not None:
        final["frozen_tools"]["proposal_finalizer"] = {
            "path": relative, "sha256": sha256(finalizer),
        }
    write_json(final_spec_path, final)
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
