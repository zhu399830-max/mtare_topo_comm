#!/home/zeng-workstation/anaconda3/bin/python
"""Bind V1R composed audit to the sealed aggregate-only V3R failure."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, write_json
from run_composed_frontier_attempt_audit_v1r import load_composed_source
from run_stochastic_v2_compatibility_audit_v1 import sha256


PROPOSAL_SPEC = Path(
    "configs/v3/gate6/aee_composite_v9_composed_frontier_attempt_audit_v1r.proposal.json"
)
PROPOSAL_CARD = Path(
    "configs/v3/gate6/data_cards/aee_composite_v9_composed_frontier_attempt_audit_v1r.proposal.json"
)
FINAL_SPEC = Path(
    "configs/v3/gate6/aee_composite_v9_composed_frontier_attempt_audit_v1r.json"
)
FINAL_CARD = Path(
    "configs/v3/gate6/data_cards/aee_composite_v9_composed_frontier_attempt_audit_v1r.json"
)


def _within(root: Path, relative: str | Path) -> Path:
    path = (root / relative).resolve()
    path.relative_to(root.resolve())
    return path


def finalize(
    *,
    proposal_spec_path: Path = PROPOSAL_SPEC,
    proposal_card_path: Path = PROPOSAL_CARD,
    final_spec_path: Path = FINAL_SPEC,
    final_card_path: Path = FINAL_CARD,
    project_root: Path = PROJECT_ROOT,
) -> tuple[Path, Path]:
    root = project_root.resolve()
    proposal_spec_path = _within(root, proposal_spec_path)
    proposal_card_path = _within(root, proposal_card_path)
    final_spec_path = _within(root, final_spec_path)
    final_card_path = _within(root, final_card_path)
    if final_spec_path.exists() or final_card_path.exists():
        raise RuntimeError("final V1R composed-audit card/spec already exists")
    proposal = load_json(proposal_spec_path)
    card_proposal = load_json(proposal_card_path)
    expected_pending = "PENDING_FAILED_SOURCE_SEAL_BINDING_NOT_EXECUTABLE"
    if proposal.get("status") != expected_pending:
        raise RuntimeError("V1R composed audit spec is not a pending proposal")
    if card_proposal.get("status") != expected_pending:
        raise RuntimeError("V1R composed audit card is not a pending proposal")
    for name, item in proposal["frozen_tools"].items():
        if sha256(_within(root, item["path"])) != item["sha256"]:
            raise RuntimeError(f"V1R composed audit proposal tool drift: {name}")

    source_seal_path = (
        _within(root, proposal["source_run"]) / "artifacts/evidence_sha256.txt"
    )
    if not source_seal_path.is_file():
        raise RuntimeError("V1R recovery source has no final failed-run seal")
    source_seal = sha256(source_seal_path)
    bound = copy.deepcopy(proposal)
    bound["source_seal_sha256"] = source_seal
    summaries, _, _, _, source = load_composed_source(bound, project_root=root)
    if len(summaries) != 90 or source["manifest"].get("source_counts") != {
        "failed_source_run": 69,
        "recovery_run": 21,
    }:
        raise RuntimeError("V1R composed source did not prove exact 69+21 identity")
    provenance = source["provenance"]
    if (
        provenance.get("source_status_promoted_to_pass") is not False
        or provenance.get("source_case_pass_count") != 21
        or provenance.get("source_failure_class")
        != "aggregate_status_compatibility_only"
    ):
        raise RuntimeError("V1R aggregate-only source classification drift")

    card = copy.deepcopy(card_proposal)
    card["status"] = "APPROVED_FOR_ONE_FORMAL_EXECUTION"
    card["approval"]["status"] = "APPROVED"
    card["sources"]["recovery_seal_sha256"] = source_seal
    write_json(final_card_path, card)

    final = copy.deepcopy(proposal)
    final.pop("status", None)
    final["user_authorization"]["status"] = "APPROVED"
    final["source_seal_sha256"] = source_seal
    final["config_path"] = final_spec_path.relative_to(root).as_posix()
    final["data_card"] = final_card_path.relative_to(root).as_posix()
    final["frozen_tools"].pop("data_card_proposal")
    final["frozen_tools"]["data_card"] = {
        "path": final_card_path.relative_to(root).as_posix(),
        "sha256": sha256(final_card_path),
    }
    finalizer = Path(__file__).resolve()
    final["frozen_tools"]["proposal_finalizer"] = {
        "path": finalizer.relative_to(root).as_posix(),
        "sha256": sha256(finalizer),
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
