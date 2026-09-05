#!/home/zeng-workstation/anaconda3/bin/python
"""Bind the V5 30-case corrective matrix to its two sealed prerequisites."""

from __future__ import annotations

import argparse
import copy
from collections import Counter
from pathlib import Path
from typing import Any

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, write_json
from run_mtare_single_robot_stochastic_v1 import sha256


PROPOSAL_SPEC = Path(
    "configs/v3/gate6/aee_composite_v9_combined_correction_stochastic_v1.proposal.json"
)
PROPOSAL_CARD = Path(
    "configs/v3/gate6/data_cards/"
    "aee_composite_v9_combined_correction_stochastic_v1.proposal.json"
)
FINAL_SPEC = Path(
    "configs/v3/gate6/aee_composite_v9_combined_correction_stochastic_v1.json"
)
FINAL_CARD = Path(
    "configs/v3/gate6/data_cards/aee_composite_v9_combined_correction_stochastic_v1.json"
)
PENDING_STATUS = "PENDING_READINESS_AND_COMPOSED_AUDIT_SEALS_NOT_EXECUTABLE"
AUDIT_STATUS = "PASS_COMPOSED_FRONTIER_ATTEMPT_AUDIT_V1R"
READINESS_STATUS = "PASS_AEE_COMPOSITE_V9_COMBINED_CORRECTION_READINESS_V1R"


def _within(root: Path, relative: str | Path) -> Path:
    value = (root / relative).resolve()
    value.relative_to(root.resolve())
    return value


def _seal_entries(root: Path, source: Path) -> tuple[str, dict[str, str]]:
    seal_path = source / "artifacts/evidence_sha256.txt"
    if not seal_path.is_file():
        raise RuntimeError(f"sealed source is missing evidence manifest: {source.name}")
    entries: dict[str, str] = {}
    prefix = source.relative_to(root).as_posix() + "/"
    for line in seal_path.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        if relative in entries or not relative.startswith(prefix):
            raise RuntimeError(f"invalid sealed source entry: {relative}")
        path = _within(root, relative)
        if not path.is_file() or sha256(path) != digest:
            raise RuntimeError(f"sealed source evidence drift: {relative}")
        entries[relative] = digest
    if not entries:
        raise RuntimeError(f"sealed source has no entries: {source.name}")
    return sha256(seal_path), entries


def _require_bound(
    root: Path, source: Path, entries: dict[str, str], relative: str
) -> dict[str, Any]:
    path = source / relative
    key = path.relative_to(root).as_posix()
    if entries.get(key) != sha256(path):
        raise RuntimeError(f"required source evidence is not seal-bound: {key}")
    return load_json(path)


def _audit_source(root: Path, proposal: dict[str, Any]) -> dict[str, Any]:
    source = _within(root, proposal["paired_source_audit_run"])
    seal_digest, entries = _seal_entries(root, source)
    if proposal.get("pending_paired_source_audit_seal_sha256") != "PENDING":
        raise RuntimeError("composed source audit proposal placeholder drift")
    state = _require_bound(root, source, entries, "RUN_STATE.json")
    summary = _require_bound(root, source, entries, "metrics/summary.json")
    manifest = _require_bound(root, source, entries, "config/source_manifest.json")
    provenance = _require_bound(root, source, entries, "config/source_provenance.json")
    if (
        state.get("state") != "COMPLETED"
        or state.get("overall_status") != AUDIT_STATUS
        or summary.get("overall_status") != AUDIT_STATUS
        or summary.get("source_case_count") != 90
        or summary.get("v9_case_count") != 30
        or manifest.get("case_count") != 90
        or provenance.get("source_mutation_permitted") is not False
    ):
        raise RuntimeError("composed source audit is not the exact 90-case PASS")
    cases = manifest.get("cases")
    if not isinstance(cases, list) or len(cases) != 90:
        raise RuntimeError("composed source manifest case rows are incomplete")
    m1d_sources = Counter(
        row.get("source")
        for row in cases
        if row.get("method_family") == "m1d_topology"
    )
    if m1d_sources != Counter({"failed_source_run": 22, "recovery_run": 8}):
        raise RuntimeError("composed source M1D 22+8 provenance quota drift")
    return {
        "run": source.relative_to(root).as_posix(),
        "status": AUDIT_STATUS,
        "seal_sha256": seal_digest,
        "seal_entry_count": len(entries),
        "source_case_count": 90,
        "v9_case_count": 30,
        "m1d_source_counts": dict(sorted(m1d_sources.items())),
    }


def _readiness_source(root: Path, proposal: dict[str, Any]) -> dict[str, Any]:
    source = _within(root, proposal["readiness_run"])
    seal_digest, entries = _seal_entries(root, source)
    if proposal.get("pending_readiness_seal_sha256") != "PENDING":
        raise RuntimeError("V5 readiness proposal placeholder drift")
    state = _require_bound(root, source, entries, "RUN_STATE.json")
    summary = _require_bound(root, source, entries, "metrics/summary.json")
    if (
        state.get("state") != "COMPLETED"
        or state.get("overall_status") != READINESS_STATUS
        or summary.get("overall_status") != READINESS_STATUS
        or summary.get("completed_cases") != 6
        or summary.get("all_case_gates_passed") is not True
        or float(summary.get("maximum_post_warmup_fallback_rate", 1.0)) > 0.05
    ):
        raise RuntimeError("V5 readiness is not the exact six-case PASS")
    return {
        "run": source.relative_to(root).as_posix(),
        "status": READINESS_STATUS,
        "seal_sha256": seal_digest,
        "seal_entry_count": len(entries),
        "completed_cases": 6,
        "maximum_post_warmup_fallback_rate": summary[
            "maximum_post_warmup_fallback_rate"
        ],
    }


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
        raise RuntimeError("final V5 stochastic card/spec already exists")
    proposal = load_json(proposal_spec_path)
    card_proposal = load_json(proposal_card_path)
    if (
        proposal.get("status") != PENDING_STATUS
        or card_proposal.get("status") != PENDING_STATUS
    ):
        raise RuntimeError("V5 stochastic proposal is not pending exact source binding")
    for name, item in proposal["frozen_tools"].items():
        if sha256(_within(root, item["path"])) != item["sha256"]:
            raise RuntimeError(f"V5 stochastic proposal tool drift: {name}")

    audit = _audit_source(root, proposal)
    readiness = _readiness_source(root, proposal)

    card = copy.deepcopy(card_proposal)
    card["status"] = "APPROVED_FOR_ONE_FORMAL_EXECUTION"
    card["approval"]["status"] = "APPROVED"
    card["sources"]["paired_source_audit"] = audit
    card["sources"]["readiness"] = readiness
    write_json(final_card_path, card)

    final = copy.deepcopy(proposal)
    final.pop("status", None)
    final["user_authorization"]["status"] = "APPROVED"
    final["paired_source_audit_seal_sha256"] = audit["seal_sha256"]
    final["readiness_seal_sha256"] = readiness["seal_sha256"]
    final.pop("pending_paired_source_audit_seal_sha256", None)
    final.pop("pending_readiness_seal_sha256", None)
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
            "path": relative,
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
