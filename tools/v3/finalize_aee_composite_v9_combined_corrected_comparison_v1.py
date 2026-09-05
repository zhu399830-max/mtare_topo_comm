#!/home/zeng-workstation/anaconda3/bin/python
"""Bind the read-only V5 comparison to sealed composed and corrected runs."""

from __future__ import annotations

import argparse
import copy
from collections import Counter
from pathlib import Path
from typing import Any

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, write_json
from run_stochastic_v2_compatibility_audit_v1 import schedule_content_sha256, sha256


PROPOSAL_SPEC = Path(
    "configs/v3/gate6/aee_composite_v9_combined_corrected_comparison_v1.proposal.json"
)
PROPOSAL_CARD = Path(
    "configs/v3/gate6/data_cards/"
    "aee_composite_v9_combined_corrected_comparison_v1.proposal.json"
)
FINAL_SPEC = Path(
    "configs/v3/gate6/aee_composite_v9_combined_corrected_comparison_v1.json"
)
FINAL_CARD = Path(
    "configs/v3/gate6/data_cards/aee_composite_v9_combined_corrected_comparison_v1.json"
)
PENDING_STATUS = "PENDING_COMPOSED_AND_CORRECTED_SEALS_NOT_EXECUTABLE"
AUDIT_STATUS = "PASS_COMPOSED_FRONTIER_ATTEMPT_AUDIT_V1"
CORRECTED_STATUS = "PASS_AEE_COMPOSITE_V9_COMBINED_CORRECTION_STOCHASTIC_V1R2"


def _within(root: Path, relative: str | Path) -> Path:
    value = (root / relative).resolve()
    value.relative_to(root.resolve())
    return value


def _sealed(root: Path, run: Path) -> tuple[str, dict[str, str]]:
    seal_path = run / "artifacts/evidence_sha256.txt"
    if not seal_path.is_file():
        raise RuntimeError(f"source seal is missing: {run.name}")
    entries: dict[str, str] = {}
    prefix = run.relative_to(root).as_posix() + "/"
    for line in seal_path.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        if relative in entries or not relative.startswith(prefix):
            raise RuntimeError(f"invalid source seal entry: {relative}")
        path = _within(root, relative)
        if not path.is_file() or sha256(path) != digest:
            raise RuntimeError(f"source seal evidence drift: {relative}")
        entries[relative] = digest
    if not entries:
        raise RuntimeError(f"source seal is empty: {run.name}")
    return sha256(seal_path), entries


def _bound_json(root: Path, run: Path, entries: dict[str, str], relative: str) -> dict[str, Any]:
    path = run / relative
    key = path.relative_to(root).as_posix()
    if entries.get(key) != sha256(path):
        raise RuntimeError(f"required source file is not seal-bound: {key}")
    return load_json(path)


def _audit_identity(root: Path, proposal: dict[str, Any]) -> dict[str, Any]:
    run = _within(root, proposal["source_audit_run"])
    seal_digest, entries = _sealed(root, run)
    state = _bound_json(root, run, entries, "RUN_STATE.json")
    summary = _bound_json(root, run, entries, "metrics/summary.json")
    manifest = _bound_json(root, run, entries, "config/source_manifest.json")
    provenance = _bound_json(root, run, entries, "config/source_provenance.json")
    analysis = _bound_json(root, run, entries, "metrics/stochastic_analysis.json")
    if (
        state.get("state") != "COMPLETED"
        or state.get("overall_status") != AUDIT_STATUS
        or summary.get("overall_status") != AUDIT_STATUS
        or summary.get("source_case_count") != 90
        or summary.get("v9_case_count") != 30
        or manifest.get("case_count") != 90
        or manifest.get("block_count") != 10
        or manifest.get("family_case_counts") != {
            "layered_gt_map_oracle": 30,
            "m1d_topology": 30,
            "original_mtare": 30,
        }
        or manifest.get("source_counts") != {
            "failed_source_run": 69, "recovery_run": 21,
        }
        or provenance.get("source_mutation_permitted") is not False
        or analysis.get("case_count") != 90
        or analysis.get("compatibility_bridge", {}).get("mapped_field_count") != 90
    ):
        raise RuntimeError("composed comparison prerequisite is not the exact PASS")
    rows = manifest.get("cases")
    if not isinstance(rows, list) or [row.get("index") for row in rows] != list(range(90)):
        raise RuntimeError("composed comparison prerequisite rows are not exact and ordered")
    if Counter(row.get("source") for row in rows) != Counter({
        "failed_source_run": 69, "recovery_run": 21,
    }):
        raise RuntimeError("composed comparison prerequisite source quota drift")
    return {
        "run": run.relative_to(root).as_posix(),
        "expected_status": AUDIT_STATUS,
        "seal_sha256": seal_digest,
        "schedule_file_sha256": manifest["schedule_file_sha256"],
        "schedule_content_sha256": manifest["schedule_content_sha256"],
        "seal_entry_count": len(entries),
    }


def _corrected_identity(root: Path, proposal: dict[str, Any]) -> dict[str, Any]:
    run = _within(root, proposal["corrected_v5_run"])
    seal_digest, entries = _sealed(root, run)
    state = _bound_json(root, run, entries, "RUN_STATE.json")
    summary = _bound_json(root, run, entries, "metrics/summary.json")
    schedule_path = run / "config/case_schedule.json"
    schedule_key = schedule_path.relative_to(root).as_posix()
    if entries.get(schedule_key) != sha256(schedule_path):
        raise RuntimeError("corrected V5 schedule is not seal-bound")
    schedule = load_json(schedule_path).get("cases")
    if (
        state.get("state") != "COMPLETED"
        or state.get("overall_status") != CORRECTED_STATUS
        or summary.get("overall_status") != CORRECTED_STATUS
        or summary.get("completed_case_count") != 30
        or not isinstance(schedule, list)
        or len(schedule) != 30
    ):
        raise RuntimeError("corrected V5 comparison prerequisite is not the exact PASS")
    ids = [row.get("case_id") for row in schedule]
    blocks = Counter(row.get("block_id") for row in schedule)
    seeds = Counter(row.get("checkpoint_seed") for row in schedule)
    families = Counter(row.get("method_family") for row in schedule)
    if (
        len(set(ids)) != 30
        or len(blocks) != 10
        or set(blocks.values()) != {3}
        or seeds != Counter({0: 10, 1: 10, 2: 10})
        or families != Counter({"m1d_topology": 30})
    ):
        raise RuntimeError("corrected V5 schedule block/checkpoint/family quota drift")
    identity_files = [
        "config/paired_source.json",
        "config/readiness_source.json",
        "config/schedule_audit.json",
    ]
    for relative in identity_files:
        path = run / relative
        key = path.relative_to(root).as_posix()
        if entries.get(key) != sha256(path):
            raise RuntimeError(f"corrected V5 identity evidence is not seal-bound: {relative}")
    return {
        "run": run.relative_to(root).as_posix(),
        "expected_state": "COMPLETED",
        "expected_status": CORRECTED_STATUS,
        "seal_sha256": seal_digest,
        "schedule_relative_path": "config/case_schedule.json",
        "schedule_file_sha256": sha256(schedule_path),
        "schedule_content_sha256": schedule_content_sha256(schedule),
        "identity_files": identity_files,
        "seal_entry_count": len(entries),
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
        raise RuntimeError("final combined corrected comparison card/spec already exists")
    proposal, card_proposal = load_json(proposal_spec_path), load_json(proposal_card_path)
    if proposal.get("status") != PENDING_STATUS or card_proposal.get("status") != PENDING_STATUS:
        raise RuntimeError("combined corrected comparison proposal is not pending")
    for name, item in proposal["frozen_tools"].items():
        if sha256(_within(root, item["path"])) != item["sha256"]:
            raise RuntimeError(f"combined corrected comparison proposal tool drift: {name}")
    source_audit = _audit_identity(root, proposal)
    corrected_v5 = _corrected_identity(root, proposal)

    card = copy.deepcopy(card_proposal)
    card["status"] = "APPROVED_FOR_ONE_FORMAL_EXECUTION"
    card["approval"]["status"] = "APPROVED"
    card["sources"]["source_audit"] = source_audit
    card["sources"]["corrected_v5"] = corrected_v5
    write_json(final_card_path, card)

    final = copy.deepcopy(proposal)
    final.pop("status", None)
    final["user_authorization"]["status"] = "APPROVED"
    final["source_audit"] = source_audit
    final["corrected_v5_run"] = corrected_v5
    final.pop("source_audit_run", None)
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
