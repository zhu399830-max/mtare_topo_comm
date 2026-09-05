#!/home/zeng-workstation/anaconda3/bin/python
"""Bind the V5 live probe to the sealed V1R composed audit selection."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, write_json
import run_aee_composite_v9_combined_correction_probe_v1r as probe_wrapper
from run_mtare_single_robot_stochastic_v1 import sha256


probe = probe_wrapper.base
PROPOSAL_SPEC = Path("configs/v3/gate6/aee_composite_v9_combined_correction_probe_v1r.proposal.json")
PROPOSAL_CARD = Path("configs/v3/gate6/data_cards/aee_composite_v9_combined_correction_probe_v1r.proposal.json")
FINAL_SPEC = Path("configs/v3/gate6/aee_composite_v9_combined_correction_probe_v1r.json")
FINAL_CARD = Path("configs/v3/gate6/data_cards/aee_composite_v9_combined_correction_probe_v1r.json")


def _within(root: Path, relative: str | Path) -> Path:
    value = (root / relative).resolve()
    value.relative_to(root.resolve())
    return value


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
        raise RuntimeError("final V1R combined-correction probe card/spec already exists")
    proposal = load_json(proposal_spec_path)
    card_proposal = load_json(proposal_card_path)
    pending = "PENDING_MECHANISM_AUDIT_V1R_SEAL_NOT_EXECUTABLE"
    if proposal.get("status") != pending or card_proposal.get("status") != pending:
        raise RuntimeError("V1R combined-correction probe is not a pending proposal")
    for name, item in proposal["frozen_tools"].items():
        if sha256(_within(root, item["path"])) != item["sha256"]:
            raise RuntimeError(f"V1R combined-correction probe tool drift: {name}")

    audit_root = _within(root, proposal["mechanism_audit_run"])
    seal_path = audit_root / "artifacts/evidence_sha256.txt"
    if not seal_path.is_file():
        raise RuntimeError("V1R mechanism audit has no final seal")
    audit_seal = sha256(seal_path)
    bound = copy.deepcopy(proposal)
    bound["mechanism_audit_seal_sha256"] = audit_seal
    probe.AUDIT_STATUS_PASS = probe_wrapper.AUDIT_STATUS_PASS
    source = probe.validate_composed_audit_source(bound, project_root=root)
    selected = source["selected_source_case"]
    probe_case = {
        "case_id": (
            f"{selected['world']}_env{selected['environment_seed']}_"
            f"m1d_seed{selected['checkpoint_seed']}_combined_v5_probe"
        ),
        "source_case_id": selected["case_id"],
        "world": selected["world"],
        "environment_seed": selected["environment_seed"],
        "checkpoint_seed": selected["checkpoint_seed"],
        "runtime_sec": 180.0,
    }
    probe.validate_probe_case(probe_case, selected)

    card = copy.deepcopy(card_proposal)
    card["status"] = "APPROVED_FOR_ONE_FORMAL_EXECUTION"
    card["approval"]["status"] = "APPROVED"
    card["sources"]["mechanism_audit_seal_sha256"] = audit_seal
    card["sampling"]["selected_source_case"] = selected
    card["sampling"]["live_probe_case"] = probe_case
    write_json(final_card_path, card)

    final = copy.deepcopy(proposal)
    final.pop("status", None)
    final["user_authorization"]["status"] = "APPROVED"
    final["mechanism_audit_seal_sha256"] = audit_seal
    final["probe_case"] = probe_case
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
