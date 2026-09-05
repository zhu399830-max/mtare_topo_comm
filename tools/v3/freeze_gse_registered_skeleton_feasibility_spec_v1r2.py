#!/usr/bin/env python3
"""Freeze the final ERCSS pre-science import corrective V1R2."""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, validate_data_card, validate_run_spec, write_json


SLUG = "gse_registered_skeleton_feasibility_v1r2"
RUN_ID = f"gate3_20260830_{SLUG}_seed0"
SOURCE_CARD = PROJECT_ROOT / "configs/v3/gate3/data_cards/gse_registered_skeleton_feasibility_v1r.json"
SOURCE_SPEC = PROJECT_ROOT / "configs/v3/gate3/gse_registered_skeleton_feasibility_v1r.json"
V1R_RUN = "results/gate3_semantics/gate3_20260829_gse_registered_skeleton_feasibility_v1r_seed0"
SIDECAR = "/tmp/mtare_gate4_meshing_sidecar_v1/bin/python"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    card_path = PROJECT_ROOT / f"configs/v3/gate3/data_cards/{SLUG}.json"
    spec_path = PROJECT_ROOT / f"configs/v3/gate3/{SLUG}.json"
    if card_path.exists() or spec_path.exists(): raise RuntimeError("ERCSS V1R2 freeze is immutable")
    card = copy.deepcopy(load_json(SOURCE_CARD)); card["card_id"] = SLUG
    card["status"] = "APPROVED_FOR_ONE_IMMUTABLE_GSE_REGISTERED_SKELETON_FEASIBILITY_V1R2"
    card["purpose"] = "Final pre-science import corrective: preserve the identical ERCSS audit while lazily loading the existing Torch-backed representation API so the frozen Open3D sidecar can import pure NumPy ERCSS modules without Torch."
    card["approval"] = {
        "status": "APPROVED", "approved_by": "user-standing-authorization",
        "approved_at": "2026-08-30T00:16:00+08:00", "authorized_gates": [3],
        "authorized_operations": ["audit"],
        "scope": "One final V1R2 after V1R failed before world reads because package initialization eagerly imported Torch. No scientific field changes; any further system error stops the audit chain.",
        "confirmation_reference": "User granted autonomous mainline corrective authority; the exception and final-stop condition are explicit in docs/DECISION_LOG.md.",
    }
    card["source"]["second_system_failure_run"] = V1R_RUN
    card["failure_policy"] = "Identical scientific policy. Any V1R2 system error stops the ERCSS formal audit chain. Scientific FAIL stops ERCSS. No further corrective run, threshold change or retry."
    report = validate_data_card(card)
    if not report.passed: raise RuntimeError("invalid ERCSS V1R2 card: " + "; ".join(report.errors))
    write_json(card_path, card)

    spec = copy.deepcopy(load_json(SOURCE_SPEC)); inputs = dict(spec["frozen_inputs"])
    for path in (str(SOURCE_CARD.relative_to(PROJECT_ROOT)), str(SOURCE_SPEC.relative_to(PROJECT_ROOT)), f"{V1R_RUN}/RUN_STATE.json", f"{V1R_RUN}/metrics/summary.json", f"{V1R_RUN}/artifacts/evidence_sha256.txt", "docs/DECISION_LOG.md"):
        inputs[path] = sha256(PROJECT_ROOT / path)
    tools = {name: record for name, record in spec["frozen_tools"].items() if name not in {"card", "runner", "freezer"}}
    for name, path in {
        "card": str(card_path.relative_to(PROJECT_ROOT)),
        "runner": "tools/v3/run_gse_registered_skeleton_feasibility_v1r2.py",
        "freezer": "tools/v3/freeze_gse_registered_skeleton_feasibility_spec_v1r2.py",
        "lazy_representation_package": "src/mtare_topo/representation/__init__.py",
    }.items(): tools[name] = {"path": path, "sha256": sha256(PROJECT_ROOT / path)}
    spec.update({
        "date": "20260830", "slug": SLUG,
        "question": spec["question"] + " (final V1R2 lazy-import corrective)",
        "data_card": str(card_path.relative_to(PROJECT_ROOT)), "config_path": str(card_path.relative_to(PROJECT_ROOT)),
        "user_authorization": card["approval"], "frozen_inputs": inputs, "frozen_tools": tools,
        "system_corrective": "V1R imported the unchanged evaluator in the Open3D sidecar, but representation package initialization eagerly imported Torch before any world read. V1R2 lazily resolves the same public Torch symbols; sidecar no-Torch import and Torch public-API contracts both pass. This is the final allowed system corrective.",
        "command": ["/usr/bin/timeout", "10800s", SIDECAR, "tools/v3/run_gse_registered_skeleton_feasibility_v1r2.py", "--spec", str(spec_path), "--run-dir", str(PROJECT_ROOT / "results/gate3_semantics" / RUN_ID)],
    })
    report = validate_run_spec(spec)
    if not report.passed: raise RuntimeError("invalid ERCSS V1R2 spec: " + "; ".join(report.errors))
    write_json(spec_path, spec); print(card_path.relative_to(PROJECT_ROOT)); print(spec_path.relative_to(PROJECT_ROOT)); return 0


if __name__ == "__main__": raise SystemExit(main())
