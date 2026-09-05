#!/usr/bin/env python3
"""Freeze the ERCSS V1R environment-only corrective."""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, validate_data_card, validate_run_spec, write_json


SLUG = "gse_registered_skeleton_feasibility_v1r"
RUN_ID = f"gate3_20260829_{SLUG}_seed0"
V1_CARD = PROJECT_ROOT / "configs/v3/gate3/data_cards/gse_registered_skeleton_feasibility_v1.json"
V1_SPEC = PROJECT_ROOT / "configs/v3/gate3/gse_registered_skeleton_feasibility_v1.json"
V1_RUN = "results/gate3_semantics/gate3_20260829_gse_registered_skeleton_feasibility_v1_seed0"
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
    if card_path.exists() or spec_path.exists():
        raise RuntimeError("ERCSS V1R freeze is immutable")
    card = copy.deepcopy(load_json(V1_CARD))
    card["card_id"] = SLUG
    card["status"] = "APPROVED_FOR_ONE_IMMUTABLE_GSE_REGISTERED_SKELETON_FEASIBILITY_V1R"
    card["purpose"] = "Environment-only corrective for the sealed V1 system failure: execute the identical ERCSS population, Teacher, visibility, thresholds and evidence contract with the already frozen Open3D 0.19.0 meshing sidecar used by prior native-mesh LOS proofs."
    card["approval"] = {
        "status": "APPROVED", "approved_by": "user-standing-authorization",
        "approved_at": "2026-08-30T00:08:00+08:00", "authorized_gates": [3],
        "authorized_operations": ["audit"],
        "scope": "Exactly one V1R environment corrective after V1 failed before world reads because the Torch environment lacked Open3D. Scientific data/method/threshold/capacity/evidence contracts are byte-for-byte inherited.",
        "confirmation_reference": "User instructed autonomous optimal continuation; V1 formal system evidence identifies the missing package before any scientific work.",
    }
    card["source"]["system_failure_run"] = V1_RUN
    card["source"]["sidecar_contract"] = "configs/v3/gate4/environments/gate4_meshing_sidecar_v1.json"
    card["failure_policy"] = "Same scientific failure policy as V1. In addition, any sidecar identity/version drift or nonzero V1 scientific population seals system FAIL. No retry beyond this V1R."
    report = validate_data_card(card)
    if not report.passed:
        raise RuntimeError("invalid ERCSS V1R card: " + "; ".join(report.errors))
    write_json(card_path, card)

    v1 = load_json(V1_SPEC)
    frozen_inputs = dict(v1["frozen_inputs"])
    for path in (
        str(V1_CARD.relative_to(PROJECT_ROOT)), str(V1_SPEC.relative_to(PROJECT_ROOT)),
        f"{V1_RUN}/RUN_STATE.json", f"{V1_RUN}/metrics/summary.json", f"{V1_RUN}/artifacts/evidence_sha256.txt",
        "configs/v3/gate4/environments/gate4_meshing_sidecar_v1.json",
    ):
        frozen_inputs[path] = sha256(PROJECT_ROOT / path)
    tools = {
        name: record for name, record in v1["frozen_tools"].items()
        if name not in {"card", "runner", "freezer"}
    }
    for name, path in {
        "card": str(card_path.relative_to(PROJECT_ROOT)),
        "runner": "tools/v3/run_gse_registered_skeleton_feasibility_v1r.py",
        "freezer": "tools/v3/freeze_gse_registered_skeleton_feasibility_spec_v1r.py",
    }.items():
        tools[name] = {"path": path, "sha256": sha256(PROJECT_ROOT / path)}
    spec = copy.deepcopy(v1)
    spec.update({
        "slug": SLUG, "question": v1["question"] + " (V1R environment-only corrective)",
        "data_card": str(card_path.relative_to(PROJECT_ROOT)), "config_path": str(card_path.relative_to(PROJECT_ROOT)),
        "user_authorization": card["approval"], "frozen_inputs": frozen_inputs, "frozen_tools": tools,
        "system_corrective": "V1 used the Torch/Zarr model environment, which lacked Open3D and failed before reading worlds. V1R runs unit tests in that existing test environment and the unchanged evaluator in the frozen Open3D 0.19.0 meshing sidecar; no scientific field changes.",
        "command": ["/usr/bin/timeout", "10800s", SIDECAR, "tools/v3/run_gse_registered_skeleton_feasibility_v1r.py", "--spec", str(spec_path), "--run-dir", str(PROJECT_ROOT / "results/gate3_semantics" / RUN_ID)],
    })
    report = validate_run_spec(spec)
    if not report.passed:
        raise RuntimeError("invalid ERCSS V1R spec: " + "; ".join(report.errors))
    write_json(spec_path, spec)
    print(card_path.relative_to(PROJECT_ROOT)); print(spec_path.relative_to(PROJECT_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
