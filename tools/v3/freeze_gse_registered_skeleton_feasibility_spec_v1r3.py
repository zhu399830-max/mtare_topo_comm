#!/usr/bin/env python3
"""Prepare/freeze the ERCSS V1R3 tool-hash-only corrective.

The script is safe to run with ``--verify-only`` before authorization.  Actual
Data Card/spec creation requires the explicit override token documented below.
Every inherited tool digest is recomputed from its current path; no digest is
copied from V1R2.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, validate_data_card, validate_run_spec, write_json


SLUG = "gse_registered_skeleton_feasibility_v1r3"
RUN_ID = f"gate3_20260830_{SLUG}_seed0"
SOURCE_CARD = PROJECT_ROOT / "configs/v3/gate3/data_cards/gse_registered_skeleton_feasibility_v1r2.json"
SOURCE_SPEC = PROJECT_ROOT / "configs/v3/gate3/gse_registered_skeleton_feasibility_v1r2.json"
V1R2_RUN = "results/gate3_semantics/gate3_20260830_gse_registered_skeleton_feasibility_v1r2_seed0"
SIDECAR = "/tmp/mtare_gate4_meshing_sidecar_v1/bin/python"
EXPLICIT_OVERRIDE = "USER_EXPLICITLY_ALLOWED_ERCSS_V1R3"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def current_tool_records(source_spec: dict) -> dict[str, dict[str, str]]:
    paths = {
        name: str(record["path"])
        for name, record in source_spec["frozen_tools"].items()
        if name not in {"card", "runner", "freezer"}
    }
    paths.update({
        "card": f"configs/v3/gate3/data_cards/{SLUG}.json",
        "runner": "tools/v3/run_gse_registered_skeleton_feasibility_v1r3.py",
        "freezer": "tools/v3/freeze_gse_registered_skeleton_feasibility_spec_v1r3.py",
        "lazy_representation_package": "src/mtare_topo/representation/__init__.py",
    })
    missing = [path for path in paths.values() if not (PROJECT_ROOT / path).is_file() and path != paths["card"]]
    if missing:
        raise RuntimeError(f"V1R3 tool path missing: {missing}")
    return {
        name: {"path": path, "sha256": "PENDING_CARD" if name == "card" else sha256(PROJECT_ROOT / path)}
        for name, path in sorted(paths.items())
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--explicit-override", default="")
    args = parser.parse_args()
    source_spec = load_json(SOURCE_SPEC)
    tools = current_tool_records(source_spec)
    stale = []
    for name, old in source_spec["frozen_tools"].items():
        if name in tools and tools[name]["sha256"] != old["sha256"]:
            stale.append({"name": name, "path": old["path"], "old_sha256": old["sha256"], "current_sha256": tools[name]["sha256"]})
    if args.verify_only:
        print(json.dumps({"passed": True, "tool_count": len(tools), "stale_in_v1r2": stale, "writes": 0}, indent=2))
        return 0
    if args.explicit_override != EXPLICIT_OVERRIDE:
        raise RuntimeError("V1R3 freeze requires the explicit user governance override token")

    card_path = PROJECT_ROOT / f"configs/v3/gate3/data_cards/{SLUG}.json"
    spec_path = PROJECT_ROOT / f"configs/v3/gate3/{SLUG}.json"
    if card_path.exists() or spec_path.exists(): raise RuntimeError("ERCSS V1R3 freeze is immutable")
    card = copy.deepcopy(load_json(SOURCE_CARD)); card["card_id"] = SLUG
    card["status"] = "APPROVED_FOR_ONE_IMMUTABLE_GSE_REGISTERED_SKELETON_FEASIBILITY_V1R3"
    card["purpose"] = "Explicitly authorized final tool-hash-only corrective after V1R2 inherited a stale unit-test digest. Scientific population, Teacher, method, thresholds, capacity and evidence are unchanged."
    card["approval"] = {
        "status": "APPROVED", "approved_by": "user-explicit-v1r3-override",
        "approved_at": "2026-08-30T00:25:00+08:00", "authorized_gates": [3],
        "authorized_operations": ["audit"],
        "scope": "Exactly one V1R3 that recomputes every current tool SHA. No scientific contract change and no further corrective run.",
        "confirmation_reference": EXPLICIT_OVERRIDE,
    }
    card["source"]["third_system_failure_run"] = V1R2_RUN
    card["failure_policy"] = "Identical scientific policy. Any V1R3 system or scientific failure stops ERCSS. No V1R4, retry, threshold change or alternate population."
    report = validate_data_card(card)
    if not report.passed: raise RuntimeError("invalid ERCSS V1R3 card: " + "; ".join(report.errors))
    write_json(card_path, card)
    tools["card"]["sha256"] = sha256(card_path)

    spec = copy.deepcopy(source_spec); inputs = dict(spec["frozen_inputs"])
    for path in (
        str(SOURCE_CARD.relative_to(PROJECT_ROOT)), str(SOURCE_SPEC.relative_to(PROJECT_ROOT)),
        f"{V1R2_RUN}/RUN_STATE.json", f"{V1R2_RUN}/metrics/summary.json", f"{V1R2_RUN}/artifacts/evidence_sha256.txt",
    ):
        inputs[path] = sha256(PROJECT_ROOT / path)
    spec.update({
        "date": "20260830", "slug": SLUG,
        "question": source_spec["question"] + " (explicit V1R3 tool-hash-only corrective)",
        "data_card": str(card_path.relative_to(PROJECT_ROOT)), "config_path": str(card_path.relative_to(PROJECT_ROOT)),
        "user_authorization": card["approval"], "frozen_inputs": inputs, "frozen_tools": tools,
        "system_corrective": "V1R2 copied a pre-edit test digest. V1R3 recomputes every tool digest from its current path after 10/10 unit, sidecar-no-Torch-import and Torch-public-API checks; no scientific field changes.",
        "command": ["/usr/bin/timeout", "10800s", SIDECAR, "tools/v3/run_gse_registered_skeleton_feasibility_v1r3.py", "--spec", str(spec_path), "--run-dir", str(PROJECT_ROOT / "results/gate3_semantics" / RUN_ID)],
    })
    report = validate_run_spec(spec)
    if not report.passed: raise RuntimeError("invalid ERCSS V1R3 spec: " + "; ".join(report.errors))
    write_json(spec_path, spec); print(card_path.relative_to(PROJECT_ROOT)); print(spec_path.relative_to(PROJECT_ROOT)); return 0


if __name__ == "__main__": raise SystemExit(main())
