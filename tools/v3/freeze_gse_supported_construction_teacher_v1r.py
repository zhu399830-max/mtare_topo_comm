#!/usr/bin/env python3
"""Separately freeze the original-storage precision corrective; no execution."""
from copy import deepcopy
import json
from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import build_run_id, load_json, preflight, write_json
from mtare_topo.governance_supported_teacher import validate_supported_teacher_card
from freeze_gse_supported_construction_teacher_v1 import documents as original_documents
from run_gse_supported_construction_teacher_v1 import sha

SLUG = "gse_supported_construction_teacher_v1r"
CARD = f"configs/v3/gate3/data_cards/{SLUG}.json"
SPEC = f"configs/v3/gate3/{SLUG}.json"
PREVIOUS = "results/gate3_semantics/gate3_20260905_gse_supported_construction_teacher_v1_seed0"


def documents():
    card, spec = original_documents()
    if load_json(PROJECT_ROOT / PREVIOUS / "RUN_STATE.json")["state"] != "FAILED":
        raise ValueError("requires preserved failed V1, never overwrite/retry it")
    seal_path = PREVIOUS + "/artifacts/evidence_sha256.txt"
    if sha(PROJECT_ROOT / seal_path) != "cc21ecd73b20beb730f91edd96209409fd4c9cbc86b55980b01eb8dda7a92f82":
        raise ValueError("failed V1 evidence seal drift")
    card["card_id"] = SLUG
    card["purpose"] += " V1R corrects only float64 odometry to original float32 storage before exact parity; no tolerance change."
    card["sealed_sources"][seal_path] = sha(PROJECT_ROOT / seal_path)
    card["sealed_sources"]["configs/v3/gate3/gse_supported_construction_teacher_v1.json"] = sha(PROJECT_ROOT / "configs/v3/gate3/gse_supported_construction_teacher_v1.json")
    card["approval"]["scope"] = card["purpose"]
    card["approval"]["confirmation_reference"] += " Standing in-scope software corrective, recorded in DECISION_LOG; original code preserved by Git commit 256a059. No newly claimed user exchange."
    spec.update(slug=SLUG, data_card=CARD, config_path=CARD, user_authorization=deepcopy(card["approval"]))
    spec["storage_precision_corrective"] = {"previous_run": PREVIOUS, "previous_source_git_commit": "256a059",
        "change": "Reproduce original P1b float32 storage cast before array_equal, reject even one stored ULP drift; unchanged semantic rules and population."}
    spec["command"][-3] = str(PROJECT_ROOT / SPEC)
    spec["command"][-1] = str(PROJECT_ROOT / "results/gate3_semantics" / build_run_id(spec))
    source = "tools/v3/freeze_gse_supported_construction_teacher_v1r.py"
    spec["source_sha256"][source] = sha(PROJECT_ROOT / source)
    report = validate_supported_teacher_card(card)
    if not report.passed:
        raise ValueError(report.errors)
    return card, spec


def main():
    if (PROJECT_ROOT / CARD).exists() or (PROJECT_ROOT / SPEC).exists():
        raise RuntimeError("refuse overwrite of frozen corrective")
    card, spec = documents()
    write_json(PROJECT_ROOT / CARD, card); write_json(PROJECT_ROOT / SPEC, spec)
    report = preflight(spec, load_json(PROJECT_ROOT / "results/project_status.json"), PROJECT_ROOT)
    print(json.dumps({"passed": report.passed, "errors": report.errors, "warnings": report.warnings}))
    return int(not report.passed)


if __name__ == "__main__":
    raise SystemExit(main())
