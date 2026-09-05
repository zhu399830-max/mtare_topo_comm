#!/usr/bin/env python3
"""Freeze bounded error decomposition; preserve all old scoring assignments."""
import argparse
import json

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, write_json, preflight
from mtare_topo.governance_inventory import validate_scoped_inventory_card
from run_gse_composition_inventory_v1 import sha
from run_gse_axis_error_decomposition_v1 import KIND


SLUG = "gse_axis_error_decomposition_v1"
CARD = f"configs/v3/gate3/data_cards/{SLUG}.json"
SPEC = f"configs/v3/gate3/{SLUG}.json"
OLD_RUN = "results/gate3_semantics/gate3_20260905_gse_local_teacher_audit_v1_seed0"


def documents():
    spec = load_json(PROJECT_ROOT / "configs/v3/gate3/gse_local_teacher_audit_v1.json")
    card = load_json(PROJECT_ROOT / spec["data_card"])
    scope = "Same180 C01 observations, zero new inference or labels: reproduce sealed previous scoring rows exactly, then separate axial/transverse/direction/crop errors using the existing teacher and cached32-slot prediction geometry."
    card.update({"card_id": SLUG, "purpose": scope, "diagnostic": KIND, "new_scoring_thresholds": False,
        "geometry_scoring_contract": "Previous180 audit rows and all1452 Hungarian assignments must reproduce exactly. No new assignment, gate or selected subset; diagnostic float64 errors added alongside immutable old float32 scores.",
        "additional_metadata": "Previous sealed180-row audit (matches, score, provenance); no new sensor, sidecar or checkpoint read.",
        "error_decomposition": {"quadrature_samples_per_direction": 256,
            "curve": "two-segment interpolant of three controls, not original spline",
            "unknown_chord": "<=2sqrt(3)*max input dtype ULP; numerical resolution only, not fit threshold",
            "symmetric_mean_error_bound": "(teacher_polyline_length+prediction_polyline_length)/(4*256)"}})
    card["approval"]["scope"] = scope
    oldseal = f"{OLD_RUN}/artifacts/evidence_sha256.txt"
    expected = "69f17706af3a3dc36f3cd01ca7e9ae5d64b8d60fe906270133f50212c09e4642"
    if sha(PROJECT_ROOT / oldseal) != expected:
        raise ValueError("previous audit seal drift")
    card["sealed_sources"][oldseal] = expected
    prior_rows = f"{OLD_RUN}/artifacts/observation_audit.json"
    entries = {}
    for line in (PROJECT_ROOT / oldseal).read_text().splitlines():
        digest, path = line.split(None, 1); entries[path] = digest
    card["sealed_sources"][prior_rows] = entries[prior_rows]
    validation = validate_scoped_inventory_card(card)
    if not validation.passed:
        raise ValueError(validation.errors)
    run = f"results/gate3_semantics/gate3_20260905_{SLUG}_seed0"
    spec.update({"slug": SLUG, "data_card": CARD, "config_path": CARD, "diagnostic": KIND,
        "quadrature_samples": 256, "previous_observation_audit": prior_rows,
        "question": "Do the large cropped control-point errors mainly reflect axial extent/cropping, or actual transverse/direction/layout errors?",
        "method": "Fixed previous scoring assignment and all1452 fragments; orthogonal chord decomposition, lengths and bidirectional analytic finite-polyline distance with bounded midpoint quadrature; per-parent/incident split for analysis only.",
        "baseline": "Exact reproduction of all previous180 rows and old float32 point/parameter errors. New decomposition is diagnostic, not a replacement score or a new benchmark.",
        "fallback": "Stop on any row/assignment/source drift or violated decomposition identity. Unresolved numerical directions remain None with counts; no sample deletion, threshold selection or training.",
        "user_authorization": card["approval"],
        "estimated_cost": {"disk_gb": .03, "wall_time_hours": 1/30, "host_ram_gb": 4,
            "compute": "CPU <=120s; cached predictions and same180 teacher only; 0 scans/forward/optimizer/new labels/GPU"},
        "acceptance_criteria": ["All180 previous rows, source identities, assignments and float32 scores reproduce exactly; no new matching or selection.",
            "All1452 fragments retained; orthogonal energy identity within float64 precision, finite lengths/distances, explicit unresolved direction counts.",
            "Same180/10parents/100 old nodes/900 referenced frames; all bytes sealed; no C07-C10/sensor/checkpoint/optimizer/new labels.",
            "<=120s CPU/4GiB; preserve old scores and per-row/per-parent decompositions, all-row XY/XZ and scatter, raw log, metadata, RUN_STATE/seal.",
            "Do not declare scientific PASS, transverse-node localization error from control errors, or training readiness from this diagnosis."],
        "expected_evidence": ["Same scope card/spec, previous-row parity, raw task logs, all-row/all-parent decomposition, unknown counts, integration error bounds, all180 XY/XZ, axial/transverse scatter, source/tool hashes and immutable seal."]})
    spec["source_seals"] = spec["source_seals"] + [oldseal]
    spec["command"] = list(spec["command"])
    spec["command"][4] = "tools/v3/run_gse_axis_error_decomposition_v1.py"
    spec["command"][6] = str(PROJECT_ROOT / SPEC)
    spec["command"][8] = str(PROJECT_ROOT / run)
    names = set(spec["source_sha256"]) | {
        "src/mtare_topo/evaluation/gse_axis_error_decomposition.py",
        "tools/v3/run_gse_axis_error_decomposition_v1.py", "tools/v3/freeze_gse_axis_error_decomposition_v1.py",
        "tests/v3/unit/test_gse_axis_error_decomposition.py", "tests/v3/unit/test_gse_axis_error_decomposition_runner.py"}
    spec["source_sha256"] = {p: sha(PROJECT_ROOT / p) for p in sorted(names)}
    return card, spec


def main():
    parser = argparse.ArgumentParser(__doc__); parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    if (PROJECT_ROOT / CARD).exists() or (PROJECT_ROOT / SPEC).exists():
        raise RuntimeError("card/spec already exists; no overwrite")
    card, spec = documents()
    if args.freeze:
        write_json(PROJECT_ROOT / CARD, card); write_json(PROJECT_ROOT / SPEC, spec)
        report = preflight(spec, load_json(PROJECT_ROOT / "results/project_status.json"), PROJECT_ROOT)
        print(json.dumps({"passed": report.passed, "errors": report.errors, "card": CARD, "spec": SPEC}))
        return int(not report.passed)
    print(json.dumps({"valid_card": True, "observations": 180, "files_written": 0}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
