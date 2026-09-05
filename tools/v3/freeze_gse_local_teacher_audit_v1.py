#!/usr/bin/env python3
"""Freeze exact same-row existing-teacher/cache audit; no source-array reads."""
import argparse
import json

from _bootstrap import PROJECT_ROOT
from mtare_topo.data.gse_local_teacher_audit import FIELDS, PREDICTION_FIELDS
from mtare_topo.governance import load_json, write_json, preflight
from mtare_topo.governance_inventory import validate_scoped_inventory_card
from run_gse_composition_inventory_v1 import sha


SLUG = "gse_local_teacher_audit_v1"
CARD = f"configs/v3/gate3/data_cards/{SLUG}.json"
SPEC = f"configs/v3/gate3/{SLUG}.json"
RECOVERY = "results/gate3_semantics/gate3_20260905_gse_composition_field_recovery_v1_seed0"
PYTHON = "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python"


def documents():
    previous = load_json(PROJECT_ROOT / "configs/v3/gate3/gse_composition_inventory_v1r.json")
    card = load_json(PROJECT_ROOT / previous["data_card"])
    card["card_id"] = SLUG
    scope = "Read existing visible geometry and all visible construction relations for the identical 180 C01 observations; compare already sealed six-field predictions without executing/loading a model or checkpoint; no new targets or fitting."
    card["purpose"] = scope
    card["approval"].update({"scope": scope,
        "confirmation_reference": "User-approved GSE-Graph composition implementation plan: exact-180 teacher-boundary and true/predicted geometry checks; standing authorization for in-scope work. New audit scope, not reuse of the old metadata-only authorization."})
    card["teacher_source"] = "Existing P1b cropped visible primitive geometry, support and GT construction incidence only. Physical endpoints are not cropped endpoints. Angular projection overlap is not physical overlap. GT identity used solely by scorer, never to filter predicted inputs."
    card["existing_prediction_cache_read_only"] = True
    card["no_model_or_checkpoint_interpretation"] = "No model instantiation, checkpoint load or forward execution. Previously exported six-field NPZ arrays are explicitly authorized read-only audit evidence."
    card["allowed_teacher_fields"] = sorted(FIELDS)
    card["allowed_cache_fields"] = list(PREDICTION_FIELDS)
    card["geometry_scoring_contract"] = "Reuse legacy Hungarian normalized geometry cost and relation-aware tie ordering for scoring only; retain all32 predicted slots, report physical errors/confidence, select no acceptance threshold."
    card["event_target_contract"] = "No labels generated. Neither degree, fragment presence, angular overlap nor old endpoint support proves complete visible events."
    card["unique_source_frames_referenced_not_decoded"] = 900
    card["raw_observations"] = 180
    card["effective_independent_parents"] = 10
    card["node_identities_scoring_only"] = 100
    card["leakage_audit"] = "Exact existing C01 mixed tasks/row indices only; no scans/checkpoints/C07-C10 read, no future frames, new annotation, optimization, threshold/checkpoint selection or deployment graph."
    # Old source hashes remain checked; remove unused endpoint sidecar so it is
    # not reopened and cannot silently turn into the new event teacher.
    card["sealed_sources"] = {p: h for p, h in card["sealed_sources"].items()
        if "primitive_attachment_observability_sidecar" not in p}
    recovery_seal = f"{RECOVERY}/artifacts/evidence_sha256.txt"
    card["sealed_sources"][recovery_seal] = "478e0744a7abaf5281342722a037b4aa1267f23b9b89377f5dd67415fed35481"
    for p, h in card["sealed_sources"].items():
        if sha(PROJECT_ROOT / p) != h:
            raise ValueError(f"frozen source metadata drift: {p}")
    # Names/hashes only, no array archive decoded at registration time.
    tasks = sorted({r["task"] for r in card["selected_rows"]})
    entries = {}
    for line in (PROJECT_ROOT / recovery_seal).read_text().splitlines():
        digest, relative = line.split(None, 1); entries[relative] = digest
    cache_paths = [f"{RECOVERY}/artifacts/shared_frame_predictions/{task}.npz" for task in tasks]
    cache_paths += [f"{RECOVERY}/artifacts/prediction_manifest.json"]
    for path in cache_paths:
        if path not in entries:
            raise ValueError("selected recovery cache absent from seal")
        card["sealed_sources"][path] = entries[path]
    if not validate_scoped_inventory_card(card).passed:
        raise ValueError(validate_scoped_inventory_card(card).errors)
    source_files = sorted(str(p.relative_to(PROJECT_ROOT)) for p in (PROJECT_ROOT / "src/mtare_topo").rglob("*.py"))
    source_files += ["tools/v3/_bootstrap.py", "tools/v3/preflight.py", "tools/v3/create_run.py",
        "tools/v3/run_gse_composition_inventory_v1.py", "tools/v3/run_gse_local_teacher_audit_v1.py",
        "tools/v3/freeze_gse_local_teacher_audit_v1.py", "tests/v3/unit/test_gse_local_teacher_audit.py",
        "tests/v3/unit/test_gse_local_teacher_audit_runner.py"]
    run = f"results/gate3_semantics/gate3_20260905_{SLUG}_seed0"
    spec = {"schema_version": "v3_run_spec_v1", "gate": 3, "date": "20260905", "slug": SLUG, "seed": 0,
        "operation": "audit", "question": "Which existing geometry/support/incidence facts are valid, and what are the actual prediction errors, without pretending construction identity is an observable event?",
        "method": "Same180 read-only teacher vs all32 cached predictions; full visible GT-incidence parity; unchanged legacy Hungarian scoring; all-row XY/XZ evidence.",
        "baseline": "Existing cropped teacher geometry as scoring oracle and immutable legacy incidence; not a performance comparison to another learning method.",
        "fallback": "Stop on source/shape/incidence/causality drift. Inconclusive observability remains UNKNOWN; do not change rows, create labels, tune thresholds or run training.",
        "user_authorization": card["approval"], "data_card": CARD, "config_path": CARD,
        "command": ["env", "OMP_NUM_THREADS=1", "PYTHONHASHSEED=0", PYTHON,
            "tools/v3/run_gse_local_teacher_audit_v1.py", "--spec", str(PROJECT_ROOT / SPEC), "--run-dir", str(PROJECT_ROOT / run)],
        "estimated_cost": {"disk_gb": .03, "wall_time_hours": 1/30, "host_ram_gb": 4,
            "compute": "CPU <=120s bounded existing teacher/cache audit; zero GPU, inference, optimizer, scan decoding or new labels"},
        "expected_versions": {"python": "3.13.5", "numpy": "2.1.3", "torch": "2.9.0+cu129", "zarr": "2.18.7"},
        "selection_manifest": previous["selection_manifest"], "teacher_root": previous["teacher_root"],
        "construction_root": previous["construction_root"], "prediction_root": f"{RECOVERY}/artifacts/shared_frame_predictions",
        "prediction_manifest": f"{RECOVERY}/artifacts/prediction_manifest.json",
        "source_seals": previous["source_seals"][:2] + [recovery_seal],
        "source_sha256": {p: sha(PROJECT_ROOT / p) for p in source_files},
        "acceptance_criteria": ["Exactly180 observations/10 parents/100 old node identities/900 referenced frames, no scans decoded.",
            "All accessed bytes match source seals; fragment mask/count/temporal parity; stored relations match ALL visible construction members.",
            "No GT identity in prediction filtering; all32 cached slots retained; report oracle-aligned physical errors and confidence, not detections.",
            "All180 XY/XZ previews; per-row/per-parent metrics, scope limitations and provenance; no event-label or scientific-Gate promotion.",
            "<=120s CPU/4GiB RAM;0 model execution/optimizer/new targets/C07-C10; immutable output and seal."],
        "expected_evidence": ["Card/spec snapshots, environment, raw task log, all-row geometry and membership metrics, ten all-row XY/XZ SVG sheets, source read hashes, RUN_STATE and SHA256 seal."]}
    return card, spec


def main():
    parser = argparse.ArgumentParser(__doc__); parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    if (PROJECT_ROOT / CARD).exists() or (PROJECT_ROOT / SPEC).exists():
        raise RuntimeError("card/spec already exists; no overwrite")
    card, spec = documents()
    if args.freeze:
        write_json(PROJECT_ROOT / CARD, card); write_json(PROJECT_ROOT / SPEC, spec)
        result = preflight(spec, load_json(PROJECT_ROOT / "results/project_status.json"), PROJECT_ROOT)
        print(json.dumps({"passed": result.passed, "errors": result.errors, "card": CARD, "spec": SPEC}))
        return int(not result.passed)
    print(json.dumps({"valid_card": True, "observations": 180, "source_files": len(spec["source_sha256"]), "files_written": 0}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
