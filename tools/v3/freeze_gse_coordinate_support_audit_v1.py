#!/usr/bin/env python3
"""Register exact selected sensor-coordinate audit, not inference or training."""
import argparse
import json
import scipy

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, write_json, preflight
from mtare_topo.governance_inventory import validate_scoped_coordinate_audit_card
from mtare_topo.governance_field_recovery import selection_sha256, INPUT_FIELDS
from run_gse_composition_inventory_v1 import sha
from run_gse_coordinate_support_audit_v1 import KIND


SLUG = "gse_coordinate_support_audit_v1"
CARD = f"configs/v3/gate3/data_cards/{SLUG}.json"
SPEC = f"configs/v3/gate3/{SLUG}.json"
OLD_RUN = "results/gate3_semantics/gate3_20260905_gse_local_teacher_audit_v1_seed0"


def documents():
    spec = load_json(PROJECT_ROOT / "configs/v3/gate3/gse_local_teacher_audit_v1.json")
    card = load_json(PROJECT_ROOT / spec["data_card"])
    recovery = load_json(PROJECT_ROOT / "configs/v3/gate3/gse_composition_field_recovery_v1.json")
    recovery_card = load_json(PROJECT_ROOT / recovery["data_card"])
    if recovery_card["selected_rows"] != card["selected_rows"]:
        raise ValueError("original field-recovery selection drift")
    scope = "Existing same180 C01 observations and900 original frames, raw range/valid and causal relative poses for coordinate-support capacity audit only; zero model/checkpoint/inference/optimizer/new targets. Compare complete raw return pool and900 mean-coordinate pool without GT filtering."
    digest = selection_sha256(card["selected_rows"])
    card.update({"schema_version": "v3_scoped_coordinate_audit_card_v1", "card_id": SLUG,
        "purpose": scope, "diagnostic": KIND, "selection_sha256": digest,
        "sensor_usage": "existing_coordinate_support_only_no_model_or_new_labels",
        "input_fields": list(INPUT_FIELDS), "unique_source_frame_count": 900,
        "frames_per_observation": 5, "parent_count": 10, "node_count": 100,
        "trajectory_metadata": recovery_card["trajectory_metadata"],
        "geometry_scoring_contract": "Keep previous audit row/assignment/error parity; add raw-vs-mean coordinate support lower certificates and convex reconstruction witnesses. No new selected acceptance threshold.",
        "leakage_audit": "Read only the exact selected C01 range/valid chunks plus existing teacher and relative pose fields. Full seal indices are metadata. No C07-C10 scans, checkpoint, model execution, future frames, new supervision or parameter selection.",
        "certificate_policy": "SVD/Qhull proposes geometry only; support bounds recheck all original points; every upper bound is an original-point convex witness; no jitter or retries; noncertified queries remain unresolved.",
        "coordinate_policy": "Existing register_causal_lidar_points and _token_xyz on CPU float32. Formula replay, not a claim of bitwise CUDA coordinate parity. Numeric certificates concern these exact returned coordinates, not measurement noise or physical map truth."})
    card["approval"].update({"scope": scope, "selection_sha256": digest,
        "confirmation_reference": "User-approved GSE-Graph plan and standing execution authority; same-observation causal input/geometry verification and frontend failure separation. New raw-coordinate diagnostic card, not extension of metadata-only approval."})
    card["restrictions"].pop("no_sensor_decoding", None)
    card["restrictions"]["existing_selected_sensor_only"] = True
    prior_seal = f"{OLD_RUN}/artifacts/evidence_sha256.txt"
    prior_hash = "69f17706af3a3dc36f3cd01ca7e9ae5d64b8d60fe906270133f50212c09e4642"
    if sha(PROJECT_ROOT / prior_seal) != prior_hash:
        raise ValueError("previous audit seal drift")
    prior_rows = f"{OLD_RUN}/artifacts/observation_audit.json"
    entries = {}
    for line in (PROJECT_ROOT / prior_seal).read_text().splitlines():
        h, p = line.split(None, 1); entries[p] = h
    card["sealed_sources"].update({prior_seal: prior_hash, prior_rows: entries[prior_rows]})
    report = validate_scoped_coordinate_audit_card(card)
    if not report.passed:
        raise ValueError(report.errors)
    run = f"results/gate3_semantics/gate3_20260905_{SLUG}_seed0"
    spec.update({"slug": SLUG, "data_card": CARD, "config_path": CARD, "diagnostic": KIND,
        "sensor_root": recovery["sensor_root"], "previous_observation_audit": prior_rows,
        "wall_time_cap_s": 600, "expected_scipy_version": scipy.__version__,
        "question": "Does averaging16 elevation rows and4 azimuth columns exclude teacher coordinates that unpooled returns can represent, on exactly the previous180 observations?",
        "method": "Deterministic CPU raw coordinate reprojection and original mean pooling; SVD/hull proposals with original-support lower certificates and nonnegative sum-one witnesses. Preserve all4356 controls and old scoring parity.",
        "baseline": "Same observation full unpooled return support vs old900 mean tokens. These are oracle coordinate-capacity bounds, not learned primitive fitting scores.",
        "fallback": "On Qhull degeneracy use declared weaker valid certificates only, no jitter/retry; unresolved is not pass. Stop on source/certificate/containment/count/resource drift, no changing rows or training.",
        "user_authorization": card["approval"],
        "estimated_cost": {"disk_gb": .05, "wall_time_hours": 1/6, "host_ram_gb": 4,
            "compute": "CPU <=600s,900 existing frames decoded; zero GPU/model/checkpoint/optimizer/new labels"},
        "acceptance_criteria": ["Exactly180 observations/10parents/100 old nodes/900 unique frames/4356 teacher controls; previous row scores/matches identical.",
            "Every bound rechecks original coordinates, witnesses have nonnegative sum-one weights; lower<=upper within numeric bound and raw LB<=mean UB within bound.",
            "All source bytes match seals; exact field/row allowlist,0 C07-C10/model/checkpoint/optimizer/new targets.",
            "Report mean-excluded/raw-witness, both-excluded, both-witness and unresolved populations; no geometric capability claim from a solver flag alone.",
            "<=600s/4GiB; preserve all bounds/certificates, parent summaries, original plots and diagnostic figure, logs/environment/state/seal; no scientificGate PASS or automatic model change."],
        "expected_evidence": ["Exact raw-coordinate card/spec, all-row convex witness/support certificates and original score parity, per-parent count figure, complete XY/XZ, CPU/environment/source hashes, raw row log, RUN_STATE and seal."]})
    spec["source_seals"] += [prior_seal]
    spec["command"][4] = "tools/v3/run_gse_coordinate_support_audit_v1.py"
    spec["command"][6] = str(PROJECT_ROOT / SPEC)
    spec["command"][8] = str(PROJECT_ROOT / run)
    spec["command"][1:1] = ["OPENBLAS_NUM_THREADS=1", "MKL_NUM_THREADS=1"]
    names = set(spec["source_sha256"]) | {"src/mtare_topo/evaluation/gse_coordinate_support.py",
        "tools/v3/run_gse_coordinate_support_audit_v1.py", "tools/v3/freeze_gse_coordinate_support_audit_v1.py",
        "tests/v3/unit/test_gse_coordinate_support.py", "tests/v3/unit/test_gse_coordinate_audit_card.py",
        "tests/v3/unit/test_gse_coordinate_support_runner.py"}
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
    print(json.dumps({"valid_card": True, "observations": 180, "unique_frames": 900, "files_written": 0}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
