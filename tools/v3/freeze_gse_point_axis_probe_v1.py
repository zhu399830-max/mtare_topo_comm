#!/usr/bin/env python3
"""Freeze the exact same180 geometry-only paired training contract."""
from copy import deepcopy
import json
import scipy
import matplotlib
from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, write_json, preflight
from mtare_topo.governance_point_axis import (
    SCHEMA, TRAINING, GEOMTEACHER_FIELDS, RESTRICTIONS, validate_point_axis_training_card,
)
from run_gse_composition_field_recovery_v1 import sha

SLUG="gse_point_axis_probe_v1"
CARD=f"configs/v3/gate3/data_cards/{SLUG}.json"
SPEC=f"configs/v3/gate3/{SLUG}.json"
RECOVERY="results/gate3_semantics/gate3_20260905_gse_composition_field_recovery_v1_seed0"


def documents():
    old=load_json(PROJECT_ROOT/"configs/v3/gate3/gse_composition_field_recovery_v1.json")
    card=load_json(PROJECT_ROOT/old["data_card"])
    for key in ("inference","output_fields"): card.pop(key)
    card.update({"schema_version":SCHEMA,"card_id":SLUG,
        "purpose":"Exact C01 same180 geometry-only paired fit probe; frozen seed0 backbone, learn point-axis readout and matched raw no-offset ablation. No event or point-identity targets.",
        "teacher_source":"Existing sealed P1b visible cropped axis_control_current_sensor_m and primitive_mask only; frame/source indices verify alignment. No teacher construction ID, event, endpoint-neighbor or point-ownership target enters forward/loss.",
        "restrictions":dict.fromkeys(RESTRICTIONS,True),"training":deepcopy(TRAINING),"geomteacher_fields":list(GEOMTEACHER_FIELDS),
        "leakage_audit":"Only exact180 selected C01 windows and900 unique source frames; same10 topology parents. No C07-C10 or M-TARE; no absolute pose/node/tunnel identity, future frame or teacher candidate filtering in student. Geometry matching is training/scoring only; no generalization claim.",
        "supervision_boundary":"Only existing axes/masks, reversal-invariant geometry Hungarian matching. Membership/offset receive indirect geometry loss, not new point labels. No detector precision or calibrated port confidence claim.",
        "matching_population":"All1452 existing visible fragments and all32 predictions per observation; unmatched predictions counted, not discarded from saved arrays/previews. Pure geometry matching differs from legacy multi-field Hungarian scoring; legacy axes rescored under this SAME new contract.",
        "selection_policy":"Final step540 per variant only; no early stopping, checkpoint/threshold/seed selection, resampling or retry."})
    card["approval"].update({"authorized_operations":["training"],"scope":card["purpose"],
        "confirmation_reference":"User-approved GSE-Graph plan, standing in-scope authorization, and docs/PLAN.md2026-09-05 16:48 same180 geometry small-training preparation after readout software readiness; exact new training card, not reused export permission."})
    reference_seal=f"{RECOVERY}/artifacts/evidence_sha256.txt"
    reference_digest="478e0744a7abaf5281342722a037b4aa1267f23b9b89377f5dd67415fed35481"
    if sha(PROJECT_ROOT/reference_seal)!=reference_digest: raise ValueError("reference seal drift")
    card["sealed_sources"][reference_seal]=reference_digest
    report=validate_point_axis_training_card(card)
    if not report.passed: raise ValueError(report.errors)
    spec=deepcopy(old)
    spec.update({"slug":SLUG,"operation":"training","data_card":CARD,"config_path":CARD,"training":deepcopy(TRAINING),
        "question":"On exactly the same180 fit observations, does learning slot-conditioned surface-to-axis offsets improve geometry over raw point pooling at matched initialization/order/update budget?",
        "method":"Freeze designated seed0 backbone; verify all six old fields and reconstructed old control readout before caching raw XYZ/fused memory/slot features; train only two small point readouts,3epochs/B1/540steps each,Adam.001,geometry-only normalized coordinateL1,final-only evaluation.",
        "baseline":"Matched raw_no_offset point head; identical initial predictions and540 observation schedule. Frozen old axes rescored by the same pure-geometry matcher. All32 predictions kept; not detection scoring.",
        "fallback":"Any input/feature parity, numerical, gradient, resource or ledger failure stops. A nonpositive fixed-budget fit probe stops expansion; do not increase epochs, change targets or select thresholds.",
        "wall_time_cap_s":1200,"user_authorization":card["approval"],
        "prediction_reference_root":f"{RECOVERY}/artifacts/shared_frame_predictions",
        "expected_counts":{"observations":180,"topology_parents":10,"node_identities_inventory_only":100,"sensor_frames":900,"visible_fragments":1452,"predictions_per_observation":32,"steps_per_variant":540,"optimizer_steps":1080,"new_teacher_labels":0,"test_world_reads":0},
        "estimated_cost":{"disk_gb":.25,"wall_time_hours":1/3,"host_ram_gb":4,"gpu_vram_gb":4,
            "compute":"One RTX5090D,<=1200s;180 old-model parity windows+180 cached-memory windows;2 zero-update head backward checks;1080 B1 optimizer steps and720 head evaluation windows.0 event training/new labels/test/graph."},
        "acceptance_criteria":[
            "Exact180 original observations/10 C01 parents/900 frames/1452 targets; sealed seed0 six-field batch18 parity and cached-memory reconstructed control parity exact before optimizer.",
            "New variants have exactly identical initial axes, same540 schedule,Adam.001,no weight decay; baseline offsets remain zero/frozen,backbone state/grad unchanged. Final checkpoint only.",
            "Before optimizer, both real-input variants have finite gradients and fit4GiB limits; every step finite and resource checked; no target generation or GT grouping in student.",
            "New FIT utility line (not existing scientificGate): macro observation coordinateMAE at least10% below raw/no-offset AND frozen legacy; coordinate and Euclidean improve vs initialization; Euclidean below raw; >=6 of10 parents improve coordinateMAE vs raw.",
            "Report all matching/unmatched counts and initial/final per-observation/per-parent results;10%/6-parent utility line does not establish geometry detector, event/port accuracy or held-out generalization.",
            "<=1200s/4GiB host/4GiB CUDA reserved; all sources unchanged,exact1080 updates;all predictions/final checkpoints/optimizer states/learning logs/complete180 XYXZ/summary/state/seal saved. No retries or automatic scientificGate promotion."],
        "expected_evidence":["Exact card/spec/hashes/environment;pretraining parity/resource/gradient evidence;identical initial states;540 schedule;1080 raw loss rows;initial/final axes and full metrics;two final head+optimizer checkpoints;all180 XYXZ,parent MAE plot;RUN_STATE and SHA256 seal."]})
    spec["source_seals"]=old["source_seals"]+[reference_seal]
    spec["expected_versions"].update({"scipy":scipy.__version__,"matplotlib":matplotlib.__version__})
    spec.pop("legacy_cache",None);spec.pop("selection_manifest",None)
    python=old["command"][7]
    spec["command"]=["env","CUBLAS_WORKSPACE_CONFIG=:4096:8","OPENBLAS_NUM_THREADS=1","MKL_NUM_THREADS=1","OMP_NUM_THREADS=1","PYTHONHASHSEED=0",python,
        "tools/v3/run_gse_point_axis_probe_v1.py","--spec",str(PROJECT_ROOT/SPEC),"--run-dir",str(PROJECT_ROOT/f"results/gate3_semantics/gate3_20260905_{SLUG}_seed0")]
    paths=set(old["source_sha256"])
    paths|={str(p.relative_to(PROJECT_ROOT)) for p in (PROJECT_ROOT/"src").rglob("*.py")}
    paths|={"tools/v3/run_gse_point_axis_probe_v1.py","tools/v3/freeze_gse_point_axis_probe_v1.py",
        "tests/v3/unit/test_gse_point_axis_loss.py","tests/v3/unit/test_gse_point_axis_probe.py",
        "tests/v3/unit/test_gse_point_axis_training_card.py","tests/v3/unit/test_gse_point_axis_head.py","tests/v3/unit/test_gse_point_axis_readout.py"}
    spec["source_sha256"]={p:sha(PROJECT_ROOT/p) for p in sorted(paths)}
    return card,spec


def main():
    if (PROJECT_ROOT/CARD).exists() or (PROJECT_ROOT/SPEC).exists(): raise RuntimeError("refuse overwrite of frozen card/spec")
    card,spec=documents();write_json(PROJECT_ROOT/CARD,card);write_json(PROJECT_ROOT/SPEC,spec)
    report=preflight(spec,load_json(PROJECT_ROOT/"results/project_status.json"),PROJECT_ROOT)
    print(json.dumps({"passed":report.passed,"errors":report.errors,"warnings":report.warnings,"card":CARD,"spec":SPEC}))
    return int(not report.passed)


if __name__=="__main__": raise SystemExit(main())
