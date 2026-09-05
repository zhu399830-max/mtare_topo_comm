#!/usr/bin/env python3
"""Register cached evidence completion; no training or checkpoint reads."""
import argparse
import json
from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, write_json, preflight
from mtare_topo.governance_inventory import validate_scoped_inventory_card
from run_gse_composition_inventory_v1 import sha

SLUG="gse_point_axis_evidence_v1"
CARD=f"configs/v3/gate3/data_cards/{SLUG}.json"
SPEC=f"configs/v3/gate3/{SLUG}.json"
BASE="results/gate3_semantics/gate3_20260905_gse_point_axis_probe_v1_seed0"
COORD="results/gate3_semantics/gate3_20260905_gse_coordinate_support_audit_v1_seed0"
PYTHON="/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python"


def documents():
    card=load_json(PROJECT_ROOT/"configs/v3/gate3/data_cards/gse_local_teacher_audit_v1.json")
    scope="Recompute identical180 cached axis metrics and complete missing plots after terminal plotting failure; preserve original failed run and STOP_BEFORE_EXPANSION decision. No training, checkpoint/model load, raw sensor decoding or new teacher labels."
    card.update(card_id=SLUG,purpose=scope,allowed_teacher_fields=[],
        allowed_cache_fields=["all_predictions","observation_metrics","sample_manifest","sample_schedule","progress","cached_teacher_control_points"],
        teacher_source="Restore existing float32 axis targets from the sealed coordinate-support audit queries, preserving exact original row/control identity. No source teacher arrays reopened or labels generated.",
        geometry_scoring_contract="Identical pure-geometry reversal-invariant matching; CPU/GPU float64 reductions compared within64eps, with exact matching identities and original decisions preserved.",
        leakage_audit="Only named sealed cached artifacts from the identical180 C01 fit observations. No source scans, checkpoint bytes, teacher arrays, C07-C10 or deployment inputs.",
        sealed_sources={})
    card["approval"].update(scope=scope,confirmation_reference="Standing user authorization for GSE-Graph implementation and preserving useful experiment evidence; separate audit-only recovery after plotting error, never authorization to retry training.")
    selections={BASE:["metrics/summary.json","RUN_STATE.json","artifacts/observation_metrics.json","artifacts/all_predictions.npz","artifacts/sample_manifest.json","artifacts/sample_schedule.json","logs/progress.jsonl"],COORD:["artifacts/observation_audit.json"]}
    expected={BASE:"2e1e72069e1c97422cfb2ac46853370247b1b0882aaaf774a9bdb0361de4a7c7",COORD:"5aabab7364e8d0eebe23a6b501967d0bf33231fe1e7f270100c284316791f3be"}
    for root,paths in selections.items():
        seal=root+"/artifacts/evidence_sha256.txt"
        if sha(PROJECT_ROOT/seal)!=expected[root]:raise ValueError("source seal drift")
        entries={line.split(None,1)[1]:line.split(None,1)[0] for line in (PROJECT_ROOT/seal).read_text().splitlines()}
        card["sealed_sources"][seal]=expected[root]
        for suffix in paths:
            path=root+"/"+suffix
            if sha(PROJECT_ROOT/path)!=entries[path]:raise ValueError("source artifact drift")
            card["sealed_sources"][path]=entries[path]
    report=validate_scoped_inventory_card(card)
    if not report.passed:raise ValueError(report.errors)
    source_files=sorted(str(p.relative_to(PROJECT_ROOT)) for p in (PROJECT_ROOT/"src/mtare_topo").rglob("*.py"))
    source_files += ["tools/v3/"+p for p in ("_bootstrap.py","preflight.py","create_run.py","run_gse_composition_inventory_v1.py","run_gse_composition_field_recovery_v1.py","run_gse_point_axis_probe_v1.py","run_gse_point_axis_evidence_v1.py","freeze_gse_point_axis_evidence_v1.py")]
    source_files += ["tests/v3/unit/test_gse_point_axis_evidence.py"]
    previous=load_json(PROJECT_ROOT/"configs/v3/gate3/gse_point_axis_probe_v1.json")
    run=f"results/gate3_semantics/gate3_20260905_{SLUG}_seed0"
    spec=dict(schema_version="v3_run_spec_v1",gate=3,date="20260905",slug=SLUG,seed=0,operation="audit",
        question="Can the completed fixed-budget experiment be fully evidenced without retraining or altering its failed scientific decision?",
        method=scope,baseline="Saved original per-observation metrics, matched identities, update ledger and decision.",
        fallback="Stop on any cached input, identity, metric, decision or resource mismatch; no training retry.",
        user_authorization=card["approval"],data_card=CARD,config_path=CARD,
        command=["env","CUDA_VISIBLE_DEVICES=","OPENBLAS_NUM_THREADS=1","MKL_NUM_THREADS=1","OMP_NUM_THREADS=1","PYTHONHASHSEED=0",PYTHON,"tools/v3/run_gse_point_axis_evidence_v1.py","--spec",str(PROJECT_ROOT/SPEC),"--run-dir",str(PROJECT_ROOT/run)],
        estimated_cost=dict(disk_gb=.1,wall_time_hours=1/30,host_ram_gb=4,compute="CPU<=120s; zero training, inference, checkpoint reads or raw decoding."),
        expected_versions=previous["expected_versions"],training_run=BASE,coordinate_rows=COORD+"/artifacts/observation_audit.json",
        source_sha256={p:sha(PROJECT_ROOT/p) for p in source_files},
        acceptance_criteria=["Exact180 observations/10 parents/1452 visible fragments restored from sealed caches.","All720 saved prediction evaluations reproduce matching identities and numeric metrics; original scientific STOP unchanged.","Raw ledger preserves540 updates per variant,1080 total; evidence recovery adds0 updates.","All180 XY/XZ plots plus parent comparison, environment, summaries and seal; original failed run unchanged.","CPU<=120s/4GiB; no model/checkpoint/sensor/teacher source reads or new labels."],
        expected_evidence=["Complete180-observation previews, parent comparison and metrics, verified original ledger/decisions, environment, raw log, RUN_STATE and SHA256 seal."])
    return card,spec


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument("--freeze",action="store_true");args=parser.parse_args()
    if (PROJECT_ROOT/CARD).exists() or (PROJECT_ROOT/SPEC).exists():raise RuntimeError("immutable registration already exists")
    card,spec=documents()
    if args.freeze:
        write_json(PROJECT_ROOT/CARD,card);write_json(PROJECT_ROOT/SPEC,spec)
        report=preflight(spec,load_json(PROJECT_ROOT/"results/project_status.json"),PROJECT_ROOT)
        print(json.dumps({"passed":report.passed,"errors":report.errors,"spec":SPEC}));return int(not report.passed)
    print(json.dumps({"valid_card":True,"observations":180,"files_written":0}));return 0


if __name__=="__main__":raise SystemExit(main())
