#!/usr/bin/env python3
"""Register only cached-list parser repair after zero-score failure."""
import argparse
import json
from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json,write_json,preflight
from run_gse_composition_inventory_v1 import sha


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument("--freeze",action="store_true");args=parser.parse_args()
    slug="gse_axis_observation_dependence_v1r";card_path=f"configs/v3/gate3/data_cards/{slug}.json";spec_path=f"configs/v3/gate3/{slug}.json"
    if (PROJECT_ROOT/card_path).exists() or (PROJECT_ROOT/spec_path).exists():raise RuntimeError("immutable registration exists")
    spec=load_json(PROJECT_ROOT/"configs/v3/gate3/gse_axis_observation_dependence_v1.json");card=load_json(PROJECT_ROOT/spec["data_card"])
    card["card_id"]=slug
    spec.update(slug=slug,data_card=card_path,config_path=card_path,system_corrective="sample_manifest.json and observation_audit.json are JSON lists, not configuration objects. Repair only these readers; preserve fixed derangement, all inputs/metrics/resource/decisions. V1 failed before any scores.")
    command=spec["command"];command[command.index("tools/v3/run_gse_axis_observation_dependence_v1.py")]="tools/v3/run_gse_axis_observation_dependence_v1r.py"
    command[command.index("--spec")+1]=str(PROJECT_ROOT/spec_path)
    command[command.index("--run-dir")+1]=str(PROJECT_ROOT/f"results/gate3_semantics/gate3_20260905_{slug}_seed0")
    for path in ("tools/v3/run_gse_axis_observation_dependence_v1r.py","tools/v3/freeze_gse_axis_observation_dependence_v1r.py","tests/v3/unit/test_gse_axis_observation_dependence_runner.py"):
        spec["source_sha256"][path]=sha(PROJECT_ROOT/path)
    for path,digest in {**card["sealed_sources"],**spec["source_sha256"]}.items():
        if sha(PROJECT_ROOT/path)!=digest:raise ValueError("original freeze changed")
    if args.freeze:
        write_json(PROJECT_ROOT/card_path,card);write_json(PROJECT_ROOT/spec_path,spec)
        report=preflight(spec,load_json(PROJECT_ROOT/"results/project_status.json"),PROJECT_ROOT)
        print(json.dumps({"passed":report.passed,"errors":report.errors}));return int(not report.passed)
    print(json.dumps({"files_written":0,"valid_card":True}));return 0


if __name__=="__main__":raise SystemExit(main())
