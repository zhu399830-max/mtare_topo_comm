#!/usr/bin/env python3
"""Freeze initialization-only corrective, no scientific or training changes."""
import argparse
import json
import freeze_gse_point_axis_evidence_v1 as original
from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json,write_json,preflight
from run_gse_composition_inventory_v1 import sha


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument("--freeze",action="store_true");args=parser.parse_args()
    original.SLUG="gse_point_axis_evidence_v1r"
    original.CARD=f"configs/v3/gate3/data_cards/{original.SLUG}.json"
    original.SPEC=f"configs/v3/gate3/{original.SLUG}.json"
    if (PROJECT_ROOT/original.CARD).exists() or (PROJECT_ROOT/original.SPEC).exists():raise RuntimeError("immutable registration exists")
    card,spec=original.documents()
    spec["command"][spec["command"].index("tools/v3/run_gse_point_axis_evidence_v1.py")]="tools/v3/run_gse_point_axis_evidence_v1r.py"
    spec["system_corrective"]="Create parent directories before JSON writes. Original v1 failed before cached data reads. No change to teacher, matching, decisions, training or resource budget. Original sources and runs preserved."
    for path in ("tools/v3/run_gse_point_axis_evidence_v1r.py","tools/v3/freeze_gse_point_axis_evidence_v1r.py","tests/v3/unit/test_gse_point_axis_evidence_directories.py"):
        spec["source_sha256"][path]=sha(PROJECT_ROOT/path)
    if args.freeze:
        write_json(PROJECT_ROOT/original.CARD,card);write_json(PROJECT_ROOT/original.SPEC,spec)
        report=preflight(spec,load_json(PROJECT_ROOT/"results/project_status.json"),PROJECT_ROOT)
        print(json.dumps({"passed":report.passed,"errors":report.errors}));return int(not report.passed)
    print(json.dumps({"files_written":0,"valid_card":True}));return 0


if __name__=="__main__":raise SystemExit(main())
