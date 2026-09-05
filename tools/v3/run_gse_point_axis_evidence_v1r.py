#!/usr/bin/env python3
"""Directory-initialization corrective; preserve v1 source and failed output."""
from pathlib import Path
import run_gse_point_axis_evidence_v1 as evidence
from mtare_topo.governance import write_json as original_write_json


def write_json_with_parent(path, value):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    original_write_json(path,value)


def main():
    evidence.write_json=write_json_with_parent
    return evidence.main()


if __name__=="__main__":raise SystemExit(main())
