#!/usr/bin/env python3
"""Narrow JSON-list reader corrective; preserve original executable and run."""
import json
from pathlib import Path
import run_gse_axis_observation_dependence_v1 as original
from mtare_topo.governance import load_json as load_object


def load_declared_json(path):
    path=Path(path)
    if path.name in ("sample_manifest.json","observation_audit.json"):
        value=json.loads(path.read_text())
        if not isinstance(value,list):raise ValueError("cached manifest/observation audit must be a list")
        return value
    return load_object(path)


def main():
    original.load_json=load_declared_json
    return original.main()


if __name__=="__main__":raise SystemExit(main())
