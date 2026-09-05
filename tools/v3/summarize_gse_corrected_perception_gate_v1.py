#!/usr/bin/env python3
"""Apply the frozen complete C09 gate with only slope replaced by V1R evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import PROJECT_ROOT
from mtare_topo.evaluation.gse_slope_corrective_gate import summarize_corrected_perception_gate
from mtare_topo.governance import load_json, write_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-gate", required=True, type=Path)
    parser.add_argument("--corrective-summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    original = load_json(args.original_gate.resolve())
    corrective = load_json(args.corrective_summary.resolve())
    if corrective.get("overall_status") not in {
        "PASS_GSE_SLOPE_CORRECTIVE_C09_EVALUATION_V1",
        "PASS_GSE_SLOPE_RISK_CALIBRATED_C09_EVALUATION_V2",
    }:
        raise RuntimeError("corrective C09 evaluation is not a technical PASS")
    result = summarize_corrected_perception_gate(original, corrective)
    result["sources"] = {
        "original_gate": str(args.original_gate.resolve().relative_to(PROJECT_ROOT)),
        "corrective_summary": str(args.corrective_summary.resolve().relative_to(PROJECT_ROOT)),
    }
    result["strict_test_worlds_read"] = 0
    result["mtare_worlds_read"] = 0
    result["optimizer_steps"] = 0
    result["model_updates"] = 0
    write_json(args.output.resolve(), result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
