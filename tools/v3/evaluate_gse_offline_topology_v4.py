#!/usr/bin/env python3
"""C09 baselines-first topology validation with the frozen open-set node gate."""

from __future__ import annotations

import argparse
from pathlib import Path

import evaluate_gse_offline_topology_v3 as v3


PASS_STATUS = "PASS_GSE_OFFLINE_TOPOLOGY_VALIDATION_V4"
FAIL_STATUS = "FAIL_GSE_OFFLINE_TOPOLOGY_VALIDATION_V4"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-run", required=True, type=Path)
    parser.add_argument("--perception-run", required=True, type=Path)
    parser.add_argument("--corrected-perception-run", required=True, type=Path)
    parser.add_argument("--dataset-run", required=True, type=Path)
    parser.add_argument("--teacher-run", required=True, type=Path)
    parser.add_argument("--verifier-run", required=True, type=Path)
    parser.add_argument("--ensemble-run", required=True, type=Path)
    parser.add_argument("--node-calibration-run", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    result = v3.evaluate(
        args.training_run,
        args.perception_run,
        args.corrected_perception_run,
        args.dataset_run,
        args.teacher_run,
        args.verifier_run,
        args.ensemble_run,
        args.output_dir,
        device=args.device,
        node_calibration_run=args.node_calibration_run,
        baselines_first=True,
        pass_status=PASS_STATUS,
        fail_status=FAIL_STATUS,
        schema_version="gse_offline_topology_validation_v4",
    )
    print(result["overall_status"])
    return 0 if result["overall_status"] == PASS_STATUS else 2


if __name__ == "__main__":
    raise SystemExit(main())
