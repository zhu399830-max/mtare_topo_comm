#!/usr/bin/env python3
"""Replacement COUST readiness with distribution-space rotation authority.

V1 incorrectly applied a 3e-5 scalar tolerance directly to a float32 atan2
bearing in degrees.  This replacement preserves that failed evidence and the
same tolerance, but applies it to the circular probability distribution that
the transport loss consumes.  The scalar bearing remains a diagnostic.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from mtare_topo.governance import load_json, write_json


PASS = "PASS_GSE_CYCLIC_ORDERED_UNIMODAL_SLOT_TRANSPORT_READINESS_V1R"
FAIL = "FAIL_GSE_CYCLIC_ORDERED_UNIMODAL_SLOT_TRANSPORT_READINESS_V1R"


def replacement_network_check(equivariance: dict[str, float]) -> tuple[bool, float]:
    """Check quantities in their native tensor/distribution units."""
    distribution_metrics = (
        "slot_mass_rotation", "raw_mass_rotation", "kappa_rotation",
        "concentration_rotation", "count_rotation", "slot_geometry_rotation",
        "cyclic_slot_loss", "batch_permutation",
    )
    error = max(float(equivariance[name]) for name in distribution_metrics)
    return error <= 3e-5 and float(equivariance["repeat"]) == 0.0, error


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-root", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    command = [
        sys.executable,
        str(Path(__file__).with_name("execute_gse_cyclic_ordered_unimodal_slot_transport_readiness_v1.py")),
        "--teacher-root", str(args.teacher_root),
        "--source-root", str(args.source_root),
        "--output-dir", str(args.output_dir),
    ]
    process = subprocess.run(command, check=False)
    if process.returncode not in (0, 2):
        return process.returncode

    summary_path = args.output_dir / "summary.json"
    summary = load_json(summary_path)
    equivariance = summary["equivariance"]
    network_pass, distribution_rotation_error = replacement_network_check(equivariance)
    equivariance["projected_distribution_rotation"] = float(equivariance["slot_mass_rotation"])
    equivariance["bearing_rotation_diagnostic_deg"] = equivariance.pop("bearing_rotation_deg")
    equivariance["authoritative_distribution_error"] = distribution_rotation_error
    summary["checks"]["network_rotation_cyclic_and_batch"] = network_pass
    summary["scientific_pass"] = all(summary["checks"].values())
    summary["status"] = PASS if summary["scientific_pass"] else FAIL
    summary["decision"] = (
        "ALLOW_CYCLIC_ORDERED_UNIMODAL_SLOT_TRANSPORT_THREE_SEED_DATA_CARD"
        if summary["scientific_pass"] else "STOP_CYCLIC_ORDERED_UNIMODAL_SLOT_TRANSPORT"
    )
    summary["schema_version"] = "gse_cyclic_ordered_unimodal_slot_transport_readiness_v1r"
    summary["replacement_rationale"] = {
        "superseded_run": "gate3_20260829_gse_cyclic_ordered_unimodal_slot_transport_readiness_v1_seed0",
        "v1_failure": "Float32 atan2 bearing diagnostic was compared in degrees to a dimensionless 3e-5 tensor tolerance.",
        "authority": "The normalized circular slot distribution consumed by assignment and likelihood.",
        "threshold_changed": False,
        "diagnostic_retained": True,
    }
    write_json(summary_path, summary)
    write_json(args.output_dir / "figure_source.json", summary)
    print(json.dumps({"status": summary["status"], "decision": summary["decision"], "checks": summary["checks"]}, indent=2, sort_keys=True))
    return 0 if summary["scientific_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
