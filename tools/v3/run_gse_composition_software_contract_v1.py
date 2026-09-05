#!/usr/bin/env python3
"""Seal synthetic-only software checks. This is NOT a scientific Gate PASS."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
import traceback
import xml.etree.ElementTree as ET

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import build_run_id, load_json, write_json


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    spec, run = load_json(args.spec), args.run_dir.resolve()
    expected = PROJECT_ROOT / "results/gate3_semantics" / build_run_id(spec)
    # Reject BEFORE writing anything into an already executed or foreign run.
    if run != expected or load_json(run / "RUN_STATE.json")["state"] != "CREATED_NOT_EXECUTED":
        raise RuntimeError("requires one fresh create_run directory; retry/overwrite refused")
    if load_json(run / "config/run_spec.json") != spec:
        raise RuntimeError("spec differs from create_run snapshot")
    started, error, metrics = time.monotonic(), None, {}
    write_json(run / "RUN_STATE.json", {"state": "RUNNING", "run_id": run.name})
    try:
        if spec["operation"] != "infrastructure" or spec["expected_counts"]["dataset_frames"] != 0:
            raise RuntimeError("runner supports only synthetic software checks")
        for path, digest in spec["source_sha256"].items():
            if sha(PROJECT_ROOT / path) != digest:
                raise RuntimeError(f"source drift: {path}")
        import numpy
        import pytest
        import torch
        versions = {"python": platform.python_version(), "torch": torch.__version__,
                    "numpy": numpy.__version__, "pytest": pytest.__version__}
        if versions != spec["expected_versions"]:
            raise RuntimeError(f"environment drift: {versions}")
        write_json(run / "config/execution_environment.json", {
            **versions, "executable": sys.executable, "platform": platform.platform(),
            "cuda_available_diagnostic": torch.cuda.is_available(), "device_used": "cpu",
        })
        environment = os.environ.copy()
        environment.update({"PYTHONPATH": str(PROJECT_ROOT / "src"), "OMP_NUM_THREADS": "1",
                            "PYTHONHASHSEED": "0", "CUDA_VISIBLE_DEVICES": ""})
        command = [sys.executable, "-m", "pytest", "-q", *spec["tests"],
                   "--junitxml", str(run / "artifacts/tests.xml")]
        write_json(run / "config/test_command.json", command)
        with (run / "logs/tests.log").open("x") as log:
            result = subprocess.run(command, cwd=PROJECT_ROOT, env=environment,
                                    stdout=log, stderr=subprocess.STDOUT, timeout=120)
        suites = list(ET.parse(run / "artifacts/tests.xml").getroot().iter("testsuite"))
        counts = {k: sum(int(s.get(k, "0")) for s in suites) for k in ("tests", "failures", "errors", "skipped")}
        peak = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * 1024
        if (result.returncode or counts != {"tests": spec["expected_counts"]["tests"], "failures": 0, "errors": 0, "skipped": 0}
                or peak > 4 * 1024**3):
            raise RuntimeError(f"software contract failed: rc={result.returncode}, {counts}, RSS={peak}")
        for path, digest in spec["source_sha256"].items():
            if sha(PROJECT_ROOT / path) != digest:
                raise RuntimeError(f"source changed during checks: {path}")
        metrics = {**counts, "peak_child_rss_bytes": peak, "dataset_worlds": 0,
                   "dataset_frames": 0, "optimizer_steps": 0, "mtare_runs": 0,
                   "scientific_performance_claim": False, "gate_advanced": False}
    except Exception:
        error = traceback.format_exc()
        (run / "logs/error.log").write_text(error)
    write_json(run / "metrics/summary.json", {
        "status": "SOFTWARE_CONTRACT_PASS" if error is None else "SOFTWARE_CONTRACT_FAIL",
        "duration_s": time.monotonic() - started, "metrics": metrics, "error": error,
        "limitations": ["Synthetic fixtures only, not learned accuracy or deployment qualification.",
                        "Registration verifier and calibrated thresholds are not implemented by this kernel.",
                        "Old 180-observation cache lacks common-frame geometry and per-port targets."],
    })
    write_json(run / "RUN_STATE.json", {"schema_version": "v3_run_state_v1", "run_id": run.name,
                                        "state": "COMPLETED" if error is None else "FAILED", "error": error})
    seal = run / "artifacts/evidence_sha256.txt"
    seal.write_text("".join(f"{sha(p)}  {p.relative_to(PROJECT_ROOT)}\n"
                            for p in sorted(run.rglob("*")) if p.is_file() and p != seal))
    print(json.dumps({"run": str(run), "error": error, "seal_sha256": sha(seal), "metrics": metrics}))
    return int(error is not None)


if __name__ == "__main__":
    raise SystemExit(main())
