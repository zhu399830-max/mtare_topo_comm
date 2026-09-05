#!/usr/bin/env python3
"""Execute and seal the one-shot RouteGeometryProfile proof."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time
import traceback

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, validate_data_card, write_json


RUN_ID = "gate3_20260829_gse_route_geometry_profile_proof_v1_seed0"
PASS = "PASS_GSE_ROUTE_GEOMETRY_PROFILE_PROOF_V1"
FAIL = "FAIL_GSE_ROUTE_GEOMETRY_PROFILE_PROOF_V1"
CARD_STATUS = "APPROVED_FOR_ONE_IMMUTABLE_GSE_ROUTE_GEOMETRY_PROFILE_PROOF_V1"
PYTHON = Path("/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python")
DATASET = PROJECT_ROOT / "results/gate2_representation/gate2_20260824_gse_deduplicated_dataset_export_v1_seed0"
SUPERVISION = PROJECT_ROOT / "results/gate3_semantics/gate3_20260829_gse_composer_supervision_v1r_seed0"
EVALUATOR = PROJECT_ROOT / "tools/v3/evaluate_gse_route_geometry_profile_proof_v1.py"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def seal(run_dir: Path) -> int:
    destination = run_dir / "artifacts/evidence_sha256.txt"
    paths = sorted(path for path in run_dir.rglob("*") if path.is_file() and path != destination)
    with destination.open("w", encoding="utf-8") as stream:
        for path in paths: stream.write(f"{sha256(path)}  {path.relative_to(PROJECT_ROOT)}\n")
    return len(paths)


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--spec", required=True, type=Path); parser.add_argument("--run-dir", required=True, type=Path); args = parser.parse_args()
    run_dir = args.run_dir.resolve(); spec = load_json(args.spec.resolve())
    if run_dir.name != RUN_ID or load_json(run_dir / "RUN_STATE.json").get("state") != "CREATED_NOT_EXECUTED":
        raise RuntimeError("RouteGeometryProfile proof executes exactly once")
    started = time.monotonic(); overall = FAIL; error = None; result: dict = {}; returncode = None; peak_rss = None
    before: dict[str, str] = {}; after: dict[str, str] = {}
    try:
        if spec.get("gate") != 3 or spec.get("operation") != "audit" or spec.get("seed") != 0:
            raise RuntimeError("RouteGeometryProfile formal scope mismatch")
        card = load_json(PROJECT_ROOT / spec["data_card"]); report = validate_data_card(card)
        if not report.passed or card.get("status") != CARD_STATUS or card.get("approval", {}).get("status") != "APPROVED":
            raise RuntimeError("RouteGeometryProfile Data Card mismatch")
        for record in spec["frozen_tools"].values():
            if sha256(PROJECT_ROOT / record["path"]) != record["sha256"]: raise RuntimeError(f"frozen tool drift: {record['path']}")
        for relative, expected in spec["frozen_inputs"].items():
            observed = sha256(PROJECT_ROOT / relative)
            if observed != expected: raise RuntimeError(f"frozen input drift: {relative}")
            before[relative] = observed
        versions = json.loads(subprocess.check_output([str(PYTHON), "-c", "import json,matplotlib,numpy,zarr,sys;print(json.dumps({'python':sys.version.split()[0],'numpy':numpy.__version__,'matplotlib':matplotlib.__version__,'zarr':zarr.__version__},sort_keys=True))"], text=True))
        write_json(run_dir / "config/environment.json", {"executable": str(PYTHON), "versions": versions, "cpu_only": True})
        write_json(run_dir / "config/source_integrity_before.json", before)
        write_json(run_dir / "RUN_STATE.json", {"schema_version": "v3_run_state_v1", "run_id": RUN_ID, "state": "RUNNING"})
        env = os.environ.copy(); env["PYTHONPATH"] = str(PROJECT_ROOT / "src") + os.pathsep + str(PROJECT_ROOT / "tools/v3"); env["OMP_NUM_THREADS"] = "4"; env["MKL_NUM_THREADS"] = "4"
        with (run_dir / "logs/00_unit_tests.log").open("w", encoding="utf-8") as stream:
            tests = subprocess.run([str(PYTHON), "-m", "pytest", "-q", "tests/v3/unit/test_gse_route_geometry_profile.py"], cwd=PROJECT_ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, text=True, timeout=300, check=False)
        if tests.returncode: raise RuntimeError("RouteGeometryProfile unit tests failed")
        output = run_dir / "metrics/profile_proof"
        command = ["/usr/bin/time", "-v", str(PYTHON), str(EVALUATOR), "--dataset", str(DATASET), "--supervision", str(SUPERVISION), "--output-dir", str(output)]
        (run_dir / "config/command.txt").write_text(" ".join(command) + "\n", encoding="utf-8")
        log = run_dir / "logs/01_profile_proof.log"
        with log.open("w", encoding="utf-8") as stream:
            completed = subprocess.run(command, cwd=PROJECT_ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, text=True, timeout=1500, check=False)
        returncode = int(completed.returncode)
        match = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", log.read_text(encoding="utf-8")); peak_rss = int(match.group(1)) if match else None
        result = load_json(output / "summary.json") if (output / "summary.json").is_file() else {}
        after = {relative: sha256(PROJECT_ROOT / relative) for relative in before}
        required = [output / "summary.json", output / "observation_audit.jsonl", output / "event_summary.csv", *[output / f"gse_route_geometry_profile_proof_v1.{suffix}" for suffix in ("png", "pdf", "svg")]]
        output_bytes = sum(path.stat().st_size for path in run_dir.rglob("*") if path.is_file())
        if (before != after or returncode not in (0, 2) or result.get("overall_status") not in (PASS, FAIL) or (returncode == 0) != (result.get("overall_status") == PASS) or result.get("worlds") != 80 or result.get("observations") != 188126 or result.get("event_rows") != {"geometry_transition": 1031, "turn": 1998} or result.get("optimizer_steps") != 0 or result.get("model_inference_frames") != 0 or any(result.get(name) != 0 for name in ("c09_worlds_read", "c10_worlds_read", "mtare_worlds_read", "graph_replays")) or peak_rss is None or peak_rss > 4 * 1024**2 or output_bytes > 256 * 1024**2 or not all(path.is_file() and path.stat().st_size > 0 for path in required)):
            raise RuntimeError("RouteGeometryProfile execution/evidence contract failed")
        overall = str(result["overall_status"])
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"; (run_dir / "logs/failure_traceback.log").write_text(traceback.format_exc(), encoding="utf-8")
    summary = {"schema_version": "gse_route_geometry_profile_proof_outer_v1", "overall_status": overall, "scientific_pass": overall == PASS and error is None, "error": error, "returncode": returncode, "peak_host_rss_kib": peak_rss, "source_unchanged": bool(before and before == after), "profile_proof": result, "optimizer_steps": 0, "model_inference_frames": 0, "c09_worlds_read": 0, "c10_worlds_read": 0, "mtare_worlds_read": 0, "graph_replays": 0, "duration_seconds": time.monotonic() - started}
    write_json(run_dir / "metrics/summary.json", summary)
    write_json(run_dir / "RUN_STATE.json", {"schema_version": "v3_run_state_v1", "run_id": RUN_ID, "state": "COMPLETED" if error is None else "FAILED", "overall_status": overall, "error": error})
    entries = seal(run_dir); print(json.dumps({"overall_status": overall, "error": error, "evidence_files": entries}, indent=2))
    return 0 if overall == PASS and error is None else 2


if __name__ == "__main__":
    raise SystemExit(main())
