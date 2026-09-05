#!/usr/bin/env python3
"""Execute and seal the frozen axis-anchored event-relation Teacher audit."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import time
import traceback
from pathlib import Path

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, validate_data_card, write_json


PASS = "PASS_GSE_AXIS_ANCHORED_EVENT_RELATION_TEACHER_FEASIBILITY_V1"
FAIL = "FAIL_GSE_AXIS_ANCHORED_EVENT_RELATION_TEACHER_FEASIBILITY_V1"
CARD_STATUS = "APPROVED_FOR_ONE_IMMUTABLE_GSE_AXIS_ANCHORED_EVENT_RELATION_TEACHER_FEASIBILITY_V1"
PYTHON = Path("/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python")
DATASET = PROJECT_ROOT / "results/gate2_representation/gate2_20260824_gse_deduplicated_dataset_export_v1_seed0"
EXIT_ACTION = PROJECT_ROOT / "results/gate3_semantics/gate3_20260828_gse_exit_action_transport_feasibility_v1_seed0"
COMMIT = PROJECT_ROOT / "results/gate3_semantics/gate3_20260829_gse_relational_commit_policy_feasibility_v1_seed0"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_seal(path: Path) -> int:
    checked = 0
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            expected, relative = line.rstrip("\n").split("  ", 1)
            target = PROJECT_ROOT / relative
            if not target.is_file() or sha256(target) != expected:
                raise RuntimeError(f"sealed source drift: {relative}")
            checked += 1
    return checked


def seal(run: Path) -> int:
    target = run / "artifacts/evidence_sha256.txt"
    files = sorted(path for path in run.rglob("*") if path.is_file() and path != target)
    with target.open("w", encoding="utf-8") as stream:
        for path in files:
            stream.write(f"{sha256(path)}  {path.relative_to(PROJECT_ROOT)}\n")
    return len(files)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    spec = load_json(args.spec.resolve())
    run = args.run_dir.resolve()
    run_id = f"gate{spec['gate']}_{spec['date']}_{spec['slug']}_seed{spec['seed']}"
    if run.name != run_id or load_json(run / "RUN_STATE.json").get("state") != "CREATED_NOT_EXECUTED":
        raise RuntimeError("axis-anchored Teacher feasibility executes exactly once")
    started = time.monotonic()
    overall = FAIL
    error = None
    result: dict = {}
    before: dict[str, str] = {}
    after: dict[str, str] = {}
    peak_rss_kib = None
    test_returncode = None
    executor_returncode = None
    verified_source_entries = 0
    try:
        card = load_json(PROJECT_ROOT / spec["data_card"])
        validation = validate_data_card(card)
        if not validation.passed or card.get("status") != CARD_STATUS:
            raise RuntimeError(f"Teacher feasibility card invalid: {validation.errors}")
        for record in spec["frozen_tools"].values():
            if sha256(PROJECT_ROOT / record["path"]) != record["sha256"]:
                raise RuntimeError(f"tool drift: {record['path']}")
        for relative, expected in spec["frozen_inputs"].items():
            before[relative] = sha256(PROJECT_ROOT / relative)
            if before[relative] != expected:
                raise RuntimeError(f"input drift: {relative}")
        verified_source_entries = verify_seal(DATASET / "artifacts/evidence_sha256.txt")
        environment = json.loads(subprocess.check_output([
            str(PYTHON), "-c",
            "import json,matplotlib,numpy,zarr,sys;print(json.dumps({'python':sys.version.split()[0],'numpy':numpy.__version__,'zarr':zarr.__version__,'matplotlib':matplotlib.__version__},sort_keys=True))",
        ], text=True))
        expected_environment = {"python": "3.13.5", "numpy": "2.1.3", "zarr": "2.18.7", "matplotlib": "3.10.0"}
        if environment != expected_environment:
            raise RuntimeError(f"environment drift: {environment}")
        write_json(run / "config/environment.json", {"executable": str(PYTHON), "versions": environment, "gpu_required": False})
        write_json(run / "config/source_integrity_before.json", before)
        write_json(run / "RUN_STATE.json", {"schema_version": "v3_run_state_v1", "run_id": run_id, "state": "RUNNING"})
        process_environment = os.environ.copy()
        process_environment["PYTHONPATH"] = str(PROJECT_ROOT / "src") + os.pathsep + str(PROJECT_ROOT / "tools/v3")
        test_log = run / "logs/00_unit_tests.log"
        with test_log.open("w", encoding="utf-8") as stream:
            completed = subprocess.run([
                str(PYTHON), "-m", "pytest", "-q",
                "tests/v3/unit/test_gse_axis_anchored_event_relation_teacher_feasibility.py",
            ], cwd=PROJECT_ROOT, env=process_environment, stdout=stream, stderr=subprocess.STDOUT, text=True, timeout=300, check=False)
        test_returncode = int(completed.returncode)
        if test_returncode != 0:
            raise RuntimeError("Teacher feasibility unit tests failed")
        output = run / "metrics/feasibility"
        command = [
            str(PYTHON), str(PROJECT_ROOT / "tools/v3/execute_gse_axis_anchored_event_relation_teacher_feasibility_v1.py"),
            "--dataset-root", str(DATASET / "artifacts/dataset/train"),
            "--sequence-manifest", str(DATASET / "artifacts/sequence_manifest.jsonl"),
            "--exit-action-summary", str(EXIT_ACTION / "metrics/summary.json"),
            "--commit-summary", str(COMMIT / "metrics/feasibility/summary.json"),
            "--output-dir", str(output),
        ]
        write_json(run / "config/executed_command.json", command)
        log = run / "logs/01_teacher_feasibility.log"
        with log.open("w", encoding="utf-8") as stream:
            completed = subprocess.run(["/usr/bin/time", "-v", *command], cwd=PROJECT_ROOT, env=process_environment, stdout=stream, stderr=subprocess.STDOUT, text=True, timeout=3600, check=False)
        executor_returncode = int(completed.returncode)
        match = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", log.read_text(encoding="utf-8"))
        peak_rss_kib = int(match.group(1)) if match else None
        if executor_returncode != 0:
            raise RuntimeError("Teacher feasibility executor failed")
        result = load_json(output / "summary.json")
        if result.get("status") != PASS or result.get("scientific_pass") is not True:
            raise RuntimeError("Teacher feasibility scientific contract failed")
        expected = spec["expected_counts"]
        population = result["teacher"]["population"]
        for key in ("worlds", "fit_observations", "selection_observations", "observations", "raw_frames", "visible_exit_tokens", "association_identities"):
            if population[key] != expected[key]:
                raise RuntimeError(f"Teacher feasibility population drift: {key}")
        required = [
            output / "summary.json", output / "figure_source.json", output / "identity_contract.csv", output / "event_population.csv",
            *[output / f"gse_axis_anchored_event_relation_teacher_feasibility_v1.{suffix}" for suffix in ("png", "pdf", "svg")],
        ]
        if any(not path.is_file() or path.stat().st_size == 0 for path in required):
            raise RuntimeError("Teacher feasibility evidence incomplete")
        if peak_rss_kib is None or peak_rss_kib > 8 * 1024 * 1024:
            raise RuntimeError(f"Teacher feasibility RAM contract failed: {peak_rss_kib}")
        after = {relative: sha256(PROJECT_ROOT / relative) for relative in before}
        if before != after:
            raise RuntimeError("frozen Teacher feasibility source changed")
        overall = PASS
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        (run / "logs/failure_traceback.log").write_text(traceback.format_exc(), encoding="utf-8")
    write_json(run / "metrics/summary.json", {
        "schema_version": "gse_axis_anchored_event_relation_teacher_feasibility_outer_v1",
        "overall_status": overall, "scientific_pass": overall == PASS, "error": error,
        "unit_test_returncode": test_returncode, "executor_returncode": executor_returncode,
        "verified_source_seal_entries": verified_source_entries, "peak_host_rss_kib": peak_rss_kib,
        "result": result, "duration_seconds": time.monotonic() - started,
        "source_unchanged": bool(before and before == after), "optimizer_steps": 0,
        "new_model_inference_observations": 0, "checkpoint_selection_observations": 0,
        "c09_worlds_read": 0, "c10_worlds_read": 0, "mtare_worlds_read": 0,
        "graph_replays": 0, "planner_calls": 0,
    })
    write_json(run / "RUN_STATE.json", {
        "schema_version": "v3_run_state_v1", "run_id": run_id,
        "state": "COMPLETED" if overall == PASS and error is None else "FAILED",
        "overall_status": overall, "error": error,
    })
    entries = seal(run)
    print(json.dumps({"overall_status": overall, "error": error, "decision": result.get("decision"), "verified_source_entries": verified_source_entries, "seal_entries": entries}, indent=2))
    return 0 if overall == PASS and error is None else 2


if __name__ == "__main__":
    raise SystemExit(main())
