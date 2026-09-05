#!/usr/bin/env python3
"""Execute and seal one axis-anchored event-relation readiness run."""
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


PASS = "PASS_GSE_AXIS_ANCHORED_EVENT_RELATION_READINESS_V1"
FAIL = "FAIL_GSE_AXIS_ANCHORED_EVENT_RELATION_READINESS_V1"
CARD_STATUS = "APPROVED_FOR_ONE_IMMUTABLE_GSE_AXIS_ANCHORED_EVENT_RELATION_READINESS_V1"
PASS_V1R = "PASS_GSE_AXIS_ANCHORED_EVENT_RELATION_READINESS_V1R"
FAIL_V1R = "FAIL_GSE_AXIS_ANCHORED_EVENT_RELATION_READINESS_V1R"
CARD_STATUS_V1R = "APPROVED_FOR_ONE_IMMUTABLE_GSE_AXIS_ANCHORED_EVENT_RELATION_READINESS_V1R"
PASS_V2 = "PASS_GSE_AXIS_ANCHORED_EVENT_RELATION_READINESS_V2"
FAIL_V2 = "FAIL_GSE_AXIS_ANCHORED_EVENT_RELATION_READINESS_V2"
CARD_STATUS_V2 = "APPROVED_FOR_ONE_IMMUTABLE_GSE_AXIS_ANCHORED_EVENT_RELATION_READINESS_V2"
PYTHON = Path("/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python")
DATASET = PROJECT_ROOT / "results/gate2_representation/gate2_20260824_gse_deduplicated_dataset_export_v1_seed0"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_seal(path: Path) -> int:
    count = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        target = PROJECT_ROOT / relative
        if not target.is_file() or sha256(target) != expected:
            raise RuntimeError(f"sealed source drift: {relative}")
        count += 1
    return count


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
    spec = load_json(args.spec.resolve()); run = args.run_dir.resolve()
    version = "v2" if spec["slug"].endswith("_v2") else ("v1r" if spec["slug"].endswith("_v1r") else "v1")
    expected_pass = {"v1": PASS, "v1r": PASS_V1R, "v2": PASS_V2}[version]
    expected_fail = {"v1": FAIL, "v1r": FAIL_V1R, "v2": FAIL_V2}[version]
    expected_card_status = {"v1": CARD_STATUS, "v1r": CARD_STATUS_V1R, "v2": CARD_STATUS_V2}[version]
    run_id = f"gate{spec['gate']}_{spec['date']}_{spec['slug']}_seed{spec['seed']}"
    if run.name != run_id or load_json(run / "RUN_STATE.json").get("state") != "CREATED_NOT_EXECUTED":
        raise RuntimeError("axis-anchored readiness executes exactly once")
    started = time.monotonic(); overall = expected_fail; error = None; result = {}
    before = {}; after = {}; tests = None; executor = None; peak_rss_kib = None; verified_entries = 0
    try:
        card = load_json(PROJECT_ROOT / spec["data_card"]); validation = validate_data_card(card)
        if not validation.passed or card.get("status") != expected_card_status:
            raise RuntimeError(f"axis-anchored readiness card invalid: {validation.errors}")
        for record in spec["frozen_tools"].values():
            if sha256(PROJECT_ROOT / record["path"]) != record["sha256"]:
                raise RuntimeError(f"tool drift: {record['path']}")
        for relative, expected in spec["frozen_inputs"].items():
            before[relative] = sha256(PROJECT_ROOT / relative)
            if before[relative] != expected: raise RuntimeError(f"input drift: {relative}")
        verified_entries = verify_seal(DATASET / "artifacts/evidence_sha256.txt")
        environment = json.loads(subprocess.check_output([
            str(PYTHON), "-c", "import json,numpy,torch,zarr,sys;print(json.dumps({'python':sys.version.split()[0],'numpy':numpy.__version__,'torch':torch.__version__,'zarr':zarr.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0)},sort_keys=True))"
        ], text=True))
        expected_environment = {"python": "3.13.5", "numpy": "2.1.3", "torch": "2.9.0+cu129", "zarr": "2.18.7", "cuda": "12.9", "gpu": "NVIDIA GeForce RTX 5090 D"}
        if environment != expected_environment: raise RuntimeError(f"environment drift: {environment}")
        write_json(run / "config/environment.json", {"executable": str(PYTHON), "versions": environment, "deterministic": True, "tf32": False})
        write_json(run / "config/source_integrity_before.json", before)
        write_json(run / "RUN_STATE.json", {"schema_version": "v3_run_state_v1", "run_id": run_id, "state": "RUNNING"})
        env = os.environ.copy(); env["PYTHONPATH"] = f"{PROJECT_ROOT / 'src'}{os.pathsep}{PROJECT_ROOT / 'tools/v3'}"; env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        test_paths = ["tests/v3/unit/test_gse_axis_anchored_event_relation.py"]
        if version == "v2":
            test_paths.append("tests/v3/unit/test_gse_axis_anchored_event_relation_training.py")
        with (run / "logs/00_unit_tests.log").open("w", encoding="utf-8") as stream:
            completed = subprocess.run([str(PYTHON), "-m", "pytest", "-q", *test_paths], cwd=PROJECT_ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, text=True, timeout=300, check=False)
        tests = int(completed.returncode)
        if tests != 0: raise RuntimeError("axis-anchored readiness unit tests failed")
        output = run / "metrics/readiness"
        command = [str(PYTHON), str(PROJECT_ROOT / "tools/v3/execute_gse_axis_anchored_event_relation_readiness_v1.py"), "--dataset-root", str(DATASET / "artifacts/dataset/train"), "--output-dir", str(output)]
        if version != "v1": command.extend(("--status-version", version))
        if version == "v2":
            command.extend(("--sequence-manifest", str(DATASET / "artifacts/sequence_manifest.jsonl")))
        write_json(run / "config/executed_command.json", command)
        log = run / "logs/01_readiness.log"
        with log.open("w", encoding="utf-8") as stream:
            completed = subprocess.run(["/usr/bin/time", "-v", *command], cwd=PROJECT_ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, text=True, timeout=1800, check=False)
        executor = int(completed.returncode)
        match = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", log.read_text(encoding="utf-8")); peak_rss_kib = int(match.group(1)) if match else None
        if executor != 0: raise RuntimeError("axis-anchored readiness executor failed")
        result = load_json(output / "summary.json")
        if result.get("status") != expected_pass or result.get("scientific_pass") is not True: raise RuntimeError("axis-anchored readiness scientific contract failed")
        expected_observations = 8 if version == "v2" else 7
        if result["parameters"] > 1_500_000 or result["real_batch"]["observations"] != expected_observations or not all(result["checks"].values()): raise RuntimeError("axis-anchored readiness result drift")
        required = [output / "summary.json", output / "figure_source.json", output / "real_batch_manifest.json", *[output / f"gse_axis_anchored_event_relation_readiness_v1.{suffix}" for suffix in ("png", "pdf", "svg")]]
        if any(not path.is_file() or path.stat().st_size == 0 for path in required): raise RuntimeError("axis-anchored readiness evidence incomplete")
        if peak_rss_kib is None or peak_rss_kib > 8 * 1024 * 1024: raise RuntimeError(f"RAM contract failed: {peak_rss_kib}")
        after = {relative: sha256(PROJECT_ROOT / relative) for relative in before}
        if before != after: raise RuntimeError("frozen axis-anchored source changed")
        overall = expected_pass
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"; (run / "logs/failure_traceback.log").write_text(traceback.format_exc(), encoding="utf-8")
    write_json(run / "metrics/summary.json", {"schema_version": "gse_axis_anchored_event_relation_readiness_outer_v1", "overall_status": overall, "scientific_pass": overall == expected_pass, "error": error, "unit_test_returncode": tests, "executor_returncode": executor, "verified_source_seal_entries": verified_entries, "peak_host_rss_kib": peak_rss_kib, "result": result, "duration_seconds": time.monotonic() - started, "source_unchanged": bool(before and before == after), "optimizer_steps": 0, "checkpoints_created": 0, "new_model_inference_observations": 0, "c09_worlds_read": 0, "c10_worlds_read": 0, "mtare_worlds_read": 0, "graph_replays": 0, "planner_calls": 0})
    write_json(run / "RUN_STATE.json", {"schema_version": "v3_run_state_v1", "run_id": run_id, "state": "COMPLETED" if overall == expected_pass and error is None else "FAILED", "overall_status": overall, "error": error})
    entries = seal(run)
    print(json.dumps({"overall_status": overall, "error": error, "decision": result.get("decision"), "verified_source_entries": verified_entries, "seal_entries": entries}, indent=2))
    return 0 if overall == expected_pass and error is None else 2


if __name__ == "__main__": raise SystemExit(main())
