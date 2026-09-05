#!/usr/bin/env python3
"""Execute topology comparison with the risk-calibrated GSE observation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import time
import traceback

from _bootstrap import PROJECT_ROOT
from mtare_topo.evaluation.gse_evidence_integrity import (
    verify_complete_run_seal,
    verify_failed_component_run_seal,
)
from mtare_topo.governance import load_json, write_json


RUN_ID = "gate4_20260825_gse_offline_topology_validation_v2_seed0"
PASS_STATUS = "PASS_GSE_OFFLINE_TOPOLOGY_VALIDATION_V2"
FAIL_STATUS = "FAIL_GSE_OFFLINE_TOPOLOGY_VALIDATION_V2"
PYTHON = Path("/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python")
TRAINING = PROJECT_ROOT / "results/gate2_representation/gate2_20260824_gse_graph_three_seed_training_v1r_seed0"
PERCEPTION_COMPONENT = PROJECT_ROOT / "results/gate3_semantics/gate3_20260824_gse_perception_validation_v1_seed0"
CORRECTED_PERCEPTION = PROJECT_ROOT / "results/gate3_semantics/gate3_20260825_gse_risk_calibrated_perception_validation_v2_seed0"
DATASET = PROJECT_ROOT / "results/gate2_representation/gate2_20260824_gse_deduplicated_dataset_export_v1_seed0"
TEACHER = PROJECT_ROOT / "results/gate2_representation/gate2_20260824_gse_teacher_manifest_v1_seed0"
EXPECTED_SOURCE_STATUS = {
    TRAINING: "PASS_GSE_GRAPH_THREE_SEED_TRAINING_V1R",
    CORRECTED_PERCEPTION: "PASS_GSE_RISK_CALIBRATED_PERCEPTION_VALIDATION_V2",
    DATASET: "PASS_GSE_DEDUPLICATED_DATASET_EXPORT_V1",
    TEACHER: "PASS_GSE_TEACHER_MANIFEST_V1",
}
DISK_LIMIT_BYTES = 5 * 1024**3
RSS_LIMIT_KIB = 16 * 1024**2


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_sealed_source(run: Path, expected_status: str) -> dict:
    return verify_complete_run_seal(PROJECT_ROOT, run, expected_status)


def _sources() -> dict:
    result = {
        run.name: _verify_sealed_source(run, expected)
        for run, expected in EXPECTED_SOURCE_STATUS.items()
    }
    result[PERCEPTION_COMPONENT.name] = verify_failed_component_run_seal(
        PROJECT_ROOT, PERCEPTION_COMPONENT, "FAIL_GSE_PERCEPTION_VALIDATION_V1"
    )
    component_gate = load_json(PERCEPTION_COMPONENT / "metrics/perception_gate.json")
    corrected_gate = load_json(CORRECTED_PERCEPTION / "metrics/corrected_perception_gate.json")
    if (
        component_gate.get("event_gate", {}).get("passed") is not True
        or component_gate.get("association_gate", {}).get("passed") is not True
        or corrected_gate.get("passed") is not True
    ):
        raise RuntimeError("offline topology component/corrective source contract drift")
    return result


def _seal(run_dir: Path) -> int:
    destination = run_dir / "artifacts/evidence_sha256.txt"
    files = sorted(path for path in run_dir.rglob("*") if path.is_file() and path != destination)
    destination.write_text(
        "".join(f"{_sha256(path)}  {path.relative_to(PROJECT_ROOT)}\n" for path in files),
        encoding="utf-8",
    )
    return len(files)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    spec = load_json(args.spec.resolve())
    run_dir = args.run_dir.resolve()
    if run_dir.name != RUN_ID or load_json(run_dir / "RUN_STATE.json").get("state") != "CREATED_NOT_EXECUTED":
        raise RuntimeError("one-time offline topology run identity mismatch")
    if (
        spec.get("gate") != 4
        or spec.get("operation") != "topology_replay"
        or spec.get("user_authorization", {}).get("status") != "APPROVED"
    ):
        raise RuntimeError("offline topology scope is not authorized")
    card = load_json(PROJECT_ROOT / spec["data_card"])
    if card.get("status") != "APPROVED_FOR_ONE_IMMUTABLE_GSE_OFFLINE_TOPOLOGY_VALIDATION_V2":
        raise RuntimeError("operation-bound offline topology Data Card mismatch")
    for name, record in spec["frozen_tools"].items():
        if _sha256(PROJECT_ROOT / record["path"]) != record["sha256"]:
            raise RuntimeError(f"frozen topology tool drift: {name}")
    for relative, expected in spec["frozen_inputs"].items():
        if _sha256(PROJECT_ROOT / relative) != expected:
            raise RuntimeError(f"frozen topology input drift: {relative}")

    started = time.monotonic()
    overall = FAIL_STATUS
    error = None
    source_before = None
    source_after = None
    evaluator_summary: dict = {}
    result_bytes = 0
    maximum_child_rss_kib = 0
    try:
        if shutil.disk_usage(PROJECT_ROOT).free < 8 * 1024**3:
            raise RuntimeError("less than 8 GiB free before offline topology replay")
        source_before = _sources()
        write_json(run_dir / "RUN_STATE.json", {"schema_version": "v3_run_state_v1", "run_id": RUN_ID, "state": "RUNNING"})
        environment = json.loads(
            subprocess.check_output(
                [
                    str(PYTHON),
                    "-c",
                    "import json,numpy,torch,zarr,sys;print(json.dumps({'python':sys.version.split()[0],'numpy':numpy.__version__,'torch':torch.__version__,'cuda':torch.version.cuda,'zarr':zarr.__version__},sort_keys=True))",
                ],
                text=True,
            )
        )
        write_json(run_dir / "config/environment.json", environment)
        output_dir = run_dir / "artifacts/offline_topology"
        command = [
            str(PYTHON),
            str(PROJECT_ROOT / "tools/v3/evaluate_gse_offline_topology_v2.py"),
            "--training-run", str(TRAINING),
            "--perception-run", str(PERCEPTION_COMPONENT),
            "--corrected-perception-run", str(CORRECTED_PERCEPTION),
            "--dataset-run", str(DATASET),
            "--teacher-run", str(TEACHER),
            "--output-dir", str(output_dir),
        ]
        env = os.environ.copy()
        env["PYTHONPATH"] = str(PROJECT_ROOT / "src") + os.pathsep + str(PROJECT_ROOT / "tools/v3")
        env["OMP_NUM_THREADS"] = "2"
        env["MKL_NUM_THREADS"] = "2"
        with (run_dir / "logs/01_offline_topology.log").open("w", encoding="utf-8") as stream:
            completed = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                env=env,
                text=True,
                stdout=stream,
                stderr=subprocess.STDOUT,
                timeout=8 * 60 * 60,
                check=False,
            )
        if (output_dir / "summary.json").is_file():
            evaluator_summary = load_json(output_dir / "summary.json")
        result_bytes = sum(path.stat().st_size for path in output_dir.rglob("*") if path.is_file())
        maximum_child_rss_kib = int(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
        if result_bytes > DISK_LIMIT_BYTES:
            raise RuntimeError("offline topology evidence exceeded the frozen 5 GiB limit")
        if maximum_child_rss_kib > RSS_LIMIT_KIB:
            raise RuntimeError("offline topology replay exceeded the frozen 16 GiB child RSS limit")
        if completed.returncode != 0 or evaluator_summary.get("overall_status") != PASS_STATUS:
            raise RuntimeError("GSE offline topology failed the pre-registered scientific gate")
        source_after = _sources()
        if source_before != source_after:
            raise RuntimeError("source evidence changed during topology replay")
        overall = PASS_STATUS
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        (run_dir / "logs/failure_traceback.log").write_text(traceback.format_exc(), encoding="utf-8")

    result_bytes = sum(path.stat().st_size for path in run_dir.rglob("*") if path.is_file())
    maximum_child_rss_kib = int(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)

    summary = {
        "schema_version": "gse_offline_topology_run_v2",
        "overall_status": overall,
        "error": error,
        "evaluator": evaluator_summary,
        "source_before": source_before,
        "source_after": source_after,
        "validation_worlds": 10,
        "validation_sequences": 24462,
        "strict_test_worlds_read": 0,
        "c10_worlds_read": 0,
        "mtare_worlds_read": 0,
        "optimizer_steps": 0,
        "model_updates": 0,
        "result_bytes_before_seal": result_bytes,
        "disk_limit_bytes": DISK_LIMIT_BYTES,
        "maximum_child_rss_kib": maximum_child_rss_kib,
        "rss_limit_kib": RSS_LIMIT_KIB,
        "duration_seconds": time.monotonic() - started,
    }
    write_json(run_dir / "metrics/summary.json", summary)
    write_json(
        run_dir / "RUN_STATE.json",
        {
            "schema_version": "v3_run_state_v1",
            "run_id": RUN_ID,
            "state": "COMPLETED" if overall == PASS_STATUS else "FAILED",
            "overall_status": overall,
            "error": error,
        },
    )
    sealed_files = _seal(run_dir)
    print(json.dumps({"overall_status": overall, "error": error, "sealed_files": sealed_files}, indent=2))
    return 0 if overall == PASS_STATUS else 2


if __name__ == "__main__":
    raise SystemExit(main())
