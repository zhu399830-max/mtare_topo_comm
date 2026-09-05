#!/usr/bin/env python3
"""Execute once and seal the complete deduplicated GSE dataset export."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, write_json


RUN_ID = "gate2_20260824_gse_deduplicated_dataset_export_v1_seed0"
PASS_STATUS = "PASS_GSE_DEDUPLICATED_DATASET_EXPORT_V1"
SIDECAR = Path("/tmp/mtare_gate4_meshing_sidecar_v1/bin/python")
EXECUTOR = PROJECT_ROOT / "tools/v3/execute_gse_deduplicated_dataset_export_v1.py"
REGISTRY = PROJECT_ROOT / "results/gate1_data/gate1_20260812_cano_phase2_supervised_range_dataset_v2r_seed0/config/world_registry.json"
MESH_RUN = PROJECT_ROOT / "results/gate0_baseline/gate0_20260811_cano_100_parent_perception_mesh_m1r_sanitized_assets_seed0"
MESH_ROOT = MESH_RUN / "artifacts/meshes"
SOURCE_SEAL = MESH_RUN / "artifacts/evidence_sha256.txt"
MANIFEST_RUN = PROJECT_ROOT / "results/gate2_representation/gate2_20260824_gse_teacher_manifest_v1_seed0"
EXIT_RUN = PROJECT_ROOT / "results/gate2_representation/gate2_20260824_gse_exit_token_audit_v1r_seed0"
TIME_LIMIT_SECONDS = 10800
DISK_LIMIT_BYTES = 24 * 1024**3
RSS_LIMIT_KIB = 8 * 1024**2


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _development_rows() -> list[dict]:
    registry = load_json(REGISTRY)
    schema = registry["sampling_contract"]["row_schema"]
    rows = [
        dict(zip(schema, row, strict=True)) if isinstance(row, list) else dict(row)
        for row in registry["rows"]
    ]
    selected = sorted(
        (row for row in rows if row["split"] in {"train", "validation"}),
        key=lambda row: str(row["parent_id"]),
    )
    if (
        sum(row["split"] == "train" for row in selected) != 80
        or sum(row["split"] == "validation" for row in selected) != 10
        or any(str(row["parent_id"]).endswith("_C10") for row in selected)
    ):
        raise RuntimeError("development registry scope mismatch")
    return selected


def _verify_sources() -> dict:
    entries = {}
    for line in SOURCE_SEAL.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        entries[relative] = expected
    files = []
    for row in _development_rows():
        parent_id = str(row["parent_id"])
        for filename in ("graph.json", "splines.json", "geometry_parameters.json", "mesh.obj"):
            path = MESH_ROOT / parent_id / "primary" / filename
            relative = str(path.relative_to(PROJECT_ROOT))
            observed = _sha256(path)
            if entries.get(relative) != observed:
                raise RuntimeError(f"sealed development source mismatch: {relative}")
            files.append({"path": relative, "sha256": observed})
    return {
        "development_world_count": 90,
        "verified_file_count": len(files),
        "source_seal_sha256": _sha256(SOURCE_SEAL),
        "strict_test_asset_files_opened": 0,
        "mtare_asset_files_opened": 0,
        "files": files,
    }


def _verify_run(run_dir: Path, *, status: str, summary_key: str, summary_value: int) -> dict:
    state = load_json(run_dir / "RUN_STATE.json")
    summary = load_json(run_dir / "metrics/summary.json")
    seal = run_dir / "artifacts/evidence_sha256.txt"
    if (
        state.get("state") != "COMPLETED"
        or state.get("overall_status") != status
        or summary.get("overall_status") != status
        or (
            summary.get(summary_key) != summary_value
            and summary.get("totals", {}).get(summary_key) != summary_value
        )
    ):
        raise RuntimeError(f"upstream run is not the required PASS: {run_dir.name}")
    checked = 0
    for line in seal.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        if _sha256(PROJECT_ROOT / relative) != expected:
            raise RuntimeError(f"upstream seal mismatch: {relative}")
        checked += 1
    return {
        "run": str(run_dir.relative_to(PROJECT_ROOT)),
        "seal_sha256": _sha256(seal),
        "summary_sha256": _sha256(run_dir / "metrics/summary.json"),
        "verified_seal_entries": checked,
    }


def _seal(run_dir: Path) -> int:
    destination = run_dir / "artifacts/evidence_sha256.txt"
    files = sorted(path for path in run_dir.rglob("*") if path.is_file() and path != destination)
    with destination.open("w", encoding="utf-8") as stream:
        for path in files:
            stream.write(f"{_sha256(path)}  {path.relative_to(PROJECT_ROOT)}\n")
    return len(files)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    spec = load_json(args.spec.resolve())
    run_dir = args.run_dir.resolve()
    if run_dir.name != RUN_ID or load_json(run_dir / "RUN_STATE.json").get("state") != "CREATED_NOT_EXECUTED":
        raise RuntimeError("one-time run state mismatch")
    if spec.get("gate") != 2 or spec.get("operation") != "teacher_generation" or spec.get("seed") != 0:
        raise RuntimeError("deduplicated dataset export scope mismatch")
    if spec.get("user_authorization", {}).get("status") != "APPROVED":
        raise RuntimeError("deduplicated dataset export is not authorized")
    card = load_json(PROJECT_ROOT / spec["data_card"])
    if card.get("status") != "APPROVED_FOR_ONE_IMMUTABLE_FORMAL_EXPORT":
        raise RuntimeError("operation-bound Data Card status mismatch")
    for name, record in spec["frozen_tools"].items():
        if _sha256(PROJECT_ROOT / record["path"]) != record["sha256"]:
            raise RuntimeError(f"frozen tool mismatch: {name}")
    for raw_path, expected in spec["frozen_inputs"].items():
        if _sha256(PROJECT_ROOT / raw_path) != expected:
            raise RuntimeError(f"frozen input mismatch: {raw_path}")
    environment = json.loads(
        subprocess.check_output(
            [
                str(SIDECAR),
                "-c",
                "import json,numcodecs,numpy,open3d,sys,zarr;print(json.dumps({'python':sys.version.split()[0],'numpy':numpy.__version__,'open3d':open3d.__version__,'zarr':zarr.__version__,'numcodecs':numcodecs.__version__},sort_keys=True))",
            ],
            text=True,
        )
    )
    expected_environment = {
        "python": "3.12.3",
        "numpy": "1.26.4",
        "open3d": "0.19.0",
        "zarr": "2.18.7",
        "numcodecs": "0.15.1",
    }
    if environment != expected_environment:
        raise RuntimeError(f"sidecar drift: {environment}")
    if shutil.disk_usage(PROJECT_ROOT).free < 32 * 1024**3:
        raise RuntimeError("less than 32 GiB free")

    source_before = _verify_sources()
    manifest = _verify_run(
        MANIFEST_RUN,
        status="PASS_GSE_TEACHER_MANIFEST_V1",
        summary_key="total_observation_count",
        summary_value=212588,
    )
    exit_audit = _verify_run(
        EXIT_RUN,
        status="PASS_GSE_EXIT_TOKEN_AUDIT_V1",
        summary_key="candidate_count",
        summary_value=448338,
    )
    write_json(
        run_dir / "config/environment.json",
        {
            "executable": str(SIDECAR),
            "versions": environment,
            "cpu_only": True,
            "source_verification_before": source_before,
            "upstream_teacher_manifest": manifest,
            "upstream_exit_audit": exit_audit,
        },
    )
    write_json(
        run_dir / "RUN_STATE.json",
        {
            "schema_version": "v3_run_state_v1",
            "run_id": RUN_ID,
            "state": "RUNNING",
            "note": "Unique-frame LiDAR and complete masked GSE teacher export; zero training or test reads.",
        },
    )
    command = [
        "/usr/bin/time",
        "-v",
        str(SIDECAR),
        str(EXECUTOR),
        "--run-dir",
        str(run_dir),
        "--manifest-run",
        str(MANIFEST_RUN),
        "--exit-audit-run",
        str(EXIT_RUN),
        "--mesh-root",
        str(MESH_ROOT),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT / "src") + os.pathsep + str(PROJECT_ROOT / "tools/v3")
    env["OMP_NUM_THREADS"] = "4"
    env["MKL_NUM_THREADS"] = "4"
    started = time.monotonic()
    code = 124
    output = ""
    try:
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=TIME_LIMIT_SECONDS,
            check=False,
        )
        code = completed.returncode
        output = completed.stdout
    except subprocess.TimeoutExpired as exc:
        output = (exc.stdout.decode() if isinstance(exc.stdout, bytes) else exc.stdout or "") + "\nTIMEOUT\n"
    duration = time.monotonic() - started
    match = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", output)
    peak_rss_kib = int(match.group(1)) if match else None
    (run_dir / "logs/01_deduplicated_dataset_export.log").write_text(
        output + f"\nduration_seconds={duration:.6f}\nexit_code={code}\n",
        encoding="utf-8",
    )
    print(output, end="")
    source_after = _verify_sources()
    summary_path = run_dir / "metrics/summary.json"
    summary = load_json(summary_path) if summary_path.is_file() else {}
    required = (
        run_dir / "artifacts/dataset",
        run_dir / "artifacts/sequence_manifest.jsonl",
        run_dir / "artifacts/frame_manifest.jsonl",
        run_dir / "artifacts/association_pairs_numeric.jsonl",
        run_dir / "artifacts/association_identity_map.json",
        run_dir / "artifacts/exit_identity_map.json",
        run_dir / "artifacts/shard_manifest.json",
    )
    result_bytes = sum(path.stat().st_size for path in run_dir.rglob("*") if path.is_file())
    passed = bool(
        code == 0
        and duration <= TIME_LIMIT_SECONDS
        and peak_rss_kib is not None
        and peak_rss_kib <= RSS_LIMIT_KIB
        and summary.get("overall_status") == PASS_STATUS
        and summary.get("global_totals", {}).get("full_scan_rays") == 3284444160
        and summary.get("strict_test_worlds_read") == 0
        and summary.get("mtare_worlds_read") == 0
        and summary.get("training_samples_consumed") == 0
        and all(path.exists() for path in required)
        and source_before == source_after
        and result_bytes <= DISK_LIMIT_BYTES
    )
    overall = PASS_STATUS if passed else "FAIL_GSE_DEDUPLICATED_DATASET_EXPORT_V1"
    write_json(
        run_dir / "metrics/runner_summary.json",
        {
            "overall_status": overall,
            "executor_exit_code": code,
            "duration_seconds": duration,
            "peak_rss_kib": peak_rss_kib,
            "rss_limit_kib": RSS_LIMIT_KIB,
            "source_verification_before": source_before,
            "source_verification_after": source_after,
            "source_unchanged": source_before == source_after,
            "upstream_teacher_manifest": manifest,
            "upstream_exit_audit": exit_audit,
            "result_bytes_before_seal": result_bytes,
            "disk_limit_bytes": DISK_LIMIT_BYTES,
            "strict_test_worlds_read": 0,
            "mtare_worlds_read": 0,
            "model_inference_frames": 0,
            "training_samples_consumed": 0,
            "optimizer_steps": 0,
        },
    )
    write_json(
        run_dir / "RUN_STATE.json",
        {
            "schema_version": "v3_run_state_v1",
            "run_id": RUN_ID,
            "state": "COMPLETED" if passed else "FAILED",
            "overall_status": overall,
        },
    )
    sealed_files = _seal(run_dir)
    print(json.dumps({"overall_status": overall, "sealed_files": sealed_files}, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
