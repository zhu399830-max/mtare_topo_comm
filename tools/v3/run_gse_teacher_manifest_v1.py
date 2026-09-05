#!/usr/bin/env python3
"""Execute once and seal the development GSE teacher/association manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, write_json


RUN_ID = "gate2_20260824_gse_teacher_manifest_v1_seed0"
PASS_STATUS = "PASS_GSE_TEACHER_MANIFEST_V1"
SIDECAR = Path("/tmp/mtare_gate4_meshing_sidecar_v1/bin/python")
EXECUTOR = PROJECT_ROOT / "tools/v3/execute_gse_teacher_manifest_v1.py"
REGISTRY = PROJECT_ROOT / "results/gate1_data/gate1_20260812_cano_phase2_supervised_range_dataset_v2r_seed0/config/world_registry.json"
MESH_RUN = PROJECT_ROOT / "results/gate0_baseline/gate0_20260811_cano_100_parent_perception_mesh_m1r_sanitized_assets_seed0"
MESH_ROOT = MESH_RUN / "artifacts/meshes"
SOURCE_SEAL = MESH_RUN / "artifacts/evidence_sha256.txt"
PROOF_RUN = PROJECT_ROOT / "results/gate2_representation/gate2_20260824_gse_mesh_teacher_distribution_v1r_seed0"
TIME_LIMIT_SECONDS = 1200
DISK_LIMIT_BYTES = 512 * 1024 * 1024


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _development_rows() -> list[dict]:
    registry = load_json(REGISTRY)
    schema = registry["sampling_contract"]["row_schema"]
    rows = [dict(zip(schema, row, strict=True)) if isinstance(row, list) else dict(row) for row in registry["rows"]]
    selected = sorted(
        (row for row in rows if row["split"] in {"train", "validation"}),
        key=lambda row: str(row["parent_id"]),
    )
    counts = {split: sum(row["split"] == split for row in selected) for split in ("train", "validation")}
    if counts != {"train": 80, "validation": 10} or any(
        str(row["parent_id"]).endswith("_C10") for row in selected
    ):
        raise RuntimeError("development registry scope mismatch")
    return selected


def _verify_sources() -> dict:
    entries: dict[str, str] = {}
    for line in SOURCE_SEAL.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        entries[relative] = expected
    checked: list[dict] = []
    for row in _development_rows():
        parent_id = str(row["parent_id"])
        for filename in ("graph.json", "splines.json", "geometry_parameters.json", "mesh.obj"):
            path = MESH_ROOT / parent_id / "primary" / filename
            relative = str(path.relative_to(PROJECT_ROOT))
            observed = _sha256(path)
            if entries.get(relative) != observed:
                raise RuntimeError(f"sealed development source mismatch: {relative}")
            checked.append({"path": relative, "sha256": observed})
    return {
        "development_world_count": 90,
        "train_world_count": 80,
        "validation_world_count": 10,
        "verified_file_count": len(checked),
        "strict_test_asset_files_opened": 0,
        "mtare_asset_files_opened": 0,
        "source_seal_sha256": _sha256(SOURCE_SEAL),
        "files": checked,
    }


def _verify_teacher_proof() -> dict:
    state = load_json(PROOF_RUN / "RUN_STATE.json")
    summary = load_json(PROOF_RUN / "metrics/summary.json")
    seal = PROOF_RUN / "artifacts/evidence_sha256.txt"
    if state.get("state") != "COMPLETED" or state.get("overall_status") != "PASS_GSE_MESH_TEACHER_DISTRIBUTION_V1R":
        raise RuntimeError("upstream mesh-teacher proof is not PASS")
    if summary.get("event_counts") != {
        "corridor": 135123,
        "geometry_transition": 16867,
        "junction": 26608,
        "terminal": 7525,
        "turn": 2003,
    }:
        raise RuntimeError("upstream train event distribution drift")
    return {
        "run": str(PROOF_RUN.relative_to(PROJECT_ROOT)),
        "seal_sha256": _sha256(seal),
        "summary_sha256": _sha256(PROOF_RUN / "metrics/summary.json"),
    }


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
        raise RuntimeError("one-time run state mismatch")
    if spec.get("gate") != 2 or spec.get("operation") != "audit" or spec.get("seed") != 0:
        raise RuntimeError("teacher manifest run scope mismatch")
    if spec.get("user_authorization", {}).get("status") != "APPROVED":
        raise RuntimeError("teacher manifest run is not authorized")
    card = load_json(PROJECT_ROOT / spec["data_card"])
    if card.get("status") != "APPROVED_FOR_ONE_IMMUTABLE_FORMAL_AUDIT":
        raise RuntimeError("operation-bound Data Card status mismatch")
    for name, record in spec["frozen_tools"].items():
        if _sha256(PROJECT_ROOT / record["path"]) != record["sha256"]:
            raise RuntimeError(f"frozen tool mismatch: {name}")
    for raw_path, expected in spec["frozen_inputs"].items():
        if _sha256(PROJECT_ROOT / raw_path) != expected:
            raise RuntimeError(f"frozen input mismatch: {raw_path}")
    if not SIDECAR.is_file():
        raise RuntimeError("frozen Open3D sidecar is missing")
    environment = json.loads(
        subprocess.check_output(
            [
                str(SIDECAR),
                "-c",
                "import json,numpy,open3d,sys;print(json.dumps({'python':sys.version.split()[0],'numpy':numpy.__version__,'open3d':open3d.__version__},sort_keys=True))",
            ],
            text=True,
        )
    )
    if environment != {"python": "3.12.3", "numpy": "1.26.4", "open3d": "0.19.0"}:
        raise RuntimeError(f"sidecar drift: {environment}")
    if shutil.disk_usage(PROJECT_ROOT).free < 2 * 1024**3:
        raise RuntimeError("less than 2 GiB free")
    source_before = _verify_sources()
    proof = _verify_teacher_proof()
    write_json(
        run_dir / "config/environment.json",
        {
            "executable": str(SIDECAR),
            "versions": environment,
            "cpu_only": True,
            "source_verification_before": source_before,
            "upstream_teacher_proof": proof,
        },
    )
    write_json(
        run_dir / "RUN_STATE.json",
        {
            "schema_version": "v3_run_state_v1",
            "run_id": RUN_ID,
            "state": "RUNNING",
            "note": "Development traversal/teacher/association manifest; zero sensor export or training.",
        },
    )
    command = [
        str(SIDECAR),
        str(EXECUTOR),
        "--run-dir",
        str(run_dir),
        "--registry",
        str(REGISTRY),
        "--mesh-root",
        str(MESH_ROOT),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT / "src") + os.pathsep + str(PROJECT_ROOT / "tools/v3")
    env["OMP_NUM_THREADS"] = "2"
    env["MKL_NUM_THREADS"] = "2"
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
    (run_dir / "logs/01_teacher_manifest.log").write_text(
        output + f"\nduration_seconds={duration:.6f}\nexit_code={code}\n",
        encoding="utf-8",
    )
    print(output, end="")
    source_after = _verify_sources()
    summary_path = run_dir / "metrics/summary.json"
    summary = load_json(summary_path) if summary_path.is_file() else {}
    required = (
        run_dir / "artifacts/traversal_manifest.jsonl",
        run_dir / "artifacts/teacher_observations.jsonl",
        run_dir / "artifacts/association_pairs.jsonl",
        run_dir / "artifacts/world_manifest_summary.jsonl",
        run_dir / "artifacts/edge_event_identities.jsonl",
        run_dir / "artifacts/resource_estimate.json",
    )
    result_bytes = sum(path.stat().st_size for path in run_dir.rglob("*") if path.is_file())
    passed = bool(
        code == 0
        and duration <= TIME_LIMIT_SECONDS
        and summary.get("overall_status") == PASS_STATUS
        and summary.get("total_observation_count") == 212588
        and summary.get("strict_test_worlds_read") == 0
        and summary.get("mtare_worlds_read") == 0
        and summary.get("training_samples_consumed") == 0
        and all(path.is_file() for path in required)
        and source_before == source_after
        and result_bytes <= DISK_LIMIT_BYTES
    )
    overall = PASS_STATUS if passed else "FAIL_GSE_TEACHER_MANIFEST_V1"
    write_json(
        run_dir / "metrics/runner_summary.json",
        {
            "overall_status": overall,
            "executor_exit_code": code,
            "duration_seconds": duration,
            "source_verification_before": source_before,
            "source_verification_after": source_after,
            "source_unchanged": source_before == source_after,
            "upstream_teacher_proof": proof,
            "result_bytes_before_seal": result_bytes,
            "disk_limit_bytes": DISK_LIMIT_BYTES,
            "model_inference_frames": 0,
            "training_samples_consumed": 0,
            "optimizer_steps": 0,
            "strict_test_worlds_read": 0,
            "mtare_worlds_read": 0,
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
