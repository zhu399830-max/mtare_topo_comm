#!/usr/bin/env python3
"""Run once and seal the C01-C08 causal change-point Teacher proof."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
import time

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, write_json


RUN_ID = "gate3_20260826_gse_causal_change_point_proof_v1_seed0"
PASS_STATUS = "PASS_GSE_CAUSAL_CHANGE_POINT_PROOF_V1"
FAIL_STATUS = "FAIL_GSE_CAUSAL_CHANGE_POINT_PROOF_V1"
SIDECAR = Path("/tmp/mtare_gate4_meshing_sidecar_v1/bin/python")
EXECUTOR = PROJECT_ROOT / "tools/v3/execute_gse_causal_change_point_proof_v1.py"
REGISTRY = PROJECT_ROOT / "results/gate1_data/gate1_20260812_cano_phase2_supervised_range_dataset_v2r_seed0/config/world_registry.json"
MESH_RUN = PROJECT_ROOT / "results/gate0_baseline/gate0_20260811_cano_100_parent_perception_mesh_m1r_sanitized_assets_seed0"
MESH_ROOT = MESH_RUN / "artifacts/meshes"
MESH_SEAL = MESH_RUN / "artifacts/evidence_sha256.txt"
OLD_TEACHER_RUN = PROJECT_ROOT / "results/gate2_representation/gate2_20260824_gse_teacher_manifest_v1_seed0"
TIME_LIMIT_SECONDS = 600
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
    decoded = [
        dict(zip(schema, row, strict=True)) if isinstance(row, list) else dict(row)
        for row in registry["rows"]
    ]
    selected = sorted(
        (
            row
            for row in decoded
            if row["split"] == "train"
            and str(row["parent_id"]).endswith(tuple(f"_C{index:02d}" for index in range(1, 9)))
        ),
        key=lambda row: str(row["parent_id"]),
    )
    if len(selected) != 80 or any(str(row["parent_id"]).endswith(("_C09", "_C10")) for row in selected):
        raise RuntimeError("C01-C08 registry scope mismatch")
    return selected


def _verify_sources() -> dict:
    seal_entries: dict[str, str] = {}
    for line in MESH_SEAL.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        seal_entries[relative] = expected
    files: list[dict[str, str]] = []
    for row in _development_rows():
        parent_id = str(row["parent_id"])
        for filename in ("graph.json", "splines.json", "geometry_parameters.json", "mesh.obj"):
            path = MESH_ROOT / parent_id / "primary" / filename
            relative = str(path.relative_to(PROJECT_ROOT))
            observed = _sha256(path)
            if seal_entries.get(relative) != observed:
                raise RuntimeError(f"sealed C01-C08 source mismatch: {relative}")
            files.append({"path": relative, "sha256": observed})
    old_state = load_json(OLD_TEACHER_RUN / "RUN_STATE.json")
    if old_state.get("state") != "COMPLETED" or old_state.get("overall_status") != "PASS_GSE_TEACHER_MANIFEST_V1":
        raise RuntimeError("old Teacher comparison source is not sealed PASS")
    old_seal = OLD_TEACHER_RUN / "artifacts/evidence_sha256.txt"
    return {
        "world_count": 80,
        "verified_mesh_asset_file_count": len(files),
        "mesh_source_seal_sha256": _sha256(MESH_SEAL),
        "old_teacher_seal_sha256": _sha256(old_seal),
        "old_teacher_state_sha256": _sha256(OLD_TEACHER_RUN / "RUN_STATE.json"),
        "old_teacher_identity_manifest_sha256": _sha256(
            OLD_TEACHER_RUN / "artifacts/edge_event_identities.jsonl"
        ),
        "files": files,
        "c09_records_consumed": 0,
        "c10_records_consumed": 0,
        "mtare_records_consumed": 0,
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
    spec_path = args.spec.resolve()
    run_dir = args.run_dir.resolve()
    spec = load_json(spec_path)
    if run_dir.name != RUN_ID:
        raise RuntimeError("unexpected run identity")
    state = load_json(run_dir / "RUN_STATE.json")
    if state.get("state") != "CREATED_NOT_EXECUTED":
        raise RuntimeError("causal change-point proof may execute only once")
    if spec.get("gate") != 3 or spec.get("operation") != "audit" or spec.get("seed") != 0:
        raise RuntimeError("run spec scope mismatch")
    if spec.get("user_authorization", {}).get("status") != "APPROVED":
        raise RuntimeError("research decision A is not bound to the run spec")
    card = load_json(PROJECT_ROOT / spec["data_card"])
    if card.get("approval", {}).get("status") != "APPROVED_FOR_ONE_IMMUTABLE_PROOF":
        raise RuntimeError("operation-bound Data Card is not approved")
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
                "import json,matplotlib,numpy,open3d,sys;print(json.dumps({'python':sys.version.split()[0],'numpy':numpy.__version__,'open3d':open3d.__version__,'matplotlib':matplotlib.__version__},sort_keys=True))",
            ],
            text=True,
        )
    )
    expected_environment = {
        "python": "3.12.3",
        "numpy": "1.26.4",
        "open3d": "0.19.0",
        "matplotlib": "3.11.1",
    }
    if environment != expected_environment:
        raise RuntimeError(f"sidecar drift: {environment}")
    if shutil.disk_usage(PROJECT_ROOT).free < 2 * 1024**3:
        raise RuntimeError("less than 2 GiB free")

    source_before = _verify_sources()
    write_json(
        run_dir / "config/environment.json",
        {
            "executable": str(SIDECAR),
            "versions": environment,
            "cpu_only": True,
            "source_verification_before": source_before,
        },
    )
    write_json(
        run_dir / "RUN_STATE.json",
        {
            "schema_version": "v3_run_state_v1",
            "run_id": RUN_ID,
            "state": "RUNNING",
            "note": "C01-C08 read-only causal geometry change-point proof; zero training or model inference.",
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
        "--old-teacher-run",
        str(OLD_TEACHER_RUN),
    ]
    (run_dir / "config/command.txt").write_text(" ".join(command) + "\n", encoding="utf-8")
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
    (run_dir / "logs/01_causal_change_point_proof.log").write_text(
        output + f"\nduration_seconds={duration:.6f}\nexit_code={code}\n",
        encoding="utf-8",
    )
    print(output, end="")
    source_after = _verify_sources()
    summary_path = run_dir / "metrics/summary.json"
    summary = load_json(summary_path) if summary_path.is_file() else {}
    required = (
        run_dir / "artifacts/world_change_point_summary.jsonl",
        run_dir / "artifacts/bidirectional_change_points.jsonl",
        run_dir / "artifacts/directional_persistent_episodes.jsonl",
        run_dir / "artifacts/past_only_causal_episodes.jsonl",
        run_dir / "artifacts/causal_change_point_labels.jsonl",
        run_dir / "artifacts/old_new_identity_mapping.jsonl",
        run_dir / "artifacts/family_summary.csv",
        run_dir / "previews/gse_causal_change_point_proof.png",
        run_dir / "previews/gse_causal_change_point_proof.pdf",
        run_dir / "previews/gse_causal_change_point_proof.svg",
        run_dir / "previews/gse_causal_change_point_proof_source.json",
        run_dir / "previews/gse_causal_change_point_proof_provenance.json",
    )
    result_bytes = sum(path.stat().st_size for path in run_dir.rglob("*") if path.is_file())
    passed = bool(
        code == 0
        and duration <= TIME_LIMIT_SECONDS
        and summary.get("overall_status") == PASS_STATUS
        and summary.get("scope", {}).get("worlds") == 80
        and summary.get("c09_worlds_consumed") == 0
        and summary.get("strict_test_worlds_read") == 0
        and summary.get("mtare_worlds_read") == 0
        and summary.get("model_inference_frames") == 0
        and summary.get("optimizer_steps") == 0
        and all(path.is_file() for path in required)
        and source_before == source_after
        and result_bytes <= DISK_LIMIT_BYTES
    )
    overall = PASS_STATUS if passed else FAIL_STATUS
    write_json(
        run_dir / "metrics/runner_summary.json",
        {
            "overall_status": overall,
            "executor_exit_code": code,
            "duration_seconds": duration,
            "source_verification_before": source_before,
            "source_verification_after": source_after,
            "source_unchanged": source_before == source_after,
            "result_bytes_before_seal": result_bytes,
            "disk_limit_bytes": DISK_LIMIT_BYTES,
            "c09_worlds_consumed": 0,
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
