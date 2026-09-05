#!/usr/bin/env python3
"""Run and seal one immutable three-seed explicit Composer cache export."""

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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, validate_data_card, write_json


PASS = "PASS_GSE_EXPLICIT_COMPOSER_CACHE_EXPORT_V1"
FAIL = "FAIL_GSE_EXPLICIT_COMPOSER_CACHE_EXPORT_V1"
CARD_STATUS = "APPROVED_FOR_ONE_IMMUTABLE_GSE_EXPLICIT_COMPOSER_CACHE_EXPORT_V1"
PYTHON = Path("/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python")
DATASET = PROJECT_ROOT / "results/gate2_representation/gate2_20260824_gse_deduplicated_dataset_export_v1_seed0"
BASELINE = PROJECT_ROOT / "results/gate3_semantics/gate3_20260829_gse_sparse_circular_relation_transport_three_seed_training_v2r5_seed0"
READINESS = PROJECT_ROOT / "results/gate3_semantics/gate3_20260829_gse_typed_composer_readiness_v1_seed0"
DISK_LIMIT_BYTES = 4 * 1024**3
GPU_LIMIT_BYTES = 16 * 1024**3


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_run_seal(run: Path, expected_status: str) -> dict[str, object]:
    state = load_json(run / "RUN_STATE.json")
    summary = load_json(run / "metrics/summary.json")
    if (
        state.get("state") != "COMPLETED" or state.get("error") is not None
        or state.get("overall_status") != expected_status
        or summary.get("overall_status", summary.get("status")) != expected_status
    ):
        raise RuntimeError(f"required source status drift: {run.name}")
    seal = run / "artifacts/evidence_sha256.txt"
    covered = set()
    for line in seal.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        path = (PROJECT_ROOT / relative).resolve()
        path.relative_to(run.resolve())
        if path in covered or not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"source seal drift: {relative}")
        covered.add(path)
    actual = {
        path.resolve() for path in run.rglob("*")
        if path.is_file() and path.resolve() != seal.resolve()
    }
    if covered != actual:
        raise RuntimeError(f"source seal coverage drift: {run.name}")
    return {"run": str(run.relative_to(PROJECT_ROOT)), "entries": len(covered), "seal_sha256": sha256(seal)}


def seal(run: Path) -> int:
    target = run / "artifacts/evidence_sha256.txt"
    files = sorted(path for path in run.rglob("*") if path.is_file() and path != target)
    with target.open("w", encoding="utf-8") as stream:
        for path in files:
            stream.write(f"{sha256(path)}  {path.relative_to(PROJECT_ROOT)}\n")
    return len(files)


def _peak_rss(path: Path) -> int | None:
    match = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", path.read_text(encoding="utf-8"))
    return int(match.group(1)) if match else None


def _plot(output: Path, manifests: list[dict[str, object]]) -> None:
    seeds = [f"seed{manifest['seed']}" for manifest in manifests]
    sizes = [
        sum(int(record["bytes"]) for record in manifest["records"]) / 1024**2
        for manifest in manifests
    ]
    duration = [float(manifest["duration_seconds"]) / 60.0 for manifest in manifests]
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    axes[0].bar(seeds, sizes, color="#2b6cb0"); axes[0].set_ylabel("compressed MiB")
    axes[0].set_title("Geometry-only cache size")
    axes[1].bar(seeds, duration, color="#2f855a"); axes[1].set_ylabel("minutes")
    axes[1].set_title("Frozen inference duration")
    figure.suptitle("GSE explicit Composer cache export")
    figure.tight_layout()
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(output / f"gse_explicit_composer_cache_export_v1.{suffix}", dpi=220)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    spec = load_json(args.spec.resolve()); run = args.run_dir.resolve()
    run_id = f"gate{spec['gate']}_{spec['date']}_{spec['slug']}_seed{spec['seed']}"
    if run.name != run_id or load_json(run / "RUN_STATE.json").get("state") != "CREATED_NOT_EXECUTED":
        raise RuntimeError("explicit Composer cache export executes exactly once")
    started = time.monotonic(); overall = FAIL; error = None; before = {}; after = {}
    subprocesses = []; manifests = []; peak_rss_kib = 0; source_evidence = {}
    try:
        card = load_json(PROJECT_ROOT / spec["data_card"]); report = validate_data_card(card)
        if not report.passed or card.get("status") != CARD_STATUS:
            raise RuntimeError(f"explicit cache Data Card invalid: {report.errors}")
        for record in spec["frozen_tools"].values():
            if sha256(PROJECT_ROOT / record["path"]) != record["sha256"]:
                raise RuntimeError(f"tool drift: {record['path']}")
        for relative, expected in spec["frozen_inputs"].items():
            before[relative] = sha256(PROJECT_ROOT / relative)
            if before[relative] != expected:
                raise RuntimeError(f"input drift: {relative}")
        source_evidence["dataset"] = verify_run_seal(DATASET, "PASS_GSE_DEDUPLICATED_DATASET_EXPORT_V1")
        source_evidence["baseline"] = verify_run_seal(
            BASELINE, "FAIL_GSE_SPARSE_CIRCULAR_RELATION_TRANSPORT_THREE_SEED_TRAINING_V2",
        )
        source_evidence["readiness"] = verify_run_seal(READINESS, "PASS_GSE_TYPED_COMPOSER_READINESS_V1")
        environment = json.loads(subprocess.check_output([
            str(PYTHON), "-c",
            "import json,numpy,torch,zarr,sys;print(json.dumps({'python':sys.version.split()[0],'numpy':numpy.__version__,'torch':torch.__version__,'cuda':torch.version.cuda,'zarr':zarr.__version__,'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},sort_keys=True))",
        ], text=True))
        expected_environment = {
            "python": "3.13.5", "numpy": "2.1.3", "torch": "2.9.0+cu129",
            "cuda": "12.9", "zarr": "2.18.7", "gpu": "NVIDIA GeForce RTX 5090 D",
        }
        if environment != expected_environment:
            raise RuntimeError(f"environment drift: {environment}")
        write_json(run / "config/environment.json", {"executable": str(PYTHON), "versions": environment, "deterministic": True})
        write_json(run / "config/source_evidence.json", source_evidence)
        write_json(run / "config/source_integrity_before.json", before)
        write_json(run / "RUN_STATE.json", {"schema_version": "v3_run_state_v1", "run_id": run_id, "state": "RUNNING"})
        process_environment = os.environ.copy()
        process_environment["PYTHONPATH"] = str(PROJECT_ROOT / "src") + os.pathsep + str(PROJECT_ROOT / "tools/v3")
        process_environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        unit_log = run / "logs/00_unit_tests.log"
        with unit_log.open("w", encoding="utf-8") as stream:
            unit = subprocess.run([
                str(PYTHON), "-m", "pytest", "-q",
                "tests/v3/unit/test_gse_explicit_composer_cache.py",
                "tests/v3/unit/test_gse_typed_composers.py",
            ], cwd=PROJECT_ROOT, env=process_environment, stdout=stream, stderr=subprocess.STDOUT,
                text=True, timeout=600, check=False)
        subprocesses.append({"stage": "unit_tests", "returncode": int(unit.returncode)})
        if unit.returncode != 0:
            raise RuntimeError("explicit cache unit tests failed")
        cache_root = run / "artifacts/cache"; cache_root.mkdir()
        for seed in range(3):
            command = [
                str(PYTHON), str(PROJECT_ROOT / "tools/v3/export_gse_explicit_composer_cache_v1.py"),
                "--dataset-root", str(DATASET / "artifacts/dataset/train"),
                "--sequence-manifest", str(DATASET / "artifacts/sequence_manifest.jsonl"),
                "--checkpoint", str(BASELINE / f"artifacts/models/seed{seed}/best.pt"),
                "--development-prediction-root", str(BASELINE / f"artifacts/models/seed{seed}/development_predictions"),
                "--output-dir", str(cache_root / f"seed{seed}"), "--seed", str(seed),
            ]
            (run / f"config/seed{seed}_command.txt").write_text(" ".join(command) + "\n", encoding="utf-8")
            log = run / f"logs/0{seed + 1}_seed{seed}_export.log"
            with log.open("w", encoding="utf-8") as stream:
                completed = subprocess.run(
                    ["/usr/bin/time", "-v", *command], cwd=PROJECT_ROOT,
                    env=process_environment, stdout=stream, stderr=subprocess.STDOUT,
                    text=True, timeout=3600, check=False,
                )
            subprocesses.append({"stage": f"seed{seed}_export", "returncode": int(completed.returncode)})
            if completed.returncode != 0:
                raise RuntimeError(f"explicit cache seed{seed} export failed")
            current_peak = _peak_rss(log)
            peak_rss_kib = max(peak_rss_kib, current_peak or 0)
            manifest = load_json(cache_root / f"seed{seed}/manifest.json")
            if (
                manifest.get("seed") != seed or manifest.get("worlds") != 80
                or manifest.get("split_rows") != {"fit": 142184, "c07": 21548, "c08": 24394}
                or manifest.get("development_parity_worlds") != 20
                or manifest.get("maximum_development_parity_error") != 0.0
                or manifest.get("optimizer_steps") != 0 or manifest.get("checkpoints_created") != 0
                or manifest.get("c09_worlds_read") != 0 or manifest.get("c10_worlds_read") != 0
                or manifest.get("graph_replays") != 0
                or int(manifest.get("peak_gpu_memory_bytes", GPU_LIMIT_BYTES + 1)) > GPU_LIMIT_BYTES
            ):
                raise RuntimeError(f"explicit cache seed{seed} manifest contract failed")
            manifests.append(manifest)
        disk_bytes = sum(path.stat().st_size for path in run.rglob("*") if path.is_file())
        if disk_bytes > DISK_LIMIT_BYTES:
            raise RuntimeError(f"explicit cache disk limit exceeded: {disk_bytes}")
        _plot(run / "metrics", manifests)
        for relative in spec["frozen_inputs"]:
            after[relative] = sha256(PROJECT_ROOT / relative)
        if after != before:
            raise RuntimeError("frozen export source changed")
        write_json(run / "config/source_integrity_after.json", after)
        overall = PASS
        write_json(run / "metrics/summary.json", {
            "schema_version": "gse_explicit_composer_cache_export_runner_v1",
            "overall_status": overall, "scientific_pass": True,
            "seeds": 3, "worlds_per_seed": 80, "observations_per_seed": 188126,
            "model_forward_observations": 564378,
            "development_parity_worlds": 60, "maximum_development_parity_error": 0.0,
            "fields": manifests[0]["fields"], "forbidden_fields": manifests[0]["forbidden_fields"],
            "cache_files": 240, "disk_bytes_before_seal": disk_bytes,
            "peak_host_rss_kib": peak_rss_kib,
            "peak_gpu_memory_bytes": max(int(manifest["peak_gpu_memory_bytes"]) for manifest in manifests),
            "subprocesses": subprocesses, "source_evidence": source_evidence,
            "optimizer_steps": 0, "checkpoints_created": 0,
            "c09_worlds_read": 0, "c10_worlds_read": 0, "strict_test_worlds_read": 0,
            "mtare_worlds_read": 0, "graph_replays": 0, "planner_calls": 0,
            "duration_seconds": time.monotonic() - started,
        })
    except Exception as exception:
        error = f"{type(exception).__name__}: {exception}"; overall = FAIL
        (run / "logs/runner_error.log").write_text(traceback.format_exc(), encoding="utf-8")
        write_json(run / "metrics/summary.json", {
            "schema_version": "gse_explicit_composer_cache_export_runner_v1",
            "overall_status": overall, "scientific_pass": False, "error": error,
            "subprocesses": subprocesses, "optimizer_steps": 0, "checkpoints_created": 0,
            "c09_worlds_read": 0, "c10_worlds_read": 0, "strict_test_worlds_read": 0,
            "mtare_worlds_read": 0, "graph_replays": 0,
            "duration_seconds": time.monotonic() - started,
        })
    write_json(run / "RUN_STATE.json", {
        "schema_version": "v3_run_state_v1", "run_id": run_id,
        "state": "COMPLETED" if error is None else "FAILED",
        "overall_status": overall, "error": error, "duration_seconds": time.monotonic() - started,
    })
    evidence = seal(run)
    print(json.dumps({"run_id": run_id, "overall_status": overall, "error": error, "evidence_files": evidence}, indent=2, sort_keys=True))
    return 0 if error is None and overall == PASS else 2


if __name__ == "__main__":
    raise SystemExit(main())
