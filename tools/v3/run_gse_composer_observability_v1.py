#!/usr/bin/env python3
"""Execute and seal one immutable dual-Composer observability audit."""

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
from mtare_topo.governance import load_json, write_json


RUN_ID = "gate3_20260829_gse_composer_observability_v1_seed0"
PASS = "PASS_GSE_COMPOSER_OBSERVABILITY_V1"
FAIL = "FAIL_GSE_COMPOSER_OBSERVABILITY_V1"
CARD_STATUS = "APPROVED_FOR_ONE_IMMUTABLE_GSE_COMPOSER_OBSERVABILITY_V1"
PYTHON = Path(
    "/home/zeng-workstation/.local/share/mtare_topo_comm/"
    "envs/phase3_torch290_cu129_zarr2187_v1/bin/python"
)
TEACHER = PROJECT_ROOT / (
    "results/gate2_representation/"
    "gate2_20260827_gse_corrected_causal_teacher_manifest_v1r_seed0"
)
BASELINE = PROJECT_ROOT / (
    "results/gate3_semantics/"
    "gate3_20260829_gse_sparse_circular_relation_transport_three_seed_training_v2r5_seed0"
)
EVALUATOR = PROJECT_ROOT / "tools/v3/evaluate_gse_composer_observability_v1.py"
TIME_LIMIT_SECONDS = 3600
RAM_LIMIT_KIB = 4 * 1024**2
DISK_LIMIT_BYTES = 512 * 1024**2


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _peak_rss(path: Path) -> int | None:
    match = re.search(
        r"Maximum resident set size \(kbytes\):\s*(\d+)",
        path.read_text(encoding="utf-8"),
    )
    return int(match.group(1)) if match else None


def _seal(run_dir: Path) -> int:
    seal = run_dir / "artifacts/evidence_sha256.txt"
    files = sorted(path for path in run_dir.rglob("*") if path.is_file() and path != seal)
    with seal.open("w", encoding="utf-8") as stream:
        for path in files:
            stream.write(f"{_sha256(path)}  {path.relative_to(PROJECT_ROOT)}\n")
    return len(files)


def _verify_source_seal(run_dir: Path) -> int:
    seal = run_dir / "artifacts/evidence_sha256.txt"
    if not seal.is_file():
        raise RuntimeError(f"source run has no seal: {run_dir.name}")
    covered: set[Path] = set()
    for line in seal.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        path = (PROJECT_ROOT / relative).resolve()
        path.relative_to(run_dir.resolve())
        if path in covered or not path.is_file() or _sha256(path) != expected:
            raise RuntimeError(f"source seal drift: {relative}")
        covered.add(path)
    actual = {
        path.resolve()
        for path in run_dir.rglob("*")
        if path.is_file() and path.resolve() != seal.resolve()
    }
    if covered != actual:
        raise RuntimeError(f"source seal coverage drift: {run_dir.name}")
    return len(covered)


def _verify_baseline() -> dict:
    state = load_json(BASELINE / "RUN_STATE.json")
    summary = load_json(BASELINE / "metrics/summary.json")
    if (
        state.get("run_id") != BASELINE.name
        or state.get("state") != "COMPLETED"
        or state.get("overall_status") not in {
            "PASS_GSE_SPARSE_CIRCULAR_RELATION_TRANSPORT_THREE_SEED_TRAINING_V2",
            "FAIL_GSE_SPARSE_CIRCULAR_RELATION_TRANSPORT_THREE_SEED_TRAINING_V2",
        }
        or state.get("error") is not None
        or summary.get("optimizer_steps") != 36690
        or summary.get("c09_worlds_read") != 0
        or summary.get("c10_worlds_read") != 0
        or summary.get("mtare_worlds_read") != 0
    ):
        raise RuntimeError("V2R5 is not a complete leakage-free three-seed component baseline")
    for seed in range(3):
        root = BASELINE / f"artifacts/models/seed{seed}"
        seed_summary = load_json(root / "summary.json")
        if (
            seed_summary.get("optimizer_steps") != 12230
            or seed_summary.get("development_output_observations") != 45942
            or not (root / "best.pt").is_file()
            or len(list((root / "development_predictions").glob("*.npz"))) != 20
        ):
            raise RuntimeError(f"V2R5 seed{seed} evidence incomplete")
    return {
        "run": str(BASELINE.relative_to(PROJECT_ROOT)),
        "overall_status": state["overall_status"],
        "seal_entries": _verify_source_seal(BASELINE),
        "seal_sha256": _sha256(BASELINE / "artifacts/evidence_sha256.txt"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    spec = load_json(args.spec.resolve())
    if run_dir.name != RUN_ID or load_json(run_dir / "RUN_STATE.json").get("state") != "CREATED_NOT_EXECUTED":
        raise RuntimeError("Composer observability audit may execute only once")

    started = time.monotonic()
    overall = FAIL
    error = None
    returncode = None
    peak_rss = None
    result: dict = {}
    before: dict[str, str] = {}
    after: dict[str, str] = {}
    baseline: dict = {}
    try:
        if spec.get("gate") != 3 or spec.get("operation") != "audit" or spec.get("seed") != 0:
            raise RuntimeError("Composer observability formal scope mismatch")
        if spec.get("user_authorization", {}).get("status") != "APPROVED":
            raise RuntimeError("standing authorization is not bound to the audit")
        card = load_json(PROJECT_ROOT / spec["data_card"])
        if card.get("status") != CARD_STATUS or card.get("approval", {}).get("status") != "APPROVED":
            raise RuntimeError("Composer observability Data Card mismatch")
        for name, record in spec["frozen_tools"].items():
            if _sha256(PROJECT_ROOT / record["path"]) != record["sha256"]:
                raise RuntimeError(f"frozen audit tool drift: {name}")
        for relative, expected in spec["frozen_inputs"].items():
            observed = _sha256(PROJECT_ROOT / relative)
            if observed != expected:
                raise RuntimeError(f"frozen audit input drift: {relative}")
            before[relative] = observed
        baseline = _verify_baseline()
        versions = json.loads(
            subprocess.check_output(
                [
                    str(PYTHON),
                    "-c",
                    "import json,matplotlib,numpy,sklearn,sys;print(json.dumps({'python':sys.version.split()[0],'numpy':numpy.__version__,'matplotlib':matplotlib.__version__,'sklearn':sklearn.__version__},sort_keys=True))",
                ],
                text=True,
            )
        )
        expected = {
            "python": "3.13.5",
            "numpy": "2.1.3",
            "matplotlib": "3.10.0",
            "sklearn": "1.6.1",
        }
        if versions != expected:
            raise RuntimeError(f"Composer observability environment drift: {versions}")
        write_json(run_dir / "config/environment.json", {"executable": str(PYTHON), "versions": versions, "cpu_only": True})
        write_json(run_dir / "config/source_integrity_before.json", before)
        write_json(run_dir / "config/baseline_integrity.json", baseline)
        write_json(run_dir / "RUN_STATE.json", {"schema_version": "v3_run_state_v1", "run_id": RUN_ID, "state": "RUNNING"})
        env = os.environ.copy()
        env.update(
            {
                "PYTHONPATH": str(PROJECT_ROOT / "src") + os.pathsep + str(PROJECT_ROOT / "tools/v3"),
                "OMP_NUM_THREADS": "4",
                "MKL_NUM_THREADS": "4",
            }
        )
        test_log = run_dir / "logs/00_unit_tests.log"
        with test_log.open("w", encoding="utf-8") as stream:
            tests = subprocess.run(
                [
                    str(PYTHON),
                    "-m",
                    "pytest",
                    "-q",
                    "tests/v3/unit/test_gse_composer_observability.py",
                    "tests/v3/unit/test_evaluate_gse_composer_observability_v1.py",
                ],
                cwd=PROJECT_ROOT,
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=300,
                check=False,
            )
        if tests.returncode != 0:
            raise RuntimeError("Composer observability unit tests failed")
        output = run_dir / "metrics/observability"
        command = [
            "/usr/bin/time",
            "-v",
            str(PYTHON),
            str(EVALUATOR),
            "--teacher",
            str(TEACHER / "artifacts/teacher_observations.jsonl"),
        ]
        for seed in range(3):
            command.extend(("--seed-dir", str(BASELINE / f"artifacts/models/seed{seed}")))
        command.extend(("--output-dir", str(output)))
        (run_dir / "config/command.txt").write_text(" ".join(command) + "\n", encoding="utf-8")
        log = run_dir / "logs/01_observability.log"
        with log.open("w", encoding="utf-8") as stream:
            completed = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=TIME_LIMIT_SECONDS,
                check=False,
            )
        returncode = int(completed.returncode)
        peak_rss = _peak_rss(log)
        result = load_json(output / "summary.json") if (output / "summary.json").is_file() else {}
        after = {relative: _sha256(PROJECT_ROOT / relative) for relative in before}
        output_bytes = sum(path.stat().st_size for path in run_dir.rglob("*") if path.is_file())
        required = [
            output / "summary.json",
            output / "event_observability.csv",
            *[output / f"gse_composer_observability.{suffix}" for suffix in ("png", "pdf", "svg")],
        ]
        if (
            before != after
            or returncode not in (0, 2)
            or result.get("overall_status") not in (PASS, FAIL)
            or (returncode == 0) != (result.get("overall_status") == PASS)
            or result.get("rows") != 45942
            or result.get("split_rows") != {"C07": 21548, "C08": 24394}
            or result.get("diagnostic_linear_probe_fits") != 12
            or any(result.get(name) != 0 for name in ("main_model_optimizer_steps", "checkpoint_updates", "c09_worlds_read", "c10_worlds_read", "mtare_worlds_read"))
            or peak_rss is None
            or peak_rss > RAM_LIMIT_KIB
            or output_bytes > DISK_LIMIT_BYTES
            or not all(path.is_file() and path.stat().st_size for path in required)
        ):
            raise RuntimeError("Composer observability execution/evidence contract failed")
        overall = str(result["overall_status"])
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        (run_dir / "logs/failure_traceback.log").write_text(traceback.format_exc(), encoding="utf-8")

    summary = {
        "schema_version": "gse_composer_observability_outer_v1",
        "overall_status": overall,
        "scientific_pass": overall == PASS and error is None,
        "error": error,
        "duration_seconds": time.monotonic() - started,
        "returncode": returncode,
        "peak_host_rss_kib": peak_rss,
        "source_unchanged": bool(before and before == after),
        "baseline": baseline,
        "observability": result,
        "main_model_optimizer_steps": 0,
        "checkpoint_updates": 0,
        "c09_worlds_read": 0,
        "c10_worlds_read": 0,
        "mtare_worlds_read": 0,
        "graph_replays": 0,
    }
    write_json(run_dir / "metrics/summary.json", summary)
    write_json(
        run_dir / "RUN_STATE.json",
        {
            "schema_version": "v3_run_state_v1",
            "run_id": RUN_ID,
            "state": "COMPLETED" if error is None else "FAILED",
            "overall_status": overall,
            "error": error,
        },
    )
    entries = _seal(run_dir)
    print(json.dumps({"overall_status": overall, "error": error, "seal_entries": entries}, indent=2))
    return 0 if overall == PASS and error is None else 2


if __name__ == "__main__":
    raise SystemExit(main())
