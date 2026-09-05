#!/usr/bin/env python3
"""Run a frozen, stop-on-first-pass Cano dependency compatibility matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import selectors
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, write_json


EXPECTED_REMOTE = "https://github.com/LorenzoCanoAn/procedural-subt-gen"
EXPECTED_COMMIT = "b6c77621187404b4dfab1249c7a1b40f63ad9ab3"
EXPECTED_SNIPPET_SHA256 = (
    "869bc12e0344da8b4bccaba3463f507ab9ec9531dbcbe9d6db6476fb3ff05f12"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_output(source: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(source), *args], text=True, stderr=subprocess.STDOUT
    ).strip()


def _run_logged(
    argv: list[str],
    *,
    cwd: Path,
    log_path: Path,
    env: dict[str, str] | None = None,
    timeout_s: int | None = None,
) -> tuple[int, float]:
    started = time.monotonic()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("argv=" + json.dumps(argv, ensure_ascii=False) + "\n")
        log.write(f"cwd={cwd}\n")
        log.write(f"started_at_utc={datetime.now(timezone.utc).isoformat()}\n")
        log.flush()
        process = subprocess.Popen(
            argv,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ)
        return_code: int | None = None
        while return_code is None:
            if timeout_s is not None and time.monotonic() - started > timeout_s:
                process.kill()
                process.wait()
                log.write(f"TIMEOUT_AFTER_SECONDS={timeout_s}\n")
                return_code = 124
                break
            for key, _ in selector.select(timeout=0.5):
                line = key.fileobj.readline()
                if line:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                    log.write(line)
                    log.flush()
            polled = process.poll()
            if polled is not None:
                remainder = process.stdout.read()
                if remainder:
                    sys.stdout.write(remainder)
                    sys.stdout.flush()
                    log.write(remainder)
                return_code = polled
        selector.close()
        duration = time.monotonic() - started
        log.write(f"finished_at_utc={datetime.now(timezone.utc).isoformat()}\n")
        log.write(f"duration_seconds={duration:.6f}\n")
        log.write(f"exit_code={return_code}\n")
        return return_code, duration


def _validate_spec(spec: dict[str, Any]) -> list[dict[str, Any]]:
    matrix = spec.get("compatibility_matrix")
    if not isinstance(matrix, list) or not matrix:
        raise ValueError("compatibility_matrix must be a non-empty list")
    if len(matrix) > 2:
        raise ValueError("at most two new compatibility environments are allowed")
    expected_ids = ["E1", "E2"]
    if [item.get("id") for item in matrix] != expected_ids[: len(matrix)]:
        raise ValueError("matrix IDs/order must be E1 then E2")
    for item in matrix:
        for key in (
            "id",
            "python_provider",
            "python_version",
            "numpy",
            "scipy",
            "opencv_python_headless",
            "environment_prefix",
        ):
            if not isinstance(item.get(key), str) or not item[key]:
                raise ValueError(f"matrix {item.get('id')} missing non-empty {key}")
        prefix = Path(item["environment_prefix"])
        if not prefix.is_absolute() or prefix.parent != Path("/tmp"):
            raise ValueError("environment_prefix must be an explicit child of /tmp")
        if item["python_provider"] not in {"system_venv", "conda"}:
            raise ValueError("python_provider must be system_venv or conda")
    return matrix


def _prepare_environment(
    item: dict[str, Any], source: Path, run_dir: Path
) -> tuple[Path | None, dict[str, Any]]:
    env_id = item["id"]
    prefix = Path(item["environment_prefix"])
    result: dict[str, Any] = {
        "id": env_id,
        "status": "PREPARING",
        "environment_prefix": str(prefix),
    }
    if prefix.exists():
        result.update(status="BLOCKED", error="environment prefix already exists")
        return None, result

    if item["python_provider"] == "system_venv":
        create_argv = [sys.executable, "-m", "venv", str(prefix)]
    else:
        conda = Path("/home/zeng-workstation/anaconda3/bin/conda")
        if not conda.is_file():
            result.update(status="BLOCKED", error=f"conda not found: {conda}")
            return None, result
        create_argv = [
            str(conda),
            "create",
            "--yes",
            "--prefix",
            str(prefix),
            f"python={item['python_version']}",
            "pip",
        ]

    rc, duration = _run_logged(
        create_argv,
        cwd=PROJECT_ROOT,
        log_path=run_dir / "logs" / f"{env_id}_01_create_environment.log",
        timeout_s=900,
    )
    result["create_duration_seconds"] = duration
    if rc != 0:
        result.update(status="BLOCKED", error=f"environment creation exited {rc}")
        return None, result

    python = prefix / "bin" / "python"
    constraints = run_dir / "config" / f"constraints_{env_id}.txt"
    constraints.write_text(
        f"numpy=={item['numpy']}\n"
        f"scipy=={item['scipy']}\n"
        f"opencv-python-headless=={item['opencv_python_headless']}\n",
        encoding="utf-8",
    )
    install_argv = [
        str(python),
        "-m",
        "pip",
        "install",
        "--constraint",
        str(constraints),
        "-r",
        str(source / "requirements.txt"),
        str(source),
        f"opencv-python-headless=={item['opencv_python_headless']}",
    ]
    rc, duration = _run_logged(
        install_argv,
        cwd=PROJECT_ROOT,
        log_path=run_dir / "logs" / f"{env_id}_02_install.log",
        timeout_s=3600,
    )
    result["install_duration_seconds"] = duration
    if rc != 0:
        result.update(status="BLOCKED", error=f"dependency install exited {rc}")
        return None, result

    freeze_path = run_dir / "config" / f"pip_freeze_{env_id}.txt"
    freeze = subprocess.check_output(
        [str(python), "-m", "pip", "freeze"], text=True, stderr=subprocess.STDOUT
    )
    freeze_path.write_text(freeze, encoding="utf-8")
    check_rc, check_duration = _run_logged(
        [str(python), "-m", "pip", "check"],
        cwd=PROJECT_ROOT,
        log_path=run_dir / "logs" / f"{env_id}_03_pip_check.log",
        timeout_s=120,
    )
    result["pip_check_duration_seconds"] = check_duration
    if check_rc != 0:
        result.update(status="BLOCKED", error=f"pip check exited {check_rc}")
        return None, result

    version_code = (
        "import cv2,numpy,scipy,sys;"
        "print(sys.version);"
        "print('numpy='+numpy.__version__);"
        "print('scipy='+scipy.__version__);"
        "print('opencv='+cv2.__version__)"
    )
    versions = subprocess.check_output(
        [str(python), "-c", version_code], text=True, stderr=subprocess.STDOUT
    )
    (run_dir / "config" / f"runtime_versions_{env_id}.txt").write_text(
        versions, encoding="utf-8"
    )
    result["status"] = "READY"
    return python, result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    spec_path = args.spec if args.spec.is_absolute() else PROJECT_ROOT / args.spec
    run_dir = args.run_dir if args.run_dir.is_absolute() else PROJECT_ROOT / args.run_dir
    spec = load_json(spec_path)
    matrix = _validate_spec(spec)

    source = PROJECT_ROOT / spec["source"]["checkout"]
    snippet = source / "scripts" / "snippet_2.py"
    if _git_output(source, "rev-parse", "HEAD") != EXPECTED_COMMIT:
        raise RuntimeError("source commit mismatch")
    remotes = _git_output(source, "remote", "get-url", "origin")
    if remotes.rstrip("/") != EXPECTED_REMOTE.rstrip("/"):
        raise RuntimeError("source remote mismatch")
    if _git_output(source, "status", "--porcelain=v1"):
        raise RuntimeError("third-party source is not clean before matrix")
    if _sha256(snippet) != EXPECTED_SNIPPET_SHA256:
        raise RuntimeError("snippet_2.py hash mismatch")

    if args.validate_only:
        print(json.dumps({"status": "VALID", "matrix": matrix}, indent=2))
        return 0
    if not run_dir.is_dir():
        raise FileNotFoundError(f"run directory does not exist: {run_dir}")

    write_json(
        run_dir / "RUN_STATE.json",
        {
            "schema_version": "v3_run_state_v1",
            "run_id": run_dir.name,
            "state": "RUNNING",
            "note": "Frozen compatibility matrix is running; stop on first executable environment.",
        },
    )
    previous_summary = load_json(
        PROJECT_ROOT
        / "results/gate0_baseline/gate0_20260810_cano_source_executability_smoke_v1_seed0/metrics/summary.json"
    )
    results: list[dict[str, Any]] = [
        {
            "id": "E0",
            "status": "FAIL_REFERENCE_NOT_RERUN",
            "numpy": "2.5.2",
            "scipy": "1.18.0",
            "exit_code": previous_summary["execution"]["python_exit_code"],
            "failure": previous_summary["execution"]["failure_message"],
            "evidence": "results/gate0_baseline/gate0_20260810_cano_source_executability_smoke_v1_seed0",
        }
    ]
    selected_environment: str | None = None

    for item in matrix:
        env_id = item["id"]
        python, result = _prepare_environment(item, source, run_dir)
        result.update(
            python_version=item["python_version"],
            numpy=item["numpy"],
            scipy=item["scipy"],
            opencv_python_headless=item["opencv_python_headless"],
        )
        if python is None:
            results.append(result)
            continue

        env = dict(os.environ)
        env.update(PYTHONHASHSEED="0", PYVISTA_OFF_SCREEN="true")
        log_path = run_dir / "logs" / f"{env_id}_04_snippet_2_original.log"
        rc, duration = _run_logged(
            [str(python), "scripts/snippet_2.py"],
            cwd=source,
            log_path=log_path,
            env=env,
            timeout_s=300,
        )
        log_text = log_path.read_text(encoding="utf-8")
        grown = [int(value) for value in re.findall(r"Generating grown tunnel (\d+)", log_text)]
        connectors = [
            int(value)
            for value in re.findall(r"Generating connector tunnel (\d+)", log_text)
        ]
        source_clean = not _git_output(source, "status", "--porcelain=v1")
        snippet_unchanged = _sha256(snippet) == EXPECTED_SNIPPET_SHA256
        executable = (
            rc == 0
            and max(grown, default=0) == 10
            and max(connectors, default=0) == 5
            and source_clean
            and snippet_unchanged
        )
        result.update(
            status="PASS_EXECUTABLE" if executable else "FAIL_EXECUTION",
            exit_code=rc,
            duration_seconds=duration,
            max_grown_progress=max(grown, default=0),
            max_connector_progress=max(connectors, default=0),
            source_clean_after=source_clean,
            snippet_hash_unchanged=snippet_unchanged,
        )
        results.append(result)
        if not source_clean or not snippet_unchanged:
            break
        if executable:
            selected_environment = env_id
            break

    final_clean = not _git_output(source, "status", "--porcelain=v1")
    final_hash = _sha256(snippet)
    overall = "PARTIAL_EXECUTABILITY_PASS" if selected_environment else "FAIL_BLOCKED"
    summary = {
        "schema_version": "cano_dependency_compatibility_matrix_summary_v1",
        "run_id": run_dir.name,
        "overall_status": overall,
        "selected_environment": selected_environment,
        "stop_on_first_pass": True,
        "environments": results,
        "counts": {
            "reference_failures": 1,
            "new_execution_attempts": sum(
                1 for item in results if item["id"] != "E0" and "exit_code" in item
            ),
            "completed_original_script_runs": sum(
                item.get("status") == "PASS_EXECUTABLE" for item in results
            ),
            "exported_worlds": 0,
            "point_clouds": 0,
            "meshes": 0,
            "lidar_samples": 0,
            "labels": 0,
            "training_samples": 0,
        },
        "source": {
            "commit": EXPECTED_COMMIT,
            "remote": EXPECTED_REMOTE,
            "clean_after": final_clean,
            "snippet_sha256_after": final_hash,
        },
        "host": {
            "platform": platform.platform(),
            "orchestrator_python": sys.version,
        },
        "claim_boundary": (
            "A selected environment proves only unmodified snippet_2.py process "
            "executability. It does not prove deterministic replay, topology export, "
            "mesh generation, graph-mesh consistency, license permission, dataset "
            "construction, Isaac, LiDAR, labels, or training."
        ),
    }
    write_json(run_dir / "metrics" / "summary.json", summary)
    write_json(
        run_dir / "RUN_STATE.json",
        {
            "schema_version": "v3_run_state_v1",
            "run_id": run_dir.name,
            "state": overall,
            "note": summary["claim_boundary"],
        },
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if selected_environment else 2


if __name__ == "__main__":
    raise SystemExit(main())
