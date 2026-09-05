#!/usr/bin/env python3
"""Run and seal the approved one-shot C08 hybrid implicit qualification."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, write_json


RUN_ID = "gate4_20260816_cano_c08_route_conditioned_support_corrective_v8r_seed0"
SIDECAR = Path("/tmp/mtare_gate4_meshing_sidecar_v1/bin/python")
EXECUTOR = PROJECT_ROOT / "tools/v3/execute_cano_c08_hybrid_implicit_qualification_v3.py"
EXPECTED_FREEZE_SHA256 = "6447ba5efb58f9458e17aa9cb28db38b5aa4dfef5d59633e1072d1ea119c4035"
RAM_LIMIT_BYTES = 4 * 1024**3
DISK_LIMIT_BYTES = 12 * 1024**3
TIME_LIMIT_SECONDS = 12 * 60 * 60
PASS_STATUS = "PASS_CANO_C08_ROUTE_CONDITIONED_SUPPORT_CORRECTIVE_V8R"
FAIL_STATUS = "FAIL_CANO_C08_ROUTE_CONDITIONED_SUPPORT_CORRECTIVE_V8R"
RUNNER_SCHEMA_VERSION = "cano_c08_route_conditioned_support_corrective_runner_v8r"
EXPECTED_SUMMARY = {
    "worlds": 3, "windows": 25, "arc_endpoint_records": 624, "directed_traversals": 624,
    "frames": 4773, "resolution_frame_audits": 14319, "horizontal_rays": 10309680,
    "collision_vertical_rays": 14319, "support_height_evaluations": 14319,
    "window_frames": 466, "outside_window_frames": 4307, "visual_patch_count": 150,
    "c09_worlds_read": 0, "preview_count": 3,
}
CLAIM_BOUNDARY = "C08 route-conditioned support and layered collision qualification only; causal replay and Gate 4 remain unexecuted."


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def seal(run: Path) -> int:
    destination = run / "artifacts/evidence_sha256.txt"
    files = sorted(path for path in run.rglob("*") if path.is_file() and path != destination)
    destination.write_text("".join(f"{sha(path)}  {path.relative_to(PROJECT_ROOT)}\n" for path in files), encoding="utf-8")
    return len(files)


def freeze_identity() -> tuple[str, str]:
    completed = subprocess.run([str(SIDECAR), "-m", "pip", "freeze"], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True)
    normalized = "\n".join(sorted(line for line in completed.stdout.splitlines() if line.strip())) + "\n"
    return hashlib.sha256(normalized.encode()).hexdigest(), normalized


def child_limits() -> None:
    resource.setrlimit(resource.RLIMIT_AS, (RAM_LIMIT_BYTES, RAM_LIMIT_BYTES))


def terminate_process_group(process: subprocess.Popen, grace_seconds: float = 5.0) -> None:
    """Terminate the complete executor process group without leaving an orphan."""
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    spec = load_json(args.spec.resolve())
    run = args.run_dir.resolve()
    if run.name != RUN_ID or load_json(run / "RUN_STATE.json").get("state") != "CREATED_NOT_EXECUTED":
        raise RuntimeError("run identity/state mismatch")
    if spec.get("gate") != 4 or spec.get("operation") != "topology_replay" or spec.get("continuous_layered_qualification_only") is not True:
        raise RuntimeError("scope mismatch")
    card = load_json(PROJECT_ROOT / spec["data_card"])
    if card.get("approval", {}).get("status") != "APPROVED" or card["approval"].get("authorized_operations") != ["topology_replay"]:
        raise RuntimeError("data card is not approved for this operation")
    observed_inputs = {}
    for relative, expected in spec["frozen_inputs"].items():
        observed_inputs[relative] = sha(PROJECT_ROOT / relative)
        if observed_inputs[relative] != expected:
            raise RuntimeError(f"frozen input drift: {relative}")
    observed_tools = {}
    for name, item in spec["frozen_tools"].items():
        observed_tools[name] = sha(PROJECT_ROOT / item["path"])
        if observed_tools[name] != item["sha256"]:
            raise RuntimeError(f"frozen tool drift: {name}")
    freeze_sha, freeze_text = freeze_identity()
    if freeze_sha != EXPECTED_FREEZE_SHA256:
        raise RuntimeError(f"sidecar freeze drift: {freeze_sha}")
    check = subprocess.run([str(SIDECAR), "-m", "pip", "check"], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    if check.returncode != 0:
        raise RuntimeError(f"sidecar pip check failed: {check.stdout}")
    identity = json.loads(subprocess.check_output([
        str(SIDECAR), "-c",
        "import json,sys,numpy,scipy,skimage,open3d,matplotlib;print(json.dumps({'python':sys.version.split()[0],'numpy':numpy.__version__,'scipy':scipy.__version__,'skimage':skimage.__version__,'open3d':open3d.__version__,'matplotlib':matplotlib.__version__}))",
    ], text=True))
    write_json(run / "config/input_hashes.json", observed_inputs)
    write_json(run / "config/tool_hashes.json", observed_tools)
    write_json(run / "config/executor_environment_identity.json", {
        **identity, "sidecar_freeze_sha256": freeze_sha, "pip_check": check.stdout.strip(),
        "platform": platform.platform(), "ram_limit_bytes": RAM_LIMIT_BYTES,
    })
    (run / "config/sidecar_pip_freeze.txt").write_text(freeze_text, encoding="utf-8")
    write_json(run / "RUN_STATE.json", {"schema_version": "v3_run_state_v1", "run_id": RUN_ID, "state": "RUNNING"})

    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join((str(PROJECT_ROOT / "src"), str(PROJECT_ROOT / "tools/v3")))
    environment.update({"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"})
    argv = [str(SIDECAR), str(EXECUTOR), "--run-dir", str(run)]
    log_path = run / "logs/01_hybrid_implicit_qualification.log"
    started = time.monotonic()
    return_code = 125
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"argv={json.dumps(argv)}\nstarted_at_utc={datetime.now(timezone.utc).isoformat()}\n")
        log.flush()
        process = subprocess.Popen(
            argv, cwd=PROJECT_ROOT, env=environment, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1,
            preexec_fn=child_limits, start_new_session=True,
        )
        assert process.stdout is not None
        previous_sigterm = signal.getsignal(signal.SIGTERM)

        def handle_sigterm(_signum, _frame):
            terminate_process_group(process)
            raise TimeoutError("runner received SIGTERM and terminated the executor process group")

        signal.signal(signal.SIGTERM, handle_sigterm)
        try:
            for line in process.stdout:
                log.write(line); log.flush(); print(line, end="", flush=True)
                if time.monotonic() - started > TIME_LIMIT_SECONDS:
                    terminate_process_group(process)
                    raise TimeoutError(f"qualification exceeded {TIME_LIMIT_SECONDS / 3600:g} hour monotonic-active cap")
            return_code = process.wait()
        except BaseException as exc:
            terminate_process_group(process)
            log.write(f"runner_exception={type(exc).__name__}: {exc}\n")
            print(f"runner_exception={type(exc).__name__}: {exc}", flush=True)
        finally:
            signal.signal(signal.SIGTERM, previous_sigterm)
        duration = time.monotonic() - started
        log.write(f"finished_at_utc={datetime.now(timezone.utc).isoformat()}\nduration_seconds={duration:.6f}\nexit_code={return_code}\n")

    summary = load_json(run / "metrics/summary.json") if (run / "metrics/summary.json").is_file() else {}
    size = sum(path.stat().st_size for path in run.rglob("*") if path.is_file())
    passed = bool(
        return_code == 0
        and summary.get("overall_status") == PASS_STATUS
        and all(summary.get(key) == value for key, value in EXPECTED_SUMMARY.items() if key != "preview_count")
        and summary.get("outside_window_pose_exact") is True
        and summary.get("all_frames_passed") is True
        and summary.get("inference_frames") == summary.get("graph_updates") == summary.get("training_samples_consumed") == 0
        and summary.get("c10_worlds_read") == summary.get("mtare_worlds_read") == 0
        and size <= DISK_LIMIT_BYTES and len(list((run / "previews").glob("*.png"))) == EXPECTED_SUMMARY["preview_count"]
    )
    overall = PASS_STATUS if passed else FAIL_STATUS
    write_json(run / "metrics/runner_summary.json", {
        "schema_version": RUNNER_SCHEMA_VERSION, "overall_status": overall,
        "executor_exit_code": return_code, "duration_seconds": duration, "result_bytes_before_seal": size,
        "disk_limit_bytes": DISK_LIMIT_BYTES, "ram_limit_bytes": RAM_LIMIT_BYTES,
        "worlds": summary.get("worlds", 0), "windows": summary.get("windows", 0), "frames": summary.get("frames", 0),
        "claim_boundary": CLAIM_BOUNDARY,
    })
    write_json(run / "RUN_STATE.json", {"schema_version": "v3_run_state_v1", "run_id": RUN_ID, "state": "COMPLETED" if passed else "FAILED", "overall_status": overall})
    count = seal(run)
    print(json.dumps({"overall_status": overall, "sealed_files": count}, indent=2), flush=True)
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
