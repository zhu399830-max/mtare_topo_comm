#!/home/zeng-workstation/anaconda3/bin/python
"""Prove the V1R3 environment/argv entry contract without reading model data."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import run_aee_head_adaptation_v1 as runner
from _bootstrap import PROJECT_ROOT
from mtare_topo.data.aee_domain_adaptation import sha256
from mtare_topo.governance import load_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    args = parser.parse_args()
    spec = load_json(args.spec.resolve())
    python = runner.preserve_venv_executable(spec["python_executable"])
    identity = runner.probe_environment(python)

    freeze = spec["sidecar_freeze"]
    freeze_path = PROJECT_ROOT / freeze["path"]
    pip_freeze_path = PROJECT_ROOT / freeze["pip_freeze_path"]
    if sha256(freeze_path) != freeze["sha256"]:
        raise RuntimeError("sidecar freeze file drift")
    if sha256(pip_freeze_path) != freeze["pip_freeze_sha256"]:
        raise RuntimeError("pip-freeze evidence drift")

    live_freeze = subprocess.run(
        [str(python), "-m", "pip", "freeze"],
        text=True,
        capture_output=True,
        check=True,
    )
    live_freeze_sha256 = runner.sorted_pip_freeze_sha256(live_freeze.stdout)
    if live_freeze_sha256 != freeze["pip_freeze_sha256"]:
        raise RuntimeError("live pip-freeze identity drift")
    live_check = subprocess.run(
        [str(python), "-m", "pip", "check"],
        text=True,
        capture_output=True,
        check=False,
    )
    pip_check = runner.validate_pip_check(
        live_check.returncode, live_check.stdout, live_check.stderr
    )

    checkpoint = spec["source_checkpoints"][0]
    argv = runner.child_argv(spec, Path("/nonmaterial/preentry"), checkpoint, python)
    if argv[0] != str(python):
        raise RuntimeError("child trainer does not preserve the verified venv path")
    if Path(argv[0]).resolve() == Path(argv[0]):
        raise RuntimeError("proof requires a venv symlink distinct from its base target")

    proof = {
        "status": "PASS_AEE_HEAD_ADAPTATION_PREENTRY_PROOF_V1R3",
        "python_executable": str(python),
        "resolved_base_executable": str(Path(python).resolve()),
        "venv_path_preserved": True,
        "environment": identity,
        "live_pip_freeze_sha256": live_freeze_sha256,
        "pip_check": pip_check,
        "child_argv_python": argv[0],
        "datasets_opened": 0,
        "checkpoints_opened": 0,
        "model_inference_frames": 0,
        "optimizer_steps": 0,
        "c09_frames_read": 0,
        "c10_frames_read": 0,
        "files_written": 0,
    }
    print(json.dumps(proof, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
