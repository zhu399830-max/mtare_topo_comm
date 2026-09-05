"""CPU-only contracts for the AEE head-adaptation outer runner.

These tests deliberately exercise the orchestration layer with synthetic
metadata.  They must never import a CUDA device, open a dataset shard, or
start the single-seed trainer.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
RUNNER_PATH = ROOT / "tools/v3/run_aee_head_adaptation_v1.py"


def _load_runner():
    # The runner is intentionally allowed to be added after this test module;
    # until then the test is a pending contract, not a reason to run training.
    if not RUNNER_PATH.is_file():
        pytest.skip("AEE head-adaptation outer runner is not implemented yet")
    import sys

    tools = str(ROOT / "tools/v3")
    if tools not in sys.path:
        sys.path.insert(0, tools)
    spec = importlib.util.spec_from_file_location("run_aee_head_adaptation_v1", RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _call_child_argv(runner, tmp_path: Path, seed: int) -> list[str]:
    spec = {
        "cano_dataset_run": "cano",
        "source_sensor_run": "sensor",
        "source_teacher_run": "teacher",
    }
    checkpoint = {"seed": seed, "path": f"checkpoint{seed}.pt", "sha256": f"hash{seed}"}
    return list(runner.child_argv(spec, tmp_path, checkpoint))


def test_child_argv_freezes_three_seed_and_all_hyperparameters(tmp_path: Path) -> None:
    runner = _load_runner()
    expected = {
        "--epochs": "10",
        "--batch-size": "128",
        "--learning-rate": "0.0001",
        "--weight-decay": "0.0001",
        "--workers": "0",
    }
    for seed in (0, 1, 2):
        argv = _call_child_argv(runner, tmp_path, seed)
        assert "--seed" in argv
        assert argv[argv.index("--seed") + 1] == str(seed)
        for flag, value in expected.items():
            assert flag in argv, flag
            assert argv[argv.index(flag) + 1] == value
        assert argv.count("--seed") == 1
        # The outer runner may not silently add an alternate model, split,
        # optimizer, or checkpoint-selection setting.
        assert "--mode" not in argv or argv[argv.index("--mode") + 1] == "M1D"


def test_child_argv_uses_the_explicit_v1r2_python(tmp_path: Path) -> None:
    runner = _load_runner()
    python = tmp_path / "verified-sidecar/bin/python"
    argv = runner.child_argv(
        {
            "cano_dataset_run": "cano",
            "source_sensor_run": "sensor",
            "source_teacher_run": "teacher",
        },
        tmp_path,
        {"seed": 0, "path": "checkpoint0.pt", "sha256": "hash0"},
        python,
    )
    assert argv[0] == str(python)
    assert str(runner.PYTHON) not in argv


def _passing(seed: int) -> dict:
    return {
        "seed": seed,
        "status": "PASS_AEE_HEAD_ADAPTATION_SEED_V1",
        "epochs_completed": 10,
        "train_samples_per_epoch": {"cano": 3000, "aee": 3000},
        "aee_validation_frames": 3000,
        "cano_validation_frames": 12500,
        "strict_test_frames_read": 0,
        "c10_frames_read": 0,
        "cano_c09_validation_frames_read": 12500,
        "later_sealed_world_frames_read": 0,
        "gate": {
            "aee_direction_vs_b0": True,
            "aee_empty_rate": True,
            "aee_count_macro_f1": True,
            "aee_role_macro_f1": True,
            "cano_direction_retention": True,
            "frozen_tensor_identity": True,
            "fixed_probe_z_role_identity": True,
            "semantic_heads_changed": True,
        },
        "adapted_aee_validation": {"direction": {"f1": 0.8, "empty_rate": 0.01}, "count": {"macro_f1_present": 0.75}, "role": {"macro_f1_present": 0.76}},
        "adapted_cano_validation": {"direction": {"f1": 0.79}},
        "source_cano_validation": {"direction": {"f1": 0.80}},
    }


def test_aggregate_requires_exactly_three_passing_seed_records() -> None:
    runner = _load_runner()
    passing = [_passing(seed) for seed in (0, 1, 2)]
    assert runner.aggregate(passing)["all_seed_gates_passed"] is True
    with pytest.raises(RuntimeError):
        runner.aggregate(passing[:2])
    failing = [_passing(seed) for seed in (0, 1, 2)]
    failing[2]["status"] = "FAIL_AEE_HEAD_ADAPTATION_SEED_V1"
    with pytest.raises(RuntimeError):
        runner.aggregate(failing)


def test_environment_identity_and_forbidden_reads_are_hard_contracts() -> None:
    runner = _load_runner()
    passing = _passing(0)
    runner.validate_seed_summary(passing, 0)
    for key in ("strict_test_frames_read", "c10_frames_read", "later_sealed_world_frames_read"):
        bad = _passing(0)
        bad[key] = 1
        with pytest.raises(RuntimeError):
            runner.validate_seed_summary(bad, 0)

    class Completed:
        returncode = 0
        stderr = ""
        stdout = json.dumps(runner.EXPECTED_ENVIRONMENT)

    original = runner.subprocess.run
    calls = []

    def fake_run(*args, **kwargs):
        calls.append((args, kwargs))
        return Completed()

    runner.subprocess.run = fake_run
    try:
        explicit_python = Path("/verified/sidecar/bin/python")
        assert runner.probe_environment(explicit_python) == runner.EXPECTED_ENVIRONMENT
        assert calls[-1][0][0][0] == str(explicit_python)
        Completed.stdout = json.dumps({**runner.EXPECTED_ENVIRONMENT, "torch": "2.8.0"})
        with pytest.raises(RuntimeError):
            runner.probe_environment(explicit_python)
    finally:
        runner.subprocess.run = original


def test_live_pip_freeze_hash_is_order_independent_and_content_strict() -> None:
    runner = _load_runner()
    expected = runner.sorted_pip_freeze_sha256("a==1\nb==2\n")
    assert runner.sorted_pip_freeze_sha256("b==2\na==1\n") == expected
    assert runner.sorted_pip_freeze_sha256("a==1\nb==3\n") != expected


def test_pip_check_must_be_exactly_healthy() -> None:
    runner = _load_runner()
    assert (
        runner.validate_pip_check(0, "No broken requirements found.\n", "")
        == "No broken requirements found."
    )
    with pytest.raises(RuntimeError):
        runner.validate_pip_check(1, "", "broken dependency")
    with pytest.raises(RuntimeError):
        runner.validate_pip_check(0, "unexpected success text\n", "")


def test_venv_executable_is_made_absolute_without_dereferencing(tmp_path: Path) -> None:
    runner = _load_runner()
    base = tmp_path / "base-python"
    base.write_text("binary-placeholder", encoding="utf-8")
    venv_python = tmp_path / "venv/bin/python"
    venv_python.parent.mkdir(parents=True)
    venv_python.symlink_to(base)
    preserved = runner.preserve_venv_executable(str(venv_python))
    assert preserved == venv_python.absolute()
    assert preserved != venv_python.resolve()
    with pytest.raises(RuntimeError):
        runner.preserve_venv_executable("relative/venv/bin/python")


def test_failure_seal_helper_is_atomic_and_records_reason(tmp_path: Path) -> None:
    runner = _load_runner()
    run = tmp_path / "run"
    run.mkdir()
    (run / "metrics").mkdir()
    (run / "RUN_STATE.json").write_text(json.dumps({"state": "RUNNING"}), encoding="utf-8")
    runner.seal = lambda path: 2
    runner.fail(run, 0.0, 1, "synthetic child failure")
    state = json.loads((run / "RUN_STATE.json").read_text(encoding="utf-8"))
    summary = json.loads((run / "metrics/summary.json").read_text(encoding="utf-8"))
    assert state["state"] == "FAILED"
    assert summary["failure_reason"] == "synthetic child failure"
    assert summary["completed_seeds"] == 1
