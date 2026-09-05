"""CPU contracts for the three-seed AEE encoder-adaptation runner."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
TOOLS = ROOT / "tools/v3"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
spec = importlib.util.spec_from_file_location(
    "run_aee_encoder_domain_adaptation_v2",
    TOOLS / "run_aee_encoder_domain_adaptation_v2.py",
)
assert spec is not None and spec.loader is not None
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def _passing(seed: int) -> dict:
    count = {
        "macro_f1_count_1_to_4": 0.75,
        "gate_branch_counts": [1, 2, 3, 4],
        "rare_branch_counts_diagnostic_only": {
            "5": {"support": 10, "f1": 0.1},
            "6": {"support": 2, "f1": 0.0},
        },
    }
    return {
        "seed": seed,
        "status": "PASS_AEE_ENCODER_DOMAIN_ADAPTATION_SEED_V2",
        "epochs_completed": 10,
        "independent_train_samples_per_epoch": {"cano": 3000, "aee": 3000},
        "augmented_cano_views_per_epoch": 3000,
        "aee_validation_frames": 3000,
        "cano_validation_frames": 12500,
        "strict_test_frames_read": 0,
        "cano_c09_validation_frames_read": 12500,
        "c10_frames_read": 0,
        "later_sealed_world_frames_read": 0,
        "gate": {
            "aee_direction_vs_b0": True,
            "aee_empty_rate": True,
            "aee_count_1_to_4_macro_f1": True,
            "aee_role_macro_f1": True,
            "cano_direction_retention": True,
            "all_model_tensors_finite": True,
            "encoder_changed": True,
            "embedding_changed": True,
            "semantic_heads_changed": True,
        },
        "adapted_aee_validation": {
            "direction": {"f1": 0.8, "empty_rate": 0.01},
            "count": count,
            "role": {"macro_f1_present": 0.76},
        },
        "adapted_cano_validation": {"direction": {"f1": 0.79}},
        "source_cano_validation": {"direction": {"f1": 0.80}},
    }


def test_child_argv_uses_new_trainer_and_frozen_hyperparameters(tmp_path: Path) -> None:
    argv = runner.child_argv(
        {"cano_dataset_run": "cano", "source_sensor_run": "sensor", "source_teacher_run": "teacher"},
        tmp_path,
        {"seed": 1, "path": "checkpoint.pt", "sha256": "hash"},
        Path("/verified/venv/bin/python"),
    )
    assert argv[0] == "/verified/venv/bin/python"
    assert argv[1] == str(runner.TRAINER)
    for flag, value in {
        "--seed": "1", "--epochs": "10", "--batch-size": "128",
        "--learning-rate": "0.0001", "--weight-decay": "0.0001", "--workers": "0",
    }.items():
        assert argv[argv.index(flag) + 1] == value


def test_runner_requires_all_three_seed_gates_and_count_contract() -> None:
    summaries = [_passing(seed) for seed in (0, 1, 2)]
    result = runner.aggregate(summaries)
    assert result["all_seed_gates_passed"] is True
    assert len(result["aee_count_5_to_6_diagnostic_by_seed"]) == 3
    bad = _passing(0)
    bad["gate"]["encoder_changed"] = False
    with pytest.raises(RuntimeError, match="representation gate"):
        runner.validate_seed_summary(bad, 0)
    bad = _passing(0)
    bad["adapted_aee_validation"]["count"]["gate_branch_counts"] = [1, 2, 3, 4, 5, 6]
    with pytest.raises(RuntimeError, match="common-class"):
        runner.validate_seed_summary(bad, 0)


def test_runner_rejects_any_forbidden_world_read() -> None:
    for key in ("strict_test_frames_read", "c10_frames_read", "later_sealed_world_frames_read"):
        bad = _passing(0)
        bad[key] = 1
        with pytest.raises(RuntimeError):
            runner.validate_seed_summary(bad, 0)
