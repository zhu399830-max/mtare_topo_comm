"""CPU contracts for corrective full-encoder training and aggregation."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
TOOLS = ROOT / "tools/v3"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, TOOLS / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


trainer = _load("corrective_trainer", "train_aee_corrective_full_encoder_v3.py")
runner = _load("corrective_runner", "run_aee_corrective_full_encoder_v3.py")


def _metrics(f1: float, count: float = 0.8, role: float = 0.8, empty: float = 0.0):
    return {
        "direction": {"f1": f1, "empty_rate": empty},
        "count": {"macro_f1_count_1_to_4": count},
        "role": {"macro_f1_present": role},
    }


def test_gate_is_fixed_to_sparse_gain_and_dense_retention() -> None:
    gate = trainer._gate_metrics(
        _metrics(0.79), _metrics(0.76), _metrics(0.80), _metrics(0.70),
        {"f1": 0.72}, True, True, True, True,
    )
    assert gate["sparse_direction_improvement_0p05"]
    assert gate["sparse_direction_vs_b0"]
    assert gate["dense_direction_retention"]
    assert all(gate.values())


def test_gate_rejects_exactly_subthreshold_sparse_gain() -> None:
    gate = trainer._gate_metrics(
        _metrics(0.8), _metrics(0.749), _metrics(0.8), _metrics(0.70),
        {"f1": 0.70}, True, True, True, True,
    )
    assert not gate["sparse_direction_improvement_0p05"]


def test_runner_child_contract_reads_only_combined_dataset() -> None:
    spec = {"source_dataset_run": "results/source"}
    checkpoint = {"seed": 1, "path": "results/source.pt", "sha256": "a" * 64}
    argv = runner.child_argv(spec, ROOT / runner.RUN_ID, checkpoint, Path("/venv/python"))
    assert "--dataset-run" in argv
    assert "--aee-sensor-run" not in argv
    assert "--aee-teacher-run" not in argv
    assert argv[argv.index("--seed") + 1] == "1"
