import pytest

from tools.v3.run_primitive_composition_anchor_c07_evaluation_corrective_v1 import (
    EXPECTED_EVALUATOR_ERROR,
    EXPECTED_SOURCE_ERROR,
    parse_nvidia_process_memory,
    source_failure_signature,
)


def _source_summary():
    return {
        "error": EXPECTED_SOURCE_ERROR,
        "optimizer_steps": 30_699,
        "c08_rows_read": 0,
        "graph_replays": 0,
        "mtare_worlds_read": 0,
        "subprocesses": [
            {"stage": "seed0_training", "returncode": 0},
            {"stage": "seed1_training", "returncode": 0},
            {"stage": "seed2_training", "returncode": 0},
            {"stage": "c07_evaluation", "returncode": 1},
        ],
    }


def test_source_failure_signature_requires_completed_training_and_exact_evaluator_error():
    state = {"state": "FAILED", "error": EXPECTED_SOURCE_ERROR}
    assert source_failure_signature(state, _source_summary(), EXPECTED_EVALUATOR_ERROR)
    changed = _source_summary()
    changed["optimizer_steps"] -= 1
    assert not source_failure_signature(state, changed, EXPECTED_EVALUATOR_ERROR)
    assert not source_failure_signature(state, _source_summary(), "different failure")


def test_parse_nvidia_process_memory_is_pid_specific_and_uses_binary_mebibytes():
    text = "100, 25\n205, 137 MiB\n"
    assert parse_nvidia_process_memory(text, 205) == 137 * 1024**2
    assert parse_nvidia_process_memory(text, 999) == 0


def test_parse_nvidia_process_memory_rejects_malformed_selected_value():
    with pytest.raises(ValueError):
        parse_nvidia_process_memory("205, not-a-number", 205)
