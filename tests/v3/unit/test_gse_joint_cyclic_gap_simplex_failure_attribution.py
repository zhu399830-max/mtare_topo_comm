from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = ROOT / "tools/v3/execute_gse_joint_cyclic_gap_simplex_failure_attribution_v1.py"
SPEC = importlib.util.spec_from_file_location("jcgs_failure_attribution", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def _data(target: list[float]) -> dict[str, np.ndarray]:
    presence = np.zeros((1, 180), dtype=np.uint8)
    residual = np.zeros((1, 180), dtype=np.float32)
    for bearing in target:
        index = int(np.floor(bearing / 2.0)) % 180
        presence[0, index] = 1
        residual[0, index] = bearing - 2.0 * index
    return {"presence": presence, "heading_target": residual}


def _decoded(bearings: list[float]) -> dict[str, np.ndarray]:
    cardinality = len(bearings)
    output = np.zeros((1, 4), dtype=np.float64)
    output[0, :cardinality] = bearings
    gaps = np.zeros((1, 4), dtype=np.float64)
    gaps[0, :cardinality] = module._cyclic_gaps(np.asarray(bearings))
    return {"count": np.asarray([cardinality]), "bearing": output, "gap": gaps}


def test_cyclic_gaps_close_and_preserve_order() -> None:
    gaps = module._cyclic_gaps(np.asarray([10.0, 100.0, 220.0]))
    np.testing.assert_allclose(gaps, [90.0, 120.0, 150.0])
    assert float(gaps.sum()) == 360.0


def test_oracle_phase_isolates_phase_shift() -> None:
    data = _data([10.0, 100.0, 220.0])
    summary, _ = module._component_diagnostics(data, _decoded([30.0, 120.0, 240.0]))
    assert summary["oracle_phase_exact"]["2deg"] == 1.0
    assert summary["oracle_gaps_exact"]["10deg"] == 0.0


def test_oracle_gaps_isolates_gap_shape() -> None:
    data = _data([10.0, 100.0, 220.0])
    summary, _ = module._component_diagnostics(data, _decoded([10.0, 130.0, 240.0]))
    assert summary["oracle_phase_exact"]["10deg"] == 0.0
    assert summary["oracle_gaps_exact"]["2deg"] == 1.0


def test_binary_auc_handles_ties() -> None:
    assert module.binary_auc(np.asarray([0.0, 0.0, 1.0, 1.0]), np.asarray([False, True, False, True])) == 0.5


def test_diagnosis_requires_cross_split_agreement() -> None:
    summary = {}
    for split in ("c07", "c08"):
        summary[split] = {
            "decoders": {
                "formal_ensemble": {"exact": {"2deg": 0.1}},
                "oracle_count_ensemble": {"exact": {"2deg": 0.11}},
                "target_aligned_ensemble": {"exact": {"2deg": 0.11}},
            },
            "components": {
                "complex_oracle_phase_exact": {"10deg": 0.0},
                "complex_oracle_gaps_exact": {"10deg": 0.1},
            },
        }
    diagnosis = module._diagnosis(summary)
    assert diagnosis["cross_split_component_stable"] is True
    assert diagnosis["decision"] == "REPLACE_COMPLETE_SET_REGRESSION_WITH_EVENT_PLUS_RELATION_FACTORIZATION"
