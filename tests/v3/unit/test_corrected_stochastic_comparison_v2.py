from __future__ import annotations

from copy import deepcopy

from mtare_topo.evaluation.corrected_stochastic_comparison_v2 import (
    analyze_v5_corrected_stochastic_cases,
)
from mtare_topo.evaluation.stochastic_closed_loop import METRICS
from test_corrected_stochastic_comparison import _corrected, _source


def test_v5_comparison_preserves_all_statistics_and_uses_truthful_labels():
    source = _source()
    corrected = _corrected(source)
    value = analyze_v5_corrected_stochastic_cases(source, corrected)
    assert value["schema_version"] == "corrected_v5_stochastic_comparison_v1"
    assert value["corrected_main_analysis"]["case_count"] == 90
    assert set(value["v5_vs_defective_v9"]["metrics"]) == set(METRICS)
    for result in value["v5_vs_defective_v9"]["metrics"].values():
        assert "v5_minus_defective_v9_block_differences" in result
        assert "v4_minus_v9_block_differences" not in result
        assert result["defective_v9_mean"] > 0.0
        assert len(result["relative_bootstrap_95_ci"]) == 2
        assert 0.0 <= result["holm_adjusted_p"] <= 1.0
    assert "V5" in value["composition"]["corrected_method"]


def test_v5_comparison_does_not_mutate_inputs():
    source = _source()
    corrected = _corrected(source)
    source_before, corrected_before = deepcopy(source), deepcopy(corrected)
    analyze_v5_corrected_stochastic_cases(source, corrected)
    assert source == source_before
    assert corrected == corrected_before
