from __future__ import annotations

from copy import deepcopy
import json

import pytest

from mtare_topo.evaluation.corrected_stochastic_comparison import (
    analyze_corrected_stochastic_cases,
    compose_corrected_cases,
)
from mtare_topo.evaluation.stochastic_closed_loop import METRICS
from mtare_topo.evaluation.stochastic_closed_loop_v2_bridge import SOURCE_STATUS
from test_stochastic_closed_loop import synthetic_summaries


def _source():
    values = deepcopy(synthetic_summaries())
    for index, item in enumerate(values):
        item["schema_version"] = "mtare_single_robot_case_summary_v1"
        item["status"] = SOURCE_STATUS
        item["case"].update(
            schema_version="mtare_single_robot_case_contract_v1",
            method_id=item["case"]["method_family"],
            world="tunnel" if index < 45 else "garage",
            environment_seed=(11, 23, 37, 53, 71)[(index // 9) % 5],
            execution_repeat=index % 3,
            checkpoint_seed=index % 3 if item["case"]["method_family"] == "m1d_topology" else None,
            runtime_sec=600,
        )
    return values


def _corrected(source):
    values = [deepcopy(item) for item in source if item["case"]["method_family"] == "m1d_topology"]
    for item in values:
        for metric in METRICS:
            item["metrics"][metric] += 5.0
    return values


def test_corrected_composition_replaces_only_m1d_without_mutating_sources() -> None:
    source = _source()
    corrected = _corrected(source)
    before = json.dumps([source, corrected], sort_keys=True)
    combined, provenance = compose_corrected_cases(source, corrected)
    assert len(combined) == 90
    assert provenance["replaced_case_count"] == 30
    assert provenance["reused_original_mtare_case_count"] == 30
    assert json.dumps([source, corrected], sort_keys=True) == before
    for item in combined:
        original = next(value for value in source if value["case"]["case_id"] == item["case"]["case_id"])
        delta = item["metrics"][METRICS[0]] - original["metrics"][METRICS[0]]
        assert delta == (5.0 if item["case"]["method_family"] == "m1d_topology" else 0.0)


def test_corrected_analysis_reports_block_paired_fix_effect() -> None:
    source = _source()
    result = analyze_corrected_stochastic_cases(source, _corrected(source))
    assert result["corrected_main_analysis"]["case_count"] == 90
    for metric in METRICS:
        effect = result["v4_vs_defective_v9"]["metrics"][metric]
        assert effect["mean_difference"] == pytest.approx(5.0)
        assert effect["paired_bootstrap_95_ci"] == pytest.approx([5.0, 5.0])
        assert effect["exact_two_sided_sign_flip_p"] == pytest.approx(2.0 / 1024.0)


def test_corrected_composition_fails_on_identity_drift() -> None:
    source = _source()
    corrected = _corrected(source)
    corrected[0]["case"]["environment_seed"] = 999
    with pytest.raises(ValueError, match="identity drift"):
        compose_corrected_cases(source, corrected)
