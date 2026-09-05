from __future__ import annotations

from copy import deepcopy
import json

import pytest

from mtare_topo.evaluation.stochastic_closed_loop import analyze_stochastic_cases
from mtare_topo.evaluation.stochastic_closed_loop_v2_bridge import (
    ANALYZER_STATUS,
    SOURCE_STATUS,
    analyze_finalized_v2_cases,
)

from test_stochastic_closed_loop import synthetic_summaries


def finalized_v2_summaries():
    values = deepcopy(synthetic_summaries())
    for item in values:
        item["schema_version"] = "mtare_single_robot_case_summary_v1"
        item["status"] = SOURCE_STATUS
        item["case"]["schema_version"] = "mtare_single_robot_case_contract_v1"
    return values


def test_v2_bridge_matches_frozen_analysis_without_source_mutation() -> None:
    summaries = finalized_v2_summaries()
    before = json.dumps(summaries, sort_keys=True, separators=(",", ":"))
    bridged = analyze_finalized_v2_cases(summaries)
    after = json.dumps(summaries, sort_keys=True, separators=(",", ":"))

    expected_input = deepcopy(summaries)
    for item in expected_input:
        item["status"] = ANALYZER_STATUS
    expected = analyze_stochastic_cases(expected_input)

    compatibility = bridged.pop("compatibility_bridge")
    assert bridged == expected
    assert before == after
    assert compatibility == {
        "schema_version": "stochastic_case_status_bridge_v1",
        "source_case_status": SOURCE_STATUS,
        "analyzer_case_status": ANALYZER_STATUS,
        "source_case_count": 90,
        "source_mutation_permitted": False,
        "mapped_field": "status",
        "mapped_field_count": 90,
    }


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("status", ANALYZER_STATUS, "not a finalized V2 PASS"),
        ("status", "PASS_SINGLE_ROBOT_CASE_V2_PENDING_HOST_ARCHIVE", "not a finalized V2 PASS"),
        ("schema_version", "unknown", "unsupported summary schema"),
    ],
)
def test_v2_bridge_rejects_nonfinal_or_unknown_summaries(field, value, message) -> None:
    summaries = finalized_v2_summaries()
    summaries[17][field] = value
    with pytest.raises(ValueError, match=message):
        analyze_finalized_v2_cases(summaries)


def test_v2_bridge_rejects_unknown_case_contract() -> None:
    summaries = finalized_v2_summaries()
    summaries[4]["case"]["schema_version"] = "unknown"
    with pytest.raises(ValueError, match="unsupported case contract"):
        analyze_finalized_v2_cases(summaries)
