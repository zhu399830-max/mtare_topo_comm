"""Read-only bridge from finalized V2 case summaries to the frozen V1 analysis.

The host-archive runner finalizes scientifically complete cases with a V2
status string.  The pre-registered statistics implementation predates that
archive boundary and accepts the otherwise equivalent V1 status.  This module
keeps both historical interfaces immutable: it validates V2 inputs, changes
only the status in deep in-memory copies, and delegates to the frozen
statistics implementation.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping, Sequence

from mtare_topo.evaluation.stochastic_closed_loop import analyze_stochastic_cases


SOURCE_STATUS = "PASS_SINGLE_ROBOT_CASE_V2"
ANALYZER_STATUS = "PASS_SINGLE_ROBOT_CASE_V1"
SUMMARY_SCHEMA = "mtare_single_robot_case_summary_v1"
CASE_SCHEMA = "mtare_single_robot_case_contract_v1"


def analyze_finalized_v2_cases(
    summaries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Analyze finalized V2 cases without mutating their sealed evidence."""

    normalized: list[dict[str, Any]] = []
    for index, summary in enumerate(summaries):
        if summary.get("schema_version") != SUMMARY_SCHEMA:
            raise ValueError(f"case {index} has an unsupported summary schema")
        if summary.get("status") != SOURCE_STATUS:
            raise ValueError(f"case {index} is not a finalized V2 PASS")
        case = summary.get("case")
        if not isinstance(case, Mapping) or case.get("schema_version") != CASE_SCHEMA:
            raise ValueError(f"case {index} has an unsupported case contract")
        copied = deepcopy(dict(summary))
        copied["status"] = ANALYZER_STATUS
        normalized.append(copied)

    analysis = analyze_stochastic_cases(normalized)
    analysis["compatibility_bridge"] = {
        "schema_version": "stochastic_case_status_bridge_v1",
        "source_case_status": SOURCE_STATUS,
        "analyzer_case_status": ANALYZER_STATUS,
        "source_case_count": len(summaries),
        "source_mutation_permitted": False,
        "mapped_field": "status",
        "mapped_field_count": len(summaries),
    }
    return analysis


__all__ = [
    "ANALYZER_STATUS",
    "CASE_SCHEMA",
    "SOURCE_STATUS",
    "SUMMARY_SCHEMA",
    "analyze_finalized_v2_cases",
]
