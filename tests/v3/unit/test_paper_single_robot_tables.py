from __future__ import annotations

import csv
from pathlib import Path

import pytest

from mtare_topo.evaluation.corrected_stochastic_comparison_v2 import (
    analyze_v5_corrected_stochastic_cases,
)
from mtare_topo.evaluation.paper_single_robot_tables import (
    render_single_robot_paper_tables,
)
from test_corrected_stochastic_comparison import _corrected, _source


def _mechanisms() -> dict:
    return {
        "completed_case_count": 30,
        "verified_reanchor_count_total": 12,
        "verified_reanchor_case_count": 5,
        "frontier_execution_rejection_count_total": 9,
        "frontier_execution_rejection_case_count": 4,
        "case_count_with_both_correction_types": 2,
        "frontier_execution_outcomes": {"departure_mismatch": 7, "same_node_loop": 2},
    }


def _provenance() -> dict:
    return {
        "source_audit_run": "results/gate6_single_robot/source",
        "source_audit_seal_sha256": "a" * 64,
        "corrected_v5_run": "results/gate6_single_robot/corrected",
        "corrected_v5_seal_sha256": "b" * 64,
    }


def test_writes_three_csv_markdown_latex_table_sets(tmp_path: Path):
    source = _source()
    analysis = analyze_v5_corrected_stochastic_cases(source, _corrected(source))
    manifest = render_single_robot_paper_tables(
        analysis, _mechanisms(), _provenance(), tmp_path / "tables"
    )
    assert manifest["table_count"] == 3
    assert manifest["file_count"] == 9
    assert manifest["manual_value_entry"] is False
    assert {Path(item["path"]).suffix for item in manifest["files"]} == {
        ".csv", ".md", ".tex"
    }
    assert all(item["bytes"] > 50 and len(item["sha256"]) == 64 for item in manifest["files"])
    with (tmp_path / "tables/single_robot_paired_effects.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 14
    assert {row["Comparison"] for row in rows} == {
        "V5 - Original M-TARE", "V5 - Defective V9"
    }
    assert all(row["Holm p"] for row in rows)


def test_rejects_missing_mechanism_or_provenance(tmp_path: Path):
    source = _source()
    analysis = analyze_v5_corrected_stochastic_cases(source, _corrected(source))
    with pytest.raises(ValueError, match="30 corrected"):
        render_single_robot_paper_tables(
            analysis, {}, _provenance(), tmp_path / "missing_mechanism"
        )
    with pytest.raises(ValueError, match="provenance"):
        render_single_robot_paper_tables(
            analysis, _mechanisms(), {}, tmp_path / "missing_provenance"
        )
