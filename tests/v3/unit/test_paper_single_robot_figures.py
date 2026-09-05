from __future__ import annotations

from pathlib import Path

import pytest

from mtare_topo.evaluation.corrected_stochastic_comparison_v2 import (
    analyze_v5_corrected_stochastic_cases,
)
from mtare_topo.evaluation.paper_single_robot_figures import (
    render_single_robot_paper_figures,
)
from test_corrected_stochastic_comparison import _corrected, _source


def test_renders_three_png_pdf_figure_pairs_without_manual_values(tmp_path: Path):
    source = _source()
    analysis = analyze_v5_corrected_stochastic_cases(source, _corrected(source))
    manifest = render_single_robot_paper_figures(analysis, tmp_path / "figures")
    assert manifest["figure_count"] == 3
    assert manifest["file_count"] == 6
    assert manifest["manual_value_entry"] is False
    assert {Path(item["path"]).suffix for item in manifest["files"]} == {".png", ".pdf"}
    assert all(item["bytes"] > 1000 for item in manifest["files"])
    assert all(len(item["sha256"]) == 64 for item in manifest["files"])


def test_renderer_rejects_wrong_source_schema(tmp_path: Path):
    with pytest.raises(ValueError, match="corrected V5"):
        render_single_robot_paper_figures(
            {"schema_version": "wrong"}, tmp_path / "figures"
        )
