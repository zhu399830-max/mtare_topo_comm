from __future__ import annotations

import json
from pathlib import Path

import pytest

from mtare_topo.evaluation.paper_coverage_curve import (
    CURVE_FAMILIES,
    aggregate_coverage_curves,
    load_coverage_curve,
    render_coverage_curve,
)


def _records() -> list[dict]:
    records = []
    for family_index, family in enumerate(CURVE_FAMILIES):
        for block in range(10):
            for repeat in range(3):
                scale = 1.0 + family_index + block / 10.0 + repeat / 100.0
                records.append({
                    "curve_family": family,
                    "case_id": f"{family}_{block}_{repeat}",
                    "block_id": f"block_{block}",
                    "samples": [(0.1, scale), (300.0, scale * 2.0), (599.9, scale * 3.0)],
                })
    return records


def _provenance() -> dict:
    return {
        "source_audit_run": "source",
        "source_audit_seal_sha256": "a" * 64,
        "corrected_v5_run": "corrected",
        "corrected_v5_seal_sha256": "b" * 64,
        "verified_curve_files": {f"curve_{index}": "c" * 64 for index in range(120)},
    }


def test_aggregates_ten_blocks_and_renders_png_pdf_csv_json(tmp_path: Path):
    analysis = aggregate_coverage_curves(_records())
    assert analysis["case_count"] == 120
    assert len(analysis["aggregate_rows"]) == 4 * 61
    manifest = render_coverage_curve(_records(), _provenance(), tmp_path / "coverage")
    assert manifest["figure_count"] == 1
    assert manifest["file_count"] == 5
    assert manifest["manual_value_entry"] is False
    assert {Path(item["path"]).suffix for item in manifest["files"]} == {
        ".csv", ".json", ".png", ".pdf"
    }


def test_curve_loader_rejects_non_monotonic_volume(tmp_path: Path):
    path = tmp_path / "curve.jsonl"
    path.write_text(
        "\n".join(json.dumps(row) for row in (
            {"elapsed_sec": 0.1, "explored_volume_m3": 2.0},
            {"elapsed_sec": 0.2, "explored_volume_m3": 1.0},
        )) + "\n"
    )
    with pytest.raises(ValueError, match="non-monotonic"):
        load_coverage_curve(path, expected_samples=2)
