from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]


def _load():
    spec = importlib.util.spec_from_file_location(
        "gse_uncertainty_publisher_synthetic",
        ROOT / "tools/v3/publish_gse_uncertainty_analysis_figure_v1.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _synthetic_run(tmp_path: Path, publisher) -> Path:
    run = tmp_path / "results/gate3_semantics/gate3_20260824_gse_perception_validation_v1_seed0"
    calibration = run / "artifacts/calibration"
    calibration.mkdir(parents=True)
    (run / "metrics").mkdir(parents=True)
    sources = [run / "RUN_STATE.json", run / "metrics/summary.json", run / "metrics/perception_gate.json"]
    sources[0].write_text(
        json.dumps({"state": "COMPLETED", "overall_status": publisher.EXPECTED_STATUS}),
        encoding="utf-8",
    )
    sources[1].write_text(json.dumps({"overall_status": publisher.EXPECTED_STATUS}), encoding="utf-8")
    sources[2].write_text(
        json.dumps({"passed": True, "strict_test_worlds_read": 0, "mtare_worlds_read": 0}),
        encoding="utf-8",
    )
    for seed in (0, 1, 2):
        curve = [
            {
                "bin_index": index,
                "count": 10,
                "uncertainty_min": 0.05 + 0.05 * index,
                "uncertainty_mean": 0.06 + 0.05 * index,
                "uncertainty_max": 0.07 + 0.05 * index,
                "predicted_variance_mean": 0.01 + 0.01 * index,
                "geometry_residual_mean": 0.012 + 0.011 * index,
                "absolute_calibration_gap": 0.002 + 0.001 * index,
                "event_error_rate": 0.1 + 0.1 * index,
            }
            for index in range(3)
        ]
        curve_path = calibration / f"seed{seed}_uncertainty_diagnostic_curve.jsonl"
        curve_path.write_text(
            "".join(json.dumps(row) + "\n" for row in curve),
            encoding="utf-8",
        )
        seed_summary = {
            "overall_status": "PASS_GSE_VALIDATION_CALIBRATION_SEED_V1",
            "strict_test_worlds_read": 0,
            "mtare_worlds_read": 0,
            "uncertainty_diagnostic": {
                "definition": "diagnostic_only",
                "selection_effect": "NONE",
                "frames": 30,
                "bins": 3,
                "predicted_variance_mean": 0.02,
                "geometry_residual_mean": 0.023,
                "weighted_absolute_calibration_gap": 0.003,
                "geometry_residual_pearson": 0.8 - seed * 0.05,
                "event_error_pearson": 0.6 - seed * 0.05,
                "lowest_uncertainty_bin_event_error_rate": 0.1,
                "highest_uncertainty_bin_event_error_rate": 0.3,
                "curve_file": curve_path.name,
                "curve_records": 3,
            },
        }
        summary_path = calibration / f"seed{seed}_summary.json"
        summary_path.write_text(json.dumps(seed_summary), encoding="utf-8")
        sources.extend((summary_path, curve_path))
    seal = run / "artifacts/evidence_sha256.txt"
    seal.write_text(
        "".join(
            f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(tmp_path)}\n"
            for path in sources
        ),
        encoding="utf-8",
    )
    return run


def test_uncertainty_publisher_keeps_all_seed_bins_and_vector_sources(
    tmp_path: Path,
    monkeypatch,
) -> None:
    publisher = _load()
    generator = tmp_path / "tools/v3/publish.py"
    generator.parent.mkdir(parents=True)
    generator.write_text("# synthetic generator\n", encoding="utf-8")
    monkeypatch.setattr(publisher, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(publisher, "__file__", str(generator))
    run = _synthetic_run(tmp_path, publisher)
    destination = tmp_path / "docs/figures/gse_graph"
    result = publisher.publish(run, destination)
    assert result["published_files"] == 7
    for suffix in (".png", ".pdf", ".svg", ".csv", "_source.json", "_provenance.json", "_sha256.txt"):
        assert (destination / f"gse_uncertainty_analysis{suffix}").is_file()
    rows = (destination / "gse_uncertainty_analysis.csv").read_text(encoding="utf-8").splitlines()
    assert len(rows) == 1 + 3 * 3
    source = json.loads((destination / "gse_uncertainty_analysis_source.json").read_text(encoding="utf-8"))
    assert source["selection_effect"] == "NONE"
    assert set(source["seeds"]) == {"0", "1", "2"}


def test_uncertainty_publisher_rejects_selection_bearing_diagnostic(tmp_path: Path, monkeypatch) -> None:
    publisher = _load()
    monkeypatch.setattr(publisher, "PROJECT_ROOT", tmp_path)
    run = _synthetic_run(tmp_path, publisher)
    summary_path = run / "artifacts/calibration/seed1_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["uncertainty_diagnostic"]["selection_effect"] = "THRESHOLD_SELECTION"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    seal = run / "artifacts/evidence_sha256.txt"
    lines = []
    for line in seal.read_text(encoding="utf-8").splitlines():
        _, relative = line.split("  ", 1)
        path = tmp_path / relative
        lines.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {relative}\n")
    seal.write_text("".join(lines), encoding="utf-8")
    with pytest.raises(RuntimeError, match="selection-bearing"):
        publisher.publish(run, tmp_path / "docs/figures/gse_graph")
