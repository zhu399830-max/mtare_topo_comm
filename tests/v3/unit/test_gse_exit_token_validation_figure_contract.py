from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]


def _load():
    spec = importlib.util.spec_from_file_location(
        "gse_exit_token_publisher_synthetic",
        ROOT / "tools/v3/publish_gse_exit_token_validation_figure_v1.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _synthetic_run(tmp_path: Path, publisher) -> Path:
    run = tmp_path / "results/gate3_semantics/gate3_20260824_gse_perception_validation_v1_seed0"
    calibration = run / "artifacts/calibration"
    baseline = run / "artifacts/exit_only_baseline"
    metrics = run / "metrics"
    calibration.mkdir(parents=True)
    baseline.mkdir(parents=True)
    metrics.mkdir(parents=True)
    values: dict[Path, dict] = {
        run / "RUN_STATE.json": {"state": "COMPLETED", "overall_status": publisher.EXPECTED_STATUS},
        metrics / "summary.json": {"overall_status": publisher.EXPECTED_STATUS},
        metrics / "perception_gate.json": {"passed": True, "strict_test_worlds_read": 0, "mtare_worlds_read": 0},
    }
    m1d_seeds = []
    for seed in (0, 1, 2):
        m1d_seeds.append(
            {
                "seed": seed,
                "direction": {"f1": 0.6 + 0.01 * seed, "mean_matched_angular_error_deg": 8.0 + seed},
                "count": {"exact_accuracy": 0.55, "mean_absolute_error": 0.6},
            }
        )
        values[calibration / f"seed{seed}_summary.json"] = {
            "overall_status": "PASS_GSE_VALIDATION_CALIBRATION_SEED_V1",
            "exit_tokens": {
                "direction_at_selected_threshold": {
                    "matching_tolerance_deg": 20.0,
                    "f1": 0.72 + 0.01 * seed,
                    "mean_matched_angular_error_deg": 5.0 + seed,
                },
                "count_at_selected_threshold": {"exact_accuracy": 0.7, "mean_absolute_error": 0.4},
            },
        }
    values[baseline / "summary.json"] = {
        "status": "PASS_GSE_EXIT_ONLY_BASELINE_VALIDATION_V1",
        "per_seed": m1d_seeds,
    }
    for path, value in values.items():
        path.write_text(json.dumps(value), encoding="utf-8")
    seal = run / "artifacts/evidence_sha256.txt"
    seal.write_text("".join(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(tmp_path)}\n" for path in values), encoding="utf-8")
    return run


def test_exit_token_publisher_pairs_all_seeds_under_same_tolerance(tmp_path: Path, monkeypatch) -> None:
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
        assert (destination / f"gse_exit_token_validation{suffix}").is_file()
    source = json.loads((destination / "gse_exit_token_validation_source.json").read_text(encoding="utf-8"))
    assert source["matching_tolerance_deg"] == 20.0
    assert [row["seed"] for row in source["rows"]] == [0, 1, 2]


def test_exit_token_publisher_rejects_tolerance_drift(tmp_path: Path, monkeypatch) -> None:
    publisher = _load()
    monkeypatch.setattr(publisher, "PROJECT_ROOT", tmp_path)
    run = _synthetic_run(tmp_path, publisher)
    path = run / "artifacts/calibration/seed2_summary.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["exit_tokens"]["direction_at_selected_threshold"]["matching_tolerance_deg"] = 15.0
    path.write_text(json.dumps(value), encoding="utf-8")
    seal = run / "artifacts/evidence_sha256.txt"
    lines = []
    for line in seal.read_text(encoding="utf-8").splitlines():
        _, relative = line.split("  ", 1)
        source = tmp_path / relative
        lines.append(f"{hashlib.sha256(source.read_bytes()).hexdigest()}  {relative}\n")
    seal.write_text("".join(lines), encoding="utf-8")
    with pytest.raises(RuntimeError, match="tolerance contract drift"):
        publisher.publish(run, tmp_path / "docs/figures/gse_graph")


def test_exit_token_publisher_preserves_no_matched_direction_as_missing(tmp_path: Path, monkeypatch) -> None:
    publisher = _load()
    generator = tmp_path / "tools/v3/publish.py"
    generator.parent.mkdir(parents=True)
    generator.write_text("# synthetic generator\n", encoding="utf-8")
    monkeypatch.setattr(publisher, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(publisher, "__file__", str(generator))
    run = _synthetic_run(tmp_path, publisher)
    path = run / "artifacts/calibration/seed0_summary.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["exit_tokens"]["direction_at_selected_threshold"]["mean_matched_angular_error_deg"] = None
    path.write_text(json.dumps(value), encoding="utf-8")
    seal = run / "artifacts/evidence_sha256.txt"
    lines = []
    for line in seal.read_text(encoding="utf-8").splitlines():
        _, relative = line.split("  ", 1)
        source = tmp_path / relative
        lines.append(f"{hashlib.sha256(source.read_bytes()).hexdigest()}  {relative}\n")
    seal.write_text("".join(lines), encoding="utf-8")
    destination = tmp_path / "docs/figures/gse_graph"
    publisher.publish(run, destination)
    source = json.loads((destination / "gse_exit_token_validation_source.json").read_text(encoding="utf-8"))
    assert source["rows"][0]["gse_angular_error_deg"] is None
