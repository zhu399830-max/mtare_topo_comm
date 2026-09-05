from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]


def _load():
    spec = importlib.util.spec_from_file_location(
        "gse_perception_publisher_synthetic",
        ROOT / "tools/v3/publish_gse_perception_figure_v1.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_perception_publisher_keeps_every_seed_and_geometry_field(tmp_path: Path, monkeypatch) -> None:
    publisher = _load()
    generator = tmp_path / "tools/v3/publish.py"
    generator.parent.mkdir(parents=True)
    generator.write_text("# synthetic generator\n", encoding="utf-8")
    monkeypatch.setattr(publisher, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(publisher, "__file__", str(generator))
    run = tmp_path / "results/gate3_semantics/gate3_20260824_gse_perception_validation_v1_seed0"
    (run / "metrics").mkdir(parents=True)
    artifact_dirs = (
        "exit_only_baseline",
        "nonlearning_geometry",
        "calibration",
    )
    for name in artifact_dirs:
        (run / f"artifacts/{name}").mkdir(parents=True)
    gate = {
        "passed": True,
        "strict_test_worlds_read": 0,
        "mtare_worlds_read": 0,
        "event_gate": {
            "per_seed": [
                {"seed": seed, "gse_calibrated_macro_f1": 0.70 + 0.01 * seed, "exit_only_macro_f1": 0.60}
                for seed in (0, 1, 2)
            ]
        },
        "geometry_gate": {
            "per_field_relative_improvement": {
                "width_m": 0.20,
                "height_m": 0.15,
                "slope_deg": 0.12,
                "curvature_per_m": 0.11,
            }
        },
        "association_gate": {
            "per_seed": [
                {
                    "seed": seed,
                    "place_precision": 0.99,
                    "place_false_accept_rate": 0.005,
                    "exit_precision": 0.985,
                    "exit_false_accept_rate": 0.006,
                }
                for seed in (0, 1, 2)
            ]
        },
    }
    values = {
        run / "RUN_STATE.json": {"state": "COMPLETED", "overall_status": publisher.EXPECTED_STATUS},
        run / "metrics/summary.json": {"overall_status": publisher.EXPECTED_STATUS},
        run / "metrics/perception_gate.json": gate,
        run / "artifacts/exit_only_baseline/summary.json": {"status": "PASS"},
        run / "artifacts/nonlearning_geometry/summary.json": {"status": "PASS"},
        run / "artifacts/calibration/summary.json": {"status": "PASS"},
    }
    for path, value in values.items():
        path.write_text(json.dumps(value), encoding="utf-8")
    seal = run / "artifacts/evidence_sha256.txt"
    seal.write_text(
        "".join(
            f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(tmp_path)}\n"
            for path in values
        ),
        encoding="utf-8",
    )
    destination = tmp_path / "docs/figures/gse_graph"
    result = publisher.publish(run, destination)
    assert result["published_files"] == 7
    for suffix in (".png", ".pdf", ".svg", ".csv", "_source.json", "_provenance.json", "_sha256.txt"):
        assert (destination / f"gse_perception_validation{suffix}").is_file()
    rows = (destination / "gse_perception_validation.csv").read_text(encoding="utf-8").splitlines()
    assert len(rows) == 1 + 6 + 4 + 12


def test_perception_publisher_rejects_mtare_read(tmp_path: Path, monkeypatch) -> None:
    publisher = _load()
    monkeypatch.setattr(publisher, "PROJECT_ROOT", tmp_path)
    run = tmp_path / "results/gate3_semantics/gate3_20260824_gse_perception_validation_v1_seed0"
    (run / "metrics").mkdir(parents=True)
    (run / "RUN_STATE.json").write_text(
        json.dumps({"state": "COMPLETED", "overall_status": publisher.EXPECTED_STATUS}),
        encoding="utf-8",
    )
    (run / "metrics/summary.json").write_text(
        json.dumps({"overall_status": publisher.EXPECTED_STATUS}), encoding="utf-8"
    )
    (run / "metrics/perception_gate.json").write_text(
        json.dumps({"passed": True, "strict_test_worlds_read": 0, "mtare_worlds_read": 1}),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="sealed validation PASS"):
        publisher.publish(run, tmp_path / "docs/figures/gse_graph")
