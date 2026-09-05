from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]


def _load():
    spec = importlib.util.spec_from_file_location(
        "gse_rejection_publisher_synthetic",
        ROOT / "tools/v3/publish_gse_rejection_analysis_figure_v1.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


def _synthetic_run(tmp_path: Path, publisher) -> tuple[Path, list[Path]]:
    run = tmp_path / "results/gate3_semantics/gate3_20260824_gse_perception_validation_v1_seed0"
    calibration = run / "artifacts/calibration"
    calibration.mkdir(parents=True)
    (run / "metrics").mkdir(parents=True)
    gate = {"passed": True, "strict_test_worlds_read": 0, "mtare_worlds_read": 0}
    paths = [run / "RUN_STATE.json", run / "metrics/summary.json", run / "metrics/perception_gate.json"]
    paths[0].write_text(json.dumps({"state": "COMPLETED", "overall_status": publisher.EXPECTED_STATUS}), encoding="utf-8")
    paths[1].write_text(json.dumps({"overall_status": publisher.EXPECTED_STATUS}), encoding="utf-8")
    paths[2].write_text(json.dumps(gate), encoding="utf-8")
    for seed in (0, 1, 2):
        event = [
            {"threshold": 0.3, "accepted_structural_events": 100, "macro_f1": 0.61 + seed * 0.01, "structural_recall": 0.7},
            {"threshold": 0.6, "accepted_structural_events": 60, "macro_f1": 0.70 + seed * 0.01, "structural_recall": 0.55},
        ]
        association = [
            {"threshold": 0.95, "accepted": 30, "precision": 1.0, "recall": 0.3, "false_accept_rate": 0.0},
            {"threshold": 0.85, "accepted": 50, "precision": 0.98, "recall": 0.49, "false_accept_rate": 0.01},
        ]
        seed_summary = {
            "overall_status": "PASS_GSE_VALIDATION_CALIBRATION_SEED_V1",
            "strict_test_worlds_read": 0,
            "mtare_worlds_read": 0,
            "event": {"rejection_selection": {**event[1], "maximum_uncertainty": 0.4, "structural_precision": 0.99, "rejected_structural_predictions": 40}},
            "place_association": {"selection": {**association[1], "true_positive": 49, "false_positive": 1}},
            "exit_tokens": {"descriptor_association": {"selection": {**association[0], "true_positive": 30, "false_positive": 0}}},
        }
        seed_summary_path = calibration / f"seed{seed}_summary.json"
        event_path = calibration / f"seed{seed}_event_rejection_curve.jsonl"
        place_path = calibration / f"seed{seed}_place_association_curve.jsonl"
        exit_path = calibration / f"seed{seed}_exit_descriptor_curve.jsonl"
        seed_summary_path.write_text(json.dumps(seed_summary), encoding="utf-8")
        _write_jsonl(event_path, event)
        _write_jsonl(place_path, association)
        _write_jsonl(exit_path, association)
        paths.extend((seed_summary_path, event_path, place_path, exit_path))
    seal = run / "artifacts/evidence_sha256.txt"
    seal.write_text(
        "".join(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(tmp_path)}\n" for path in paths),
        encoding="utf-8",
    )
    return run, paths


def test_rejection_publisher_keeps_complete_three_seed_curves(tmp_path: Path, monkeypatch) -> None:
    publisher = _load()
    generator = tmp_path / "tools/v3/publish.py"
    generator.parent.mkdir(parents=True)
    generator.write_text("# synthetic generator\n", encoding="utf-8")
    monkeypatch.setattr(publisher, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(publisher, "__file__", str(generator))
    run, _ = _synthetic_run(tmp_path, publisher)
    destination = tmp_path / "docs/figures/gse_graph"
    result = publisher.publish(run, destination)
    assert result["published_files"] == 7
    for suffix in (".png", ".pdf", ".svg", ".csv", "_source.json", "_provenance.json", "_sha256.txt"):
        assert (destination / f"gse_rejection_analysis{suffix}").is_file()
    rows = (destination / "gse_rejection_analysis.csv").read_text(encoding="utf-8").splitlines()
    assert len(rows) == 1 + 3 * (2 + 2 + 2)
    source = json.loads((destination / "gse_rejection_analysis_source.json").read_text(encoding="utf-8"))
    assert set(source["seeds"]) == {"0", "1", "2"}


def test_rejection_publisher_rejects_unsealed_curve_drift(tmp_path: Path, monkeypatch) -> None:
    publisher = _load()
    monkeypatch.setattr(publisher, "PROJECT_ROOT", tmp_path)
    run, _ = _synthetic_run(tmp_path, publisher)
    curve = run / "artifacts/calibration/seed1_place_association_curve.jsonl"
    curve.write_text(curve.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="absent from the run seal"):
        publisher.publish(run, tmp_path / "docs/figures/gse_graph")
