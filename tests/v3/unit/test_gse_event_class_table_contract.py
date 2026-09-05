from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
EVENTS = ("corridor", "junction", "terminal", "turn", "geometry_transition")


def _load():
    spec = importlib.util.spec_from_file_location(
        "gse_event_class_table_synthetic",
        ROOT / "tools/v3/publish_gse_event_class_table_v1.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _class_metrics(seed: int, *, gse: bool) -> dict:
    support = {
        "corridor": 50,
        "junction": 20,
        "terminal": 10,
        "turn": 8,
        "geometry_transition": 12,
    }
    result = {}
    for index, event in enumerate(EVENTS):
        base = 0.50 + 0.02 * index + 0.01 * seed
        gain = 0.10 if gse else 0.0
        result[event] = {
            "precision": base + gain,
            "recall": base + gain - 0.01,
            "f1": base + gain - 0.005,
            "support": support[event],
            "predicted": support[event],
        }
    return result


def _synthetic_run(tmp_path: Path, publisher) -> Path:
    run = tmp_path / "results/gate3_semantics/gate3_20260824_gse_perception_validation_v1_seed0"
    calibration = run / "artifacts/calibration"
    baseline_dir = run / "artifacts/exit_only_baseline"
    calibration.mkdir(parents=True)
    baseline_dir.mkdir(parents=True)
    (run / "metrics").mkdir(parents=True)
    sources = {
        run / "RUN_STATE.json": {
            "state": "COMPLETED",
            "overall_status": publisher.EXPECTED_STATUS,
        },
        run / "metrics/summary.json": {"overall_status": publisher.EXPECTED_STATUS},
        run / "metrics/perception_gate.json": {
            "passed": True,
            "strict_test_worlds_read": 0,
            "mtare_worlds_read": 0,
        },
        baseline_dir / "summary.json": {
            "status": "PASS_GSE_EXIT_ONLY_BASELINE_VALIDATION_V1",
            "per_seed": [
                {"seed": seed, "event": {"per_class": _class_metrics(seed, gse=False)}}
                for seed in (0, 1, 2)
            ],
        },
    }
    for seed in (0, 1, 2):
        macro_f1 = sum(item["f1"] for item in _class_metrics(seed, gse=True).values()) / len(EVENTS)
        sources[calibration / f"seed{seed}_summary.json"] = {
            "overall_status": "PASS_GSE_VALIDATION_CALIBRATION_SEED_V1",
            "strict_test_worlds_read": 0,
            "mtare_worlds_read": 0,
            "event": {
                "rejection_selection": {"macro_f1": macro_f1},
                "selected_class_metrics": {
                    "selection_effect": "NONE_REPLAY_OF_FROZEN_EVENT_POINT",
                    "macro_f1": macro_f1,
                    "per_class": _class_metrics(seed, gse=True),
                },
            },
        }
    for path, payload in sources.items():
        path.write_text(json.dumps(payload), encoding="utf-8")
    seal = run / "artifacts/evidence_sha256.txt"
    seal.write_text(
        "".join(
            f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(tmp_path)}\n"
            for path in sources
        ),
        encoding="utf-8",
    )
    return run


def _reseal(tmp_path: Path, run: Path) -> None:
    seal = run / "artifacts/evidence_sha256.txt"
    lines = []
    for line in seal.read_text(encoding="utf-8").splitlines():
        _, relative = line.split("  ", 1)
        path = tmp_path / relative
        lines.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {relative}\n")
    seal.write_text("".join(lines), encoding="utf-8")


def test_event_class_table_keeps_all_classes_methods_and_seeds(tmp_path: Path, monkeypatch) -> None:
    publisher = _load()
    generator = tmp_path / "tools/v3/publish.py"
    generator.parent.mkdir(parents=True)
    generator.write_text("# synthetic generator\n", encoding="utf-8")
    monkeypatch.setattr(publisher, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(publisher, "__file__", str(generator))
    run = _synthetic_run(tmp_path, publisher)
    destination = tmp_path / "docs/figures/gse_graph"

    result = publisher.publish(run, destination)

    assert result["published_files"] == 6
    for suffix in (".csv", ".md", ".tex", "_source.json", "_provenance.json", "_sha256.txt"):
        assert (destination / f"gse_event_class_table{suffix}").is_file()
    rows = (destination / "gse_event_class_table.csv").read_text(encoding="utf-8").splitlines()
    assert len(rows) == 1 + len(EVENTS) * 2 * 3
    source = json.loads((destination / "gse_event_class_table_source.json").read_text(encoding="utf-8"))
    assert source["selection_effect"] == "NONE_REPORTS_FROZEN_SELECTED_POINTS"
    assert set(source["aggregate"]) == set(EVENTS)
    assert source["aggregate"]["turn"]["f1_absolute_improvement"] == pytest.approx(0.10)
    assert "$\\pm$" in (destination / "gse_event_class_table.tex").read_text(encoding="utf-8")


def test_event_class_table_rejects_selection_bearing_gse_metrics(tmp_path: Path, monkeypatch) -> None:
    publisher = _load()
    monkeypatch.setattr(publisher, "PROJECT_ROOT", tmp_path)
    run = _synthetic_run(tmp_path, publisher)
    summary_path = run / "artifacts/calibration/seed1_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["event"]["selected_class_metrics"]["selection_effect"] = "NEW_TABLE_SELECTION"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    _reseal(tmp_path, run)

    with pytest.raises(RuntimeError, match="selected event-class evidence is invalid"):
        publisher.publish(run, tmp_path / "docs/figures/gse_graph")
