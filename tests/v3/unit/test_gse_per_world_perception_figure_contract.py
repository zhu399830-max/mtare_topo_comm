from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]


def _load():
    spec = importlib.util.spec_from_file_location(
        "gse_per_world_perception_figure_synthetic",
        ROOT / "tools/v3/publish_gse_per_world_perception_figure_v1.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _run(tmp_path: Path, publisher, *, mismatch: bool = False) -> Path:
    run = tmp_path / "results/gate3_semantics" / publisher.EXPECTED_RUN_ID
    parents = [f"C09_parent_{index:02d}" for index in range(10)]
    _write(
        run / "RUN_STATE.json",
        {"run_id": run.name, "state": "COMPLETED", "overall_status": publisher.EXPECTED_STATUS},
    )
    _write(
        run / "metrics/summary.json",
        {"overall_status": publisher.EXPECTED_STATUS, "strict_test_worlds_read": 0, "mtare_worlds_read": 0},
    )
    _write(
        run / "metrics/perception_gate.json",
        {"passed": True, "strict_test_worlds_read": 0, "mtare_worlds_read": 0},
    )
    _write(
        run / "artifacts/nonlearning_geometry/summary.json",
        {
            "per_parent_diagnostic": {
                "selection_effect": publisher.EXPECTED_BASELINE_SELECTION_EFFECT,
                "parents": [
                    {
                        "parent_id": parent,
                        "frames": 2447 if index < 2 else 2446,
                        "geometry_mae": {field: 2.0 for field in publisher.FIELDS},
                        "geometry_valid_count": {field: 2000 for field in publisher.FIELDS},
                    }
                    for index, parent in enumerate(parents)
                ]
            }
        },
    )
    m1d_seeds = []
    for seed in range(3):
        m1d_seeds.append(
            {
                "seed": seed,
                "per_parent_event": [
                    {
                        "parent_id": parent,
                        "frames": 2447 if index < 2 else 2446,
                        "event": {"macro_f1": 0.4 + seed * 0.01},
                    }
                    for index, parent in enumerate(parents)
                ],
            }
        )
        gse_parents = [
            {
                "parent_id": parent,
                "frames": 2447 if index < 2 else 2446,
                "event": {
                    "macro_f1": 0.6 + seed * 0.01,
                    "selection_effect": publisher.EXPECTED_EVENT_SELECTION_EFFECT,
                    "temperature": 1.25 + seed,
                    "threshold": 0.55 + seed * 0.01,
                },
                "geometry_mae": {field: 1.5 for field in publisher.FIELDS},
                "geometry_valid_count": {field: 2000 for field in publisher.FIELDS},
            }
            for index, parent in enumerate(parents)
        ]
        if mismatch and seed == 2:
            gse_parents[-1]["parent_id"] = "wrong_parent"
        _write(
            run / f"artifacts/calibration/seed{seed}_summary.json",
            {
                "event": {
                    "temperature": {"temperature": 1.25 + seed},
                    "rejection_selection": {"threshold": 0.55 + seed * 0.01},
                },
                "per_parent_diagnostic": {
                    "selection_effect": publisher.EXPECTED_GSE_SELECTION_EFFECT,
                    "parents": gse_parents,
                },
            },
        )
    _write(run / "artifacts/exit_only_baseline/summary.json", {"per_seed": m1d_seeds})
    evidence = sorted(path for path in run.rglob("*") if path.is_file())
    seal = run / "artifacts/evidence_sha256.txt"
    seal.write_text(
        "".join(
            f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(tmp_path)}\n"
            for path in evidence
        ),
        encoding="utf-8",
    )
    return run


def test_per_world_publisher_retains_all_parents_seeds_fields_and_formats(tmp_path: Path, monkeypatch) -> None:
    publisher = _load()
    generator = tmp_path / "tools/v3/publisher.py"
    generator.parent.mkdir(parents=True)
    generator.write_text("# synthetic\n", encoding="utf-8")
    monkeypatch.setattr(publisher, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(publisher, "__file__", str(generator))
    run = _run(tmp_path, publisher)
    destination = tmp_path / "docs/figures/gse_graph"

    result = publisher.publish(run, destination)

    assert result["published_files"] == 7
    source = json.loads((destination / "gse_per_world_perception_source.json").read_text())
    assert len(source["parents"]) == 10
    assert len(source["rows"]) == 30
    assert source["selection_effect"].startswith("NONE_ALL_TEN")
    assert all(row["event_macro_f1_gain"] == pytest.approx(0.2) for row in source["rows"])
    assert all(row["geometry_relative_improvement_macro"] == pytest.approx(0.25) for row in source["rows"])
    assert all(set(row["geometry_valid_count"]) == set(publisher.FIELDS) for row in source["rows"])
    manifest = destination / "gse_per_world_perception_sha256.txt"
    assert len(manifest.read_text().splitlines()) == 6
    assert all((destination / f"gse_per_world_perception.{suffix}").is_file() for suffix in ("png", "pdf", "svg", "csv"))


def test_per_world_publisher_rejects_parent_selection_or_drift(tmp_path: Path, monkeypatch) -> None:
    publisher = _load()
    generator = tmp_path / "tools/v3/publisher.py"
    generator.parent.mkdir(parents=True)
    generator.write_text("# synthetic\n", encoding="utf-8")
    monkeypatch.setattr(publisher, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(publisher, "__file__", str(generator))
    run = _run(tmp_path, publisher, mismatch=True)

    with pytest.raises(RuntimeError, match="same exact ten"):
        publisher.publish(run, tmp_path / "docs/figures/gse_graph")


def test_per_world_publisher_rejects_seed_frame_population_drift(tmp_path: Path, monkeypatch) -> None:
    publisher = _load()
    generator = tmp_path / "tools/v3/publisher.py"
    generator.parent.mkdir(parents=True)
    generator.write_text("# synthetic\n", encoding="utf-8")
    monkeypatch.setattr(publisher, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(publisher, "__file__", str(generator))
    run = _run(tmp_path, publisher)
    source = run / "artifacts/calibration/seed2_summary.json"
    payload = json.loads(source.read_text())
    payload["per_parent_diagnostic"]["parents"][0]["frames"] -= 1
    source.write_text(json.dumps(payload), encoding="utf-8")
    seal = run / "artifacts/evidence_sha256.txt"
    evidence = sorted(path for path in run.rglob("*") if path.is_file() and path != seal)
    seal.write_text(
        "".join(
            f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(tmp_path)}\n"
            for path in evidence
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="frame population mismatch"):
        publisher.publish(run, tmp_path / "docs/figures/gse_graph")


@pytest.mark.parametrize("drift", ("selection_effect", "frozen_threshold", "valid_count"))
def test_per_world_publisher_rejects_selection_or_pairing_drift(
    tmp_path: Path,
    monkeypatch,
    drift: str,
) -> None:
    publisher = _load()
    generator = tmp_path / "tools/v3/publisher.py"
    generator.parent.mkdir(parents=True)
    generator.write_text("# synthetic\n", encoding="utf-8")
    monkeypatch.setattr(publisher, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(publisher, "__file__", str(generator))
    run = _run(tmp_path, publisher)
    source = run / "artifacts/calibration/seed1_summary.json"
    payload = json.loads(source.read_text())
    if drift == "selection_effect":
        payload["per_parent_diagnostic"]["selection_effect"] = "PER_PARENT_SELECTED"
        message = "selection-effect"
    elif drift == "frozen_threshold":
        payload["per_parent_diagnostic"]["parents"][0]["event"]["threshold"] += 0.01
        message = "global frozen point"
    else:
        payload["per_parent_diagnostic"]["parents"][0]["geometry_valid_count"]["width_m"] -= 1
        message = "valid-count mismatch"
    source.write_text(json.dumps(payload), encoding="utf-8")
    seal = run / "artifacts/evidence_sha256.txt"
    evidence = sorted(path for path in run.rglob("*") if path.is_file() and path != seal)
    seal.write_text(
        "".join(
            f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(tmp_path)}\n"
            for path in evidence
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match=message):
        publisher.publish(run, tmp_path / "docs/figures/gse_graph")


def test_per_world_publisher_rejects_unsealed_source_drift(tmp_path: Path, monkeypatch) -> None:
    publisher = _load()
    monkeypatch.setattr(publisher, "PROJECT_ROOT", tmp_path)
    run = _run(tmp_path, publisher)
    source = run / "artifacts/calibration/seed0_summary.json"
    source.write_text(source.read_text() + "\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="source seal drift"):
        publisher.publish(run, tmp_path / "docs/figures/gse_graph")


def test_per_world_publisher_rejects_extra_unsealed_run_file(tmp_path: Path, monkeypatch) -> None:
    publisher = _load()
    monkeypatch.setattr(publisher, "PROJECT_ROOT", tmp_path)
    run = _run(tmp_path, publisher)
    (run / "artifacts/unsealed_extra.bin").write_bytes(b"not covered")

    with pytest.raises(RuntimeError, match="exactly cover"):
        publisher.publish(run, tmp_path / "docs/figures/gse_graph")
