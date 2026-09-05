from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[3]


def _load():
    spec = importlib.util.spec_from_file_location(
        "gse_training_evidence_publisher_synthetic",
        ROOT / "tools/v3/publish_gse_training_evidence_v1.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(tmp_path: Path, publisher) -> Path:
    run = tmp_path / "results/gate2_representation/gate2_20260824_gse_graph_three_seed_training_v1r_seed0"
    (run / "metrics").mkdir(parents=True)
    (run / "artifacts").mkdir()
    files = {
        run / "RUN_STATE.json": {
            "run_id": run.name,
            "state": "COMPLETED",
            "overall_status": publisher.EXPECTED_STATUS,
        },
        run / "metrics/summary.json": {
            "overall_status": publisher.EXPECTED_STATUS,
            "strict_test_worlds_read": 0,
            "mtare_worlds_read": 0,
        },
    }
    for path, payload in files.items():
        path.write_text(json.dumps(payload), encoding="utf-8")
    (run / "artifacts/evidence_sha256.txt").write_text(
        "".join(
            f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(tmp_path)}\n"
            for path in files
        ),
        encoding="utf-8",
    )
    return run


def _child(publisher, *, fail: bool = False):
    def publish(_run_dir: Path, destination: Path) -> dict:
        for suffix in publisher.SUFFIXES[:2]:
            (destination / f"{publisher.FIGURE_ID}{suffix}").write_text("partial\n", encoding="utf-8")
        if fail:
            (destination / "unexpected.tmp").write_text("partial\n", encoding="utf-8")
            raise RuntimeError("synthetic training child failure")
        for suffix in publisher.SUFFIXES[2:]:
            (destination / f"{publisher.FIGURE_ID}{suffix}").write_text("evidence\n", encoding="utf-8")
        return {"figure_id": publisher.FIGURE_ID, "published_files": len(publisher.SUFFIXES)}

    return SimpleNamespace(publish=publish)


def test_training_evidence_publisher_commits_exact_seven_files(tmp_path: Path, monkeypatch) -> None:
    publisher = _load()
    monkeypatch.setattr(publisher, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(publisher, "_load_publisher", lambda: _child(publisher))
    run = _run(tmp_path, publisher)
    destination = tmp_path / "docs/figures/gse_graph"

    result = publisher.publish(run, destination)

    assert result["published_files"] == 7
    assert result["atomic_bundle"] is True
    assert len([path for path in destination.iterdir() if path.is_file()]) == 7


def test_training_evidence_publisher_rolls_back_all_new_files(tmp_path: Path, monkeypatch) -> None:
    publisher = _load()
    monkeypatch.setattr(publisher, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(publisher, "_load_publisher", lambda: _child(publisher, fail=True))
    run = _run(tmp_path, publisher)
    destination = tmp_path / "docs/figures/gse_graph"

    with pytest.raises(RuntimeError, match="synthetic training child failure"):
        publisher.publish(run, destination)
    assert not any(path.is_file() for path in destination.iterdir())
