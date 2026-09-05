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
        "gse_offline_evidence_publisher_synthetic",
        ROOT / "tools/v3/publish_gse_offline_topology_evidence_v1.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(tmp_path: Path, publisher) -> Path:
    run = tmp_path / "results/gate4_topology/gate4_20260824_gse_offline_topology_validation_v1_seed0"
    (run / "metrics").mkdir(parents=True)
    (run / "artifacts/offline_topology").mkdir(parents=True)
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
        run / "artifacts/offline_topology/summary.json": {
            "overall_status": publisher.EXPECTED_STATUS,
            "scientific_gate": {"passed": True},
        },
    }
    for path, payload in files.items():
        path.write_text(json.dumps(payload), encoding="utf-8")
    seal = run / "artifacts/evidence_sha256.txt"
    seal.write_text(
        "".join(
            f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(tmp_path)}\n"
            for path in files
        ),
        encoding="utf-8",
    )
    return run


def _module(artifact_id: str, suffixes: tuple[str, ...], *, fail: bool = False):
    def publish(_run_dir: Path, destination: Path) -> dict:
        if fail:
            (destination / "unexpected_partial.tmp").write_text("partial\n", encoding="utf-8")
            raise RuntimeError("synthetic topology child failure")
        for suffix in suffixes:
            (destination / f"{artifact_id}{suffix}").write_text("evidence\n", encoding="utf-8")
        return {"figure_id": artifact_id, "published_files": len(suffixes)}

    return SimpleNamespace(publish=publish)


def test_offline_evidence_publisher_commits_both_bundles_and_index(tmp_path: Path, monkeypatch) -> None:
    publisher = _load()
    generator = tmp_path / "tools/v3/publish.py"
    generator.parent.mkdir(parents=True)
    generator.write_text("# synthetic\n", encoding="utf-8")
    specs = (
        ("first", "first_graph", (".csv", "_sha256.txt")),
        ("second", "second_graph", (".json", "_sha256.txt")),
    )
    modules = {
        module_name: _module(artifact_id, suffixes)
        for module_name, artifact_id, suffixes in specs
    }
    monkeypatch.setattr(publisher, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(publisher, "__file__", str(generator))
    monkeypatch.setattr(publisher, "BUNDLE_SPECS", specs)
    monkeypatch.setattr(publisher, "_load_publisher", modules.__getitem__)
    run = _run(tmp_path, publisher)
    destination = tmp_path / "docs/figures/gse_graph"

    result = publisher.publish_all(run, destination)

    assert result["bundles"] == 2
    assert result["published_files"] == 6
    index = json.loads((destination / "gse_offline_topology_evidence_index.json").read_text(encoding="utf-8"))
    assert [row["artifact_id"] for row in index["bundles"]] == ["first_graph", "second_graph"]


def test_offline_evidence_publisher_declares_16_files_and_rolls_back_child_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    publisher = _load()
    assert len(publisher._targets(ROOT / "docs/figures/gse_graph")) == 16
    generator = tmp_path / "tools/v3/publish.py"
    generator.parent.mkdir(parents=True)
    generator.write_text("# synthetic\n", encoding="utf-8")
    specs = (
        ("first", "first_graph", (".csv", "_sha256.txt")),
        ("second", "second_graph", (".json", "_sha256.txt")),
    )
    modules = {
        "first": _module("first_graph", specs[0][2]),
        "second": _module("second_graph", specs[1][2], fail=True),
    }
    monkeypatch.setattr(publisher, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(publisher, "__file__", str(generator))
    monkeypatch.setattr(publisher, "BUNDLE_SPECS", specs)
    monkeypatch.setattr(publisher, "_load_publisher", modules.__getitem__)
    run = _run(tmp_path, publisher)
    destination = tmp_path / "docs/figures/gse_graph"

    with pytest.raises(RuntimeError, match="synthetic topology child failure"):
        publisher.publish_all(run, destination)
    assert not any(path.is_file() for path in destination.iterdir())
