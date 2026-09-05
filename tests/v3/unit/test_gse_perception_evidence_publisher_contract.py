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
        "gse_perception_evidence_publisher_synthetic",
        ROOT / "tools/v3/publish_gse_perception_evidence_v1.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _synthetic_run(tmp_path: Path, publisher) -> Path:
    run = tmp_path / "results/gate3_semantics/gate3_20260824_gse_perception_validation_v1_seed0"
    (run / "metrics").mkdir(parents=True)
    (run / "artifacts").mkdir()
    (run / "RUN_STATE.json").write_text(
        json.dumps(
            {
                "run_id": run.name,
                "state": "COMPLETED",
                "overall_status": publisher.EXPECTED_STATUS,
            }
        ),
        encoding="utf-8",
    )
    (run / "metrics/summary.json").write_text(
        json.dumps(
            {
                "overall_status": publisher.EXPECTED_STATUS,
                "strict_test_worlds_read": 0,
                "mtare_worlds_read": 0,
            }
        ),
        encoding="utf-8",
    )
    (run / "metrics/perception_gate.json").write_text(
        json.dumps({"passed": True, "strict_test_worlds_read": 0, "mtare_worlds_read": 0}),
        encoding="utf-8",
    )
    evidence = (
        run / "RUN_STATE.json",
        run / "metrics/summary.json",
        run / "metrics/perception_gate.json",
    )
    (run / "artifacts/evidence_sha256.txt").write_text(
        "".join(
            f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(tmp_path)}\n"
            for path in evidence
        ),
        encoding="utf-8",
    )
    return run


def _fake_module(artifact_id: str, suffixes: tuple[str, ...], *, fail: bool = False):
    def publish(_run: Path, destination: Path) -> dict:
        if fail:
            (destination / "unexpected_partial.tmp").write_text("partial\n", encoding="utf-8")
            raise RuntimeError("synthetic child failure")
        for suffix in suffixes:
            (destination / f"{artifact_id}{suffix}").write_text(
                f"{artifact_id}{suffix}\n", encoding="utf-8"
            )
        return {"figure_id": artifact_id, "published_files": len(suffixes)}

    return SimpleNamespace(publish=publish)


def test_complete_perception_publisher_commits_all_bundles_and_index(tmp_path: Path, monkeypatch) -> None:
    publisher = _load()
    generator = tmp_path / "tools/v3/publish.py"
    generator.parent.mkdir(parents=True)
    generator.write_text("# synthetic generator\n", encoding="utf-8")
    specs = (
        ("first", "first_bundle", (".csv", "_sha256.txt")),
        ("second", "second_bundle", (".json", "_sha256.txt")),
    )
    modules = {
        module_name: _fake_module(artifact_id, suffixes)
        for module_name, artifact_id, suffixes in specs
    }
    monkeypatch.setattr(publisher, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(publisher, "__file__", str(generator))
    monkeypatch.setattr(publisher, "BUNDLE_SPECS", specs)
    monkeypatch.setattr(publisher, "_load_publisher", modules.__getitem__)
    run = _synthetic_run(tmp_path, publisher)
    destination = tmp_path / "docs/figures/gse_graph"

    result = publisher.publish_all(run, destination)

    assert result["bundles"] == 2
    assert result["published_files"] == 6
    index = json.loads((destination / "gse_perception_evidence_index.json").read_text(encoding="utf-8"))
    assert [record["artifact_id"] for record in index["bundles"]] == ["first_bundle", "second_bundle"]
    manifest = destination / "gse_perception_evidence_index_sha256.txt"
    assert len(manifest.read_text(encoding="utf-8").splitlines()) == 5
    assert result["manifest_sha256"] == hashlib.sha256(manifest.read_bytes()).hexdigest()


def test_production_perception_publication_declares_exactly_43_files() -> None:
    publisher = _load()
    destination = ROOT / "docs/figures/gse_graph"
    assert len(publisher._bundle_targets(destination)) == 41
    assert len(publisher._bundle_targets(destination)) + 2 == 43


def test_complete_perception_publisher_rolls_back_explicit_targets_on_child_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    publisher = _load()
    generator = tmp_path / "tools/v3/publish.py"
    generator.parent.mkdir(parents=True)
    generator.write_text("# synthetic generator\n", encoding="utf-8")
    specs = (
        ("first", "first_bundle", (".csv", "_sha256.txt")),
        ("second", "second_bundle", (".json", "_sha256.txt")),
    )
    modules = {
        "first": _fake_module("first_bundle", specs[0][2]),
        "second": _fake_module("second_bundle", specs[1][2], fail=True),
    }
    monkeypatch.setattr(publisher, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(publisher, "__file__", str(generator))
    monkeypatch.setattr(publisher, "BUNDLE_SPECS", specs)
    monkeypatch.setattr(publisher, "_load_publisher", modules.__getitem__)
    run = _synthetic_run(tmp_path, publisher)
    destination = tmp_path / "docs/figures/gse_graph"

    with pytest.raises(RuntimeError, match="synthetic child failure"):
        publisher.publish_all(run, destination)

    assert not any(path.is_file() for path in destination.iterdir())


@pytest.mark.parametrize(
    ("field", "value"),
    (("passed", False), ("strict_test_worlds_read", 1), ("mtare_worlds_read", 1)),
)
def test_complete_perception_publisher_rejects_invalid_gate(
    tmp_path: Path,
    monkeypatch,
    field: str,
    value: object,
) -> None:
    publisher = _load()
    monkeypatch.setattr(publisher, "PROJECT_ROOT", tmp_path)
    run = _synthetic_run(tmp_path, publisher)
    gate_path = run / "metrics/perception_gate.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    gate[field] = value
    gate_path.write_text(json.dumps(gate), encoding="utf-8")
    with pytest.raises(RuntimeError, match="sealed C09 PASS"):
        publisher.publish_all(run, tmp_path / "docs/figures/gse_graph")


def test_complete_perception_publisher_rejects_source_seal_drift(tmp_path: Path, monkeypatch) -> None:
    publisher = _load()
    monkeypatch.setattr(publisher, "PROJECT_ROOT", tmp_path)
    run = _synthetic_run(tmp_path, publisher)
    summary_path = run / "metrics/summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "overall_status": publisher.EXPECTED_STATUS,
                "strict_test_worlds_read": 0,
                "mtare_worlds_read": 0,
                "drift": True,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="source seal drift"):
        publisher.publish_all(run, tmp_path / "docs/figures/gse_graph")
