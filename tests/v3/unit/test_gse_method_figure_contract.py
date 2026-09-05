from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def _load():
    spec = importlib.util.spec_from_file_location(
        "gse_method_publisher_synthetic",
        ROOT / "tools/v3/publish_gse_method_figure_v1.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_method_publisher_keeps_structured_source_and_all_formats(tmp_path: Path, monkeypatch) -> None:
    publisher = _load()
    generator = tmp_path / "tools/v3/publish.py"
    generator.parent.mkdir(parents=True)
    generator.write_text("# synthetic generator\n", encoding="utf-8")
    contract = tmp_path / "docs/GSE_GRAPH_RESEARCH_PLAN_V1.md"
    contract.parent.mkdir(parents=True)
    contract.write_text("# synthetic method contract\n", encoding="utf-8")
    monkeypatch.setattr(publisher, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(publisher, "__file__", str(generator))

    destination = tmp_path / "docs/figures/gse_graph"
    result = publisher.publish(destination)

    assert result["published_files"] == 6
    for suffix in (".png", ".pdf", ".svg", "_source.json", "_provenance.json", "_sha256.txt"):
        assert (destination / f"gse_method_overview{suffix}").is_file()
    source = json.loads((destination / "gse_method_overview_source.json").read_text(encoding="utf-8"))
    assert len(source["nodes"]) == 7
    assert len(source["edges"]) == 6
    assert source["claims"]["edge_creation"] == "physical traversal evidence only"
    provenance = json.loads(
        (destination / "gse_method_overview_provenance.json").read_text(encoding="utf-8")
    )
    source_path = tmp_path / provenance["machine_readable_source"]
    assert hashlib.sha256(source_path.read_bytes()).hexdigest() == provenance["machine_readable_source_sha256"]

