from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def _load():
    spec = importlib.util.spec_from_file_location(
        "gse_topology_examples_synthetic",
        ROOT / "tools/v3/publish_gse_topology_examples_v1.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_topology_examples_publish_all_fixed_worlds_methods_and_formats(tmp_path: Path, monkeypatch) -> None:
    publisher = _load()
    generator = tmp_path / "tools/v3/publish.py"
    generator.parent.mkdir(parents=True)
    generator.write_text("# synthetic generator\n", encoding="utf-8")
    monkeypatch.setattr(publisher, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(publisher, "__file__", str(generator))
    run = tmp_path / "results/gate4_topology/gate4_20260824_gse_offline_topology_validation_v1_seed0"
    (run / "metrics").mkdir(parents=True)
    state = {"state": "COMPLETED", "overall_status": publisher.EXPECTED_STATUS}
    runner = {"overall_status": publisher.EXPECTED_STATUS}
    overall = {
        "overall_status": publisher.EXPECTED_STATUS,
        "scientific_gate": {"passed": True},
        "strict_test_worlds_read": 0,
        "mtare_worlds_read": 0,
    }
    files = {
        run / "RUN_STATE.json": state,
        run / "metrics/summary.json": runner,
        run / "artifacts/offline_topology/summary.json": overall,
    }
    for path, value in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")
    source_paths = list(files)
    for world_index, world in enumerate(publisher.WORLDS):
        z = float(world_index)
        for method in publisher.METHODS:
            directory = run / f"artifacts/offline_topology/{method}/selected/seed0/{world}"
            directory.mkdir(parents=True)
            nodes = [
                {"id": 0, "xyz_m": [0.0, 0.0, z]},
                {"id": 1, "xyz_m": [10.0, 2.0, z + 1.0]},
            ]
            edges = [{"id": 0, "from": 0, "to": 1}]
            summary = {
                "association_mode": "learned" if method == "gse_learned_association" else "rule",
                "teacher_node_xyz_m": {"a": [0.0, 0.0, z], "b": [10.0, 2.0, z + 1.0]},
                "teacher_graph": {"node_ids": ["a", "b"], "edges": [["a", "b"]]},
                "predicted_to_teacher": {"0": "a", "1": "b"},
                "collapsed_graph": {"node_ids": [0, 1], "edges": [[0, 1]]},
            }
            node_path, edge_path, summary_path = directory / "nodes.jsonl", directory / "edges.jsonl", directory / "summary.json"
            node_path.write_text("".join(json.dumps(row) + "\n" for row in nodes), encoding="utf-8")
            edge_path.write_text("".join(json.dumps(row) + "\n" for row in edges), encoding="utf-8")
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            source_paths.extend((node_path, edge_path, summary_path))
    seal = run / "artifacts/evidence_sha256.txt"
    seal.write_text(
        "".join(
            f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(tmp_path)}\n"
            for path in source_paths
        ),
        encoding="utf-8",
    )
    destination = tmp_path / "docs/figures/gse_graph"
    result = publisher.publish(run, destination)
    assert result["published_files"] == 7
    for suffix in (".png", ".pdf", ".svg", ".csv", "_source.json", "_provenance.json", "_sha256.txt"):
        assert (destination / f"gse_topology_examples{suffix}").is_file()
    csv_rows = (destination / "gse_topology_examples.csv").read_text(encoding="utf-8").splitlines()
    assert len(csv_rows) == 1 + len(publisher.WORLDS) * (3 + len(publisher.METHODS) * 3)
