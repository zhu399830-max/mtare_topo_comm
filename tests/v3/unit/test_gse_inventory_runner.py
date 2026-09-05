"""Exercise the entire immutable inventory executor on 180 synthetic rows."""
import importlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import zarr

from mtare_topo.governance import create_run
from tests.v3.unit.test_governance import make_spec, make_status
from tests.v3.unit.test_gse_scoped_inventory import card


def test_executor_list_manifest_through_seal(tmp_path, monkeypatch):
    tools = Path(__file__).resolve().parents[3] / "tools/v3"
    monkeypatch.syspath_prepend(str(tools))
    runner = importlib.import_module("run_gse_composition_inventory_v1")
    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)
    records, traversals = [], []
    construction_root = tmp_path / "constructions"; construction_root.mkdir()
    for family in range(1, 11):
        task = f"S{family:02d}_fixture_C01__c1_mixed"
        parent = task.split("__")[0]
        group = zarr.open_group(str(tmp_path / "teachers" / (task + ".zarr")), mode="w")
        group.attrs.update({"parent_id": parent, "partition": "fit", "geometry_realization": "c1_mixed",
                            "student_identity_input_forbidden": True})
        indices = np.full((18, 32), -1, dtype="i4"); indices[:, 0] = np.arange(18) % 10
        arrays = {"frame_row": np.arange(90).reshape(18, 5), "source_global_sequence_index": family * 100 + np.arange(18),
                  "primitive_index": indices, "primitive_mask": (indices >= 0).astype("u1"),
                  "support_ray_count": (indices >= 0).astype("i4"), "temporal_visibility": np.zeros((18, 5, 32), dtype="u1"),
                  "relative_translation_current_sensor_m": np.zeros((18, 5, 3), dtype="f4")}
        for name, data in arrays.items():
            group.create_dataset(name, data=data)
        side = zarr.open_group(str(tmp_path / "sidecars" / (task + ".zarr")), mode="w")
        side.attrs.update({"schema_version": "primitive_attachment_observability_sidecar_v1",
                           "hidden_pair_semantics": "unknown_never_negative", "support_band_m": .25})
        packed = np.zeros((18, 8), dtype="u1"); packed[:, 0] = 1
        side.create_dataset("endpoint_observed_packed", data=packed)
        side.create_dataset("source_global_sequence_index", data=arrays["source_global_sequence_index"])
        construction = {"parent_id": parent, "geometry_realization": "c1_mixed",
                        "realized_primitives": [{"primitive_id": str(i)} for i in range(10)],
                        "base_construction": {"composition_operations": [{"node_id": str(i), "degree": 1,
                        "member_endpoints": [{"primitive_id": str(i), "endpoint_index": 0}]} for i in range(10)]}}
        (construction_root / (task + ".json")).write_text(json.dumps(construction))
        for i in range(18):
            traversal = f"{parent}:trace{i}"
            records.append({"task": task, "row_index": i, "source_global_sequence_index": family * 100 + i,
                            "target_node_scoring_only": str(i % 10), "degree": 1, "traversal": traversal})
            traversals.append({"traversal_id": traversal, "to_node_id": str(i % 10), "parent_id": parent, "length_m": 5.})
    (tmp_path / "manifest.json").write_text(json.dumps(records))
    (tmp_path / "traversals.jsonl").write_text("".join(json.dumps(r) + "\n" for r in traversals))
    files = sorted(p for p in tmp_path.rglob("*") if p.is_file())
    (tmp_path / "source_seal.txt").write_text("".join(f"{runner.sha(p)}  {p.relative_to(tmp_path)}\n" for p in files))
    value = card()
    value.update({"selected_rows": [{k: r[k] for k in ("task", "row_index", "source_global_sequence_index")} for r in records],
                  "observation_count": 180, "worlds": sorted({r["task"].split("__")[0] for r in records}),
                  "sealed_sources": {"manifest.json": runner.sha(tmp_path / "manifest.json"),
                                     "source_seal.txt": runner.sha(tmp_path / "source_seal.txt")}})
    (tmp_path / "card.json").write_text(json.dumps(value))
    spec = make_spec(3, "audit")
    spec.update({"data_card": "card.json", "selection_manifest": "manifest.json", "source_sha256": {},
                 "source_seals": ["source_seal.txt"], "teacher_root": "teachers", "sidecar_root": "sidecars",
                 "construction_root": "constructions", "traversal_manifest": "traversals.jsonl",
                 "expected_versions": {"python": runner.platform.python_version(), "numpy": np.__version__, "zarr": zarr.__version__}})
    path = tmp_path / "spec.json"; path.write_text(json.dumps(spec))
    run = create_run(spec, make_status(3), tmp_path)
    monkeypatch.setattr(sys, "argv", ["runner", "--spec", str(path), "--run-dir", str(run)])
    assert runner.main() == 0
    output = json.loads((run / "metrics/summary.json").read_text())
    assert output["result"]["observations"] == 180
    assert output["result"]["unique_node_identities"] == 100
    assert output["result"]["unique_source_frames"] == 900
    assert output["result"]["complete_incident_support_observations"] == 180
    for line in (run / "artifacts/evidence_sha256.txt").read_text().splitlines():
        digest, relative = line.split(None, 1)
        assert runner.sha(tmp_path / relative) == digest
    before = runner.sha(run / "artifacts/evidence_sha256.txt")
    with pytest.raises(RuntimeError, match="no overwrite"):
        runner.main()
    assert runner.sha(run / "artifacts/evidence_sha256.txt") == before
