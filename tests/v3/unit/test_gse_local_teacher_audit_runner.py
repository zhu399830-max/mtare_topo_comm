"""Entire audit executor on temporary, synthetic, sealed 180-row inputs only."""
import importlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pytest
import torch
import zarr

from mtare_topo.data.gse_local_teacher_audit import FIELDS, PREDICTION_FIELDS
from mtare_topo.governance import create_run
from tests.v3.unit.test_governance import make_spec, make_status
from tests.v3.unit.test_gse_scoped_inventory import card


def _fixture(tmp_path, monkeypatch, *, bad_order=False, duplicate_manifest=False):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "tools/v3"))
    runner = importlib.import_module("run_gse_local_teacher_audit_v1")
    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)
    construction_root = tmp_path / "constructions"
    prediction_root = tmp_path / "predictions"
    construction_root.mkdir()
    prediction_root.mkdir()
    records, manifests = [], []
    for family in range(1, 11):
        task = f"S{family:02d}_fixture_C01__c1_mixed"
        parent = task.split("__")[0]
        group = zarr.open_group(str(tmp_path / "teachers" / (task + ".zarr")), mode="w")
        group.attrs.update({"parent_id": parent, "partition": "fit", "geometry_realization": "c1_mixed",
                            "student_identity_input_forbidden": True})
        indices = np.full((18, 32), -1, dtype=np.int32)
        indices[:, 0] = np.arange(18) % 10
        temporal = np.zeros((18, 5, 32), dtype=np.uint8)
        temporal[:, 0, 0] = 1
        axis = np.zeros((18, 32, 3, 3), dtype=np.float32)
        axis[:, 0, :, 0] = (0, 1, 2)
        arrays = {
            "frame_row": np.arange(90, dtype=np.int64).reshape(18, 5),
            "source_global_sequence_index": family * 100 + np.arange(18, dtype=np.int64),
            "primitive_index": indices,
            "primitive_mask": (indices >= 0).astype(np.uint8),
            "axis_control_current_sensor_m": axis,
            "endpoint_half_axes_m": np.ones((18, 32, 2, 2), dtype=np.float32),
            "endpoint_shape_exponent": np.ones((18, 32, 2), dtype=np.float32),
            "support_ray_count": (indices >= 0).astype(np.int32),
            "temporal_visibility": temporal,
            "endpoint_neighbor": np.full((18, 32, 2, 3), -1, dtype=np.int8),
            "disconnected_overlap_packed": np.zeros((18, 32, 4), dtype=np.uint8),
        }
        assert set(arrays) == FIELDS
        for name, data in arrays.items():
            group.create_dataset(name, data=data)
        prediction_axis = np.full((18, 32, 3, 3), 100, dtype=np.float32)
        prediction_axis[:, 0] = axis[:, 0]
        prediction = {
            "axis_control_current_sensor_m": prediction_axis,
            "endpoint_half_axes_m": arrays["endpoint_half_axes_m"].copy(),
            "endpoint_shape_exponent": arrays["endpoint_shape_exponent"].copy(),
            "existence_logits": np.full((18, 32), -100, dtype=np.float32),
            "geometry_uncertainty": np.ones((18, 32), dtype=np.float32),
            "endpoint_evidence_logits": np.full((18, 32, 2), -100, dtype=np.float32),
        }
        assert set(prediction) == set(PREDICTION_FIELDS)
        np.savez(prediction_root / (task + ".npz"), **prediction)
        manifests.append({"task": task,
                          "source_sequence_indices": arrays["source_global_sequence_index"].tolist(),
                          "frame_rows": arrays["frame_row"].tolist()})
        construction = {
            "parent_id": parent, "geometry_realization": "c1_mixed",
            "realized_primitives": [{"primitive_id": str(i)} for i in range(10)],
            "base_construction": {"composition_operations": [
                {"node_id": str(i), "degree": 1,
                 "member_endpoints": [{"primitive_id": str(i), "endpoint_index": 0}]}
                for i in range(10)]},
        }
        (construction_root / (task + ".json")).write_text(json.dumps(construction))
        for row in range(18):
            records.append({"task": task, "row_index": row,
                            "source_global_sequence_index": family * 100 + row,
                            "target_node_scoring_only": str(row % 10), "degree": 1})
    if bad_order:
        manifests[0]["frame_rows"] = manifests[0]["frame_rows"][::-1]
    if duplicate_manifest:
        manifests.append(dict(manifests[0]))
    (tmp_path / "selection.json").write_text(json.dumps(records))
    (tmp_path / "prediction_manifest.json").write_text(json.dumps(manifests))
    files = sorted(p for p in tmp_path.rglob("*") if p.is_file())
    (tmp_path / "source_seal.txt").write_text("".join(
        f"{runner.sha(p)}  {p.relative_to(tmp_path)}\n" for p in files))
    value = card()
    value.update({
        "selected_rows": [{k: r[k] for k in ("task", "row_index", "source_global_sequence_index")}
                          for r in records],
        "observation_count": 180, "worlds": sorted({r["task"].split("__")[0] for r in records}),
        "sealed_sources": {"selection.json": runner.sha(tmp_path / "selection.json"),
                           "source_seal.txt": runner.sha(tmp_path / "source_seal.txt")},
        "existing_prediction_cache_read_only": True,
        "allowed_teacher_fields": sorted(FIELDS), "allowed_cache_fields": list(PREDICTION_FIELDS),
    })
    (tmp_path / "card.json").write_text(json.dumps(value))
    spec = make_spec(3, "audit")
    spec.update({
        "data_card": "card.json", "selection_manifest": "selection.json", "source_sha256": {},
        "source_seals": ["source_seal.txt"], "teacher_root": "teachers",
        "construction_root": "constructions", "prediction_root": "predictions",
        "prediction_manifest": "prediction_manifest.json",
        "expected_versions": {"python": runner.platform.python_version(), "numpy": np.__version__,
                              "torch": torch.__version__, "zarr": zarr.__version__},
    })
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec))
    run = create_run(spec, make_status(3), tmp_path)
    monkeypatch.setattr(sys, "argv", ["runner", "--spec", str(path), "--run-dir", str(run)])
    return runner, run


def _check_seal(runner, run, root):
    seal = run / "artifacts/evidence_sha256.txt"
    covered = set()
    for line in seal.read_text().splitlines():
        digest, relative = line.split(None, 1)
        path = root / relative
        assert runner.sha(path) == digest
        covered.add(path)
    assert covered == {p for p in run.rglob("*") if p.is_file() and p != seal}


def test_executor_180_sealed_rows_previews_no_inference_targets_or_retry(tmp_path, monkeypatch):
    runner, run = _fixture(tmp_path, monkeypatch)
    assert runner.main() == 0
    summary = json.loads((run / "metrics/summary.json").read_text())
    assert summary["status"] == "AUDIT_COMPLETE" and summary["error"] is None
    assert summary["scientific_gate_pass"] is False
    result = summary["result"]
    assert result["observations"] == 180 and result["parents"] == 10
    assert result["unique_node_identities_scoring_only"] == 100
    assert result["unique_source_frames_referenced_not_decoded"] == 900
    assert result["visible_fragments"] == 180
    for key in ("sensor_frames_decoded", "model_inference", "optimizer_steps", "new_targets"):
        assert result[key] == 0
    assert result["training_ready"] is False
    assert result["construction_relations_exact"] is True
    assert result["cached_geometry_error_oracle_alignment"]["axis_control_point_mean_euclidean_m"]["max"] == 0
    assert result["cached_geometry_error_oracle_alignment"]["existence_probability"]["max"] < 1e-30
    rows = json.loads((run / "artifacts/observation_audit.json").read_text())
    assert len(rows) == 180 and not any(r["training_target_generated"] for r in rows)
    previews = sorted((run / "previews").glob("*.svg"))
    assert len(previews) == 10
    for preview in previews:
        tree = ET.parse(preview)
        labels = [node.text for node in tree.iter() if node.tag.endswith("}text")]
        assert sum(" XY |" in (text or "") for text in labels) == 18
        assert sum(" XZ |" in (text or "") for text in labels) == 18
    _check_seal(runner, run, tmp_path)
    before = runner.sha(run / "artifacts/evidence_sha256.txt")
    with pytest.raises(RuntimeError, match="no overwrite"):
        runner.main()
    assert runner.sha(run / "artifacts/evidence_sha256.txt") == before
    _check_seal(runner, run, tmp_path)


def test_executor_rejects_sealed_but_misordered_cached_prediction_manifest(tmp_path, monkeypatch):
    runner, run = _fixture(tmp_path, monkeypatch, bad_order=True)
    assert runner.main() == 1
    output = json.loads((run / "metrics/summary.json").read_text())
    assert output["status"] == "AUDIT_FAIL"
    assert "ordering drift" in output["error"]
    assert json.loads((run / "RUN_STATE.json").read_text())["state"] == "FAILED"
    assert not (run / "artifacts/observation_audit.json").exists()
    _check_seal(runner, run, tmp_path)


def test_executor_rejects_duplicate_task_prediction_manifest(tmp_path, monkeypatch):
    runner, run = _fixture(tmp_path, monkeypatch, duplicate_manifest=True)
    assert runner.main() == 1
    output = json.loads((run / "metrics/summary.json").read_text())
    assert output["status"] == "AUDIT_FAIL"
    assert "manifest" in output["error"].lower()
    _check_seal(runner, run, tmp_path)
