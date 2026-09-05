"""Real frozen-inference I/O pipeline, with only neural execution substituted."""
import hashlib
import importlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pytest
import torch
from torch import nn
import zarr

from mtare_topo.governance import build_run_id
from mtare_topo.governance_head_inference import validate_head_development_inference_card
from tests.v3.unit.test_gse_head_inference_card import card


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def _fixture(tmp_path):
    value = card()
    for task in value["tasks"]:
        rows = [row for row in value["selected_rows"] if row["task"] == task]
        common = dict(parent_id=task.split("__")[0], partition="fit", geometry_realization="c1_mixed")
        teacher = zarr.open_group(str(tmp_path / value["source_roots"]["teacher"] / (task + ".zarr")), mode="w")
        teacher.attrs.update(**common, student_identity_input_forbidden=True, window_frames=5)
        mask = np.arange(32)[None] < np.array([r["visible_fragments"] for r in rows])[:, None]
        for name, data in {
            "frame_row": np.array([r["frame_rows"] for r in rows], np.int64),
            "source_global_sequence_index": np.array([r["source_global_sequence_index"] for r in rows], np.int64),
            "primitive_mask": mask.astype(np.uint8),
            "axis_control_current_sensor_m": np.zeros((18, 32, 3, 3), np.float32),
            "relative_translation_current_sensor_m": np.zeros((18, 5, 3), np.float32),
            "relative_yaw_current_sensor_deg": np.zeros((18, 5), np.float32),
        }.items():
            teacher.create_dataset(name, data=data)
        teacher.create_dataset("undeclared_identity", data=np.full((18,), -99, np.int64))
        sensor = zarr.open_group(str(tmp_path / value["source_roots"]["sensor"] / (task + ".zarr")), mode="w")
        sensor.attrs.update(**common, sensor_shape=[16, 720], maximum_range_m=50., student_pose_input_forbidden=True)
        sensor.create_dataset("range_m", data=np.full((90, 16, 720), 5, np.float32), chunks=(5, 16, 720))
        sensor.create_dataset("valid_mask", data=np.ones((90, 16, 720), np.uint8), chunks=(5, 16, 720))
        sensor.create_dataset("undeclared_future_frame", data=np.full((90,), -99, np.int64))
    for kind, root in value["source_roots"].items():
        files = sorted(path for path in (tmp_path / root).rglob("*") if path.is_file())
        seal = tmp_path / value["source_seals"][kind]
        seal.write_text("".join(f"{_sha(path)}  {path.relative_to(tmp_path)}\n" for path in files)
                        + "0" * 64 + f"  {root}/S99_synthetic_C07__c1_mixed.zarr/frame_row/0.0\n"
                        + "0" * 64 + "  fixtures/raw_slot_offset.pt\n")
    _write(tmp_path / value["metadata_selection_path"], value["selected_rows"])
    for role, checkpoint in value["checkpoints"].items():
        path = tmp_path / checkpoint["path"]
        path.write_text(f"Synthetic {role} provenance placeholder: never deserialized")
        checkpoint["sha256"] = _sha(path)
    value["sealed_sources"] = {path: _sha(tmp_path / path) for path in value["sealed_sources"]}
    value["approval"]["checkpoint_sha256"] = {role: item["sha256"] for role, item in value["checkpoints"].items()}
    assert validate_head_development_inference_card(value).passed
    card_path = tmp_path / "configs/synthetic_inference_card.json"
    _write(card_path, value)
    source = tmp_path / "synthetic_source.txt"; source.write_text("Synthetic executable hash stand-in")
    spec = {"gate": 3, "date": "20260905", "slug": "synthetic_head_inference", "seed": 0,
            "operation": "data_export", "data_card": str(card_path.relative_to(tmp_path)),
            "source_sha256": {source.name: _sha(source)},
            "expected_versions": {"python": platform.python_version(), **{
                name: importlib.metadata.version(name) for name in ("numpy", "torch", "zarr", "scipy", "matplotlib")}}}
    spec_path = tmp_path / "configs/synthetic_inference_spec.json"
    _write(spec_path, spec)
    run = tmp_path / "results/gate3_semantics" / build_run_id(spec)
    for directory in ("config", "logs", "metrics", "previews", "artifacts"):
        (run / directory).mkdir(parents=True, exist_ok=True)
    _write(run / "config/run_spec.json", spec)
    _write(run / "config/data_card.json", value)
    _write(run / "RUN_STATE.json", {"state": "CREATED_NOT_EXECUTED"})
    return value, spec_path, run


def test_real_runner_reader_geometry_scoring_plots_ledger_and_seal(tmp_path, monkeypatch):
    value, spec_path, run = _fixture(tmp_path)
    immutable = {path: _sha(path) for path in tmp_path.rglob("*") if path.is_file() and not path.is_relative_to(run)}
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "tools/v3"))
    runner = importlib.import_module("run_gse_head_development_inference_v1")

    def contained(relative):
        assert not any(token in str(relative) for token in ("C07", "raw_slot_offset", "undeclared_identity", "undeclared_future_frame"))
        path = (tmp_path / relative).resolve()
        path.relative_to(tmp_path)
        return path

    pair = (nn.Linear(1, 1).eval().requires_grad_(False), nn.Linear(1, 1).eval().requires_grad_(False))
    calls = []

    def neural_only_stub(model, head, student, guard, repeat=False):
        assert model is pair[0] and head is pair[1] and len(student) == 18
        assert all(set(vars(row)) == {"range_valid", "relative_translation_current_sensor_m", "relative_yaw_current_sensor_deg"} for row in student)
        assert all(row.range_valid.shape == (5, 2, 16, 720) for row in student)
        guard(); calls.append(repeat)
        # Deterministic stand-in consumes only normalized student ranges, not
        # scoring targets. The synthetic scans have range 5 m, hence raw=0.
        observed = np.array([row.range_valid[0, 0, 0, 0] for row in student], np.float32)
        raw = np.broadcast_to((observed * np.float32(50) - np.float32(5))[:, None, None, None], (18, 32, 3, 3)).copy()
        return {"raw_no_offset": raw, "legacy_frozen": raw + np.float32(1)}

    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(runner, "contained", contained)
    monkeypatch.setattr(runner, "load_frozen_pair", lambda actual: pair)
    monkeypatch.setattr(runner, "infer_task", neural_only_stub)
    monkeypatch.setattr(sys, "argv", ["synthetic-inference", "--spec", str(spec_path), "--run-dir", str(run)])
    assert runner.main() == 0
    summary = json.loads((run / "metrics/summary.json").read_text())
    assert summary["status"] == "DEVELOPMENT_EVALUATED" and summary["error"] is None
    assert json.loads((run / "RUN_STATE.json").read_text())["state"] == "COMPLETED"
    assert calls == [True] + [False] * 9
    assert summary["ledger"] == {key: value["inference"][key] for key in summary["ledger"]}
    assert summary["ledger"]["optimizer_steps"] == summary["new_labels"] == 0
    result = summary["result"]
    assert (result["observations"], result["parents"], result["sensor_frames"], result["visible_fragments"]) == (180, 10, 900, 1489)
    assert result["frozen_states_unchanged"] and result["comparison"]["parents_raw_better"] == 10
    assert all(not parameter.requires_grad and parameter.grad is None for module in pair for parameter in module.parameters())
    assert summary["scientific_gate_pass"] is False
    for method in ("raw_no_offset", "legacy_frozen"):
        assert len(json.loads((run / f"artifacts/{method}_observation_metrics.json").read_text())) == 180
    for task in value["tasks"]:
        svg = run / f"previews/{task}.svg"
        assert ET.parse(svg).getroot().tag == "{http://www.w3.org/2000/svg}svg"
    assert len(list((run / "previews").glob("*.svg"))) == 11
    assert len((run / "logs/inference.jsonl").read_text().splitlines()) == 10
    opened = json.loads((run / "artifacts/source_reads_sha256.json").read_text())
    assert opened and all(not any(token in path for token in ("C07", "raw_slot_offset", "undeclared_")) for path in opened)
    with np.load(run / "artifacts/axis_predictions.npz", allow_pickle=False) as prediction:
        assert set(prediction.files) == set(value["output_fields"])
        assert all(prediction[key].shape == (180, 32, 3, 3) for key in prediction.files)
    seal = run / "artifacts/evidence_sha256.txt"
    entries = [line.split(None, 1) for line in seal.read_text().splitlines()]
    assert len(entries) == len([path for path in run.rglob("*") if path.is_file() and path != seal])
    assert all(_sha(tmp_path / path) == digest for digest, path in entries)
    assert all(_sha(path) == digest for path, digest in immutable.items())
    digest = _sha(seal)
    with pytest.raises(ValueError, match="exact fresh inference"):
        runner.main()
    assert _sha(seal) == digest
