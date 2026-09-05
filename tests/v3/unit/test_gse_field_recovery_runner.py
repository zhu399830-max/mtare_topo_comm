import importlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mtare_topo.governance import preflight
from tests.v3.unit.test_governance import make_spec, make_status
from tests.v3.unit.test_gse_field_recovery_card import card


@pytest.fixture
def runner(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "tools/v3"))
    return importlib.import_module("run_gse_composition_field_recovery_v1")


def population():
    rows = card()["selected_rows"]
    cache = {"endpoint_features": np.zeros((180, 64, 44), dtype="f4"),
             "endpoint_confidence": np.zeros((180, 64), dtype="f4"),
             "source_global_sequence_index": np.arange(180)}
    class Reader:
        selection = sorted({r["task"] for r in rows})
        def read_task(self, task):
            offset = self.selection.index(task) * 18
            return SimpleNamespace(student=[None] * 18, frame_rows=np.arange(90).reshape(18, 5),
                                   source_sequence_indices=np.arange(offset, offset + 18))
    return rows, cache, Reader()


def fake_prediction(runner):
    return ({k: np.zeros((18, *shape), dtype="f4") for k, shape in runner.FIELDS.items()},
            np.zeros((18, 64, 44), dtype="f4"), np.zeros((18, 64), dtype="f4"))


def test_all_180_exports_and_18_repeat_with_exact_cache(runner, tmp_path, monkeypatch):
    rows, cache, reader = population()
    calls = []
    def predict(model, student):
        calls.append(len(student)); return fake_prediction(runner)
    monkeypatch.setattr(runner, "predict", predict)
    result = runner.recover(None, reader, rows, cache, tmp_path, io.StringIO(), repeat_task=reader.selection[0])
    assert result["total_inference_observations"] == sum(calls) == 198
    assert result["unique_sensor_frames"] == 900
    files = list(tmp_path.glob("*.npz")); assert len(files) == 10
    with np.load(files[0], allow_pickle=False) as archive:
        assert set(archive.files) == set(runner.FIELDS)
    assert result["new_teacher_labels"] == result["optimizer_steps"] == 0


def test_cache_mismatch_stops_before_task_asset(runner, tmp_path, monkeypatch):
    rows, cache, reader = population()
    cache["endpoint_confidence"][0, 0] = 1e-6
    monkeypatch.setattr(runner, "predict", lambda *args: fake_prediction(runner))
    with pytest.raises(ValueError, match="parity"):
        runner.recover(None, reader, rows, cache, tmp_path, io.StringIO(), repeat_task=reader.selection[0])
    assert not list(tmp_path.iterdir())


def test_repeat_field_drift_stops_before_task_asset(runner, tmp_path, monkeypatch):
    rows, cache, reader = population()
    calls = 0
    def predict(*args):
        nonlocal calls
        calls += 1
        result = fake_prediction(runner)
        if calls == 2:
            result[0]["existence_logits"][0, 0] = 1e-6
        return result
    monkeypatch.setattr(runner, "predict", predict)
    with pytest.raises(ValueError, match="deterministic"):
        runner.recover(None, reader, rows, cache, tmp_path, io.StringIO(), repeat_task=reader.selection[0])
    assert not list(tmp_path.iterdir())


def test_real_model_field_interface_cpu_synthetic_only(runner):
    from mtare_topo.data.primitive_relation_training import PrimitiveRelationStudentInput
    torch.manual_seed(0)
    model = runner.ObservableSparsePortRelationNet().eval()
    for p in model.parameters(): p.requires_grad_(False)
    student = PrimitiveRelationStudentInput(np.ones((5, 2, 16, 720), dtype="f4"),
                                            np.zeros((5, 3), dtype="f4"), np.zeros(5, dtype="f4"))
    before = runner.state_sha(model)
    raw, feature, confidence = runner.predict(model, [student])
    assert set(raw) == set(runner.FIELDS)
    assert feature.shape == (1, 64, 44) and confidence.shape == (1, 64)
    assert runner.state_sha(model) == before
    assert all(p.grad is None for p in model.parameters())


def test_new_card_preflight_export_only(tmp_path):
    value = card()
    (tmp_path / "card.json").write_text(json.dumps(value))
    spec = make_spec(3, "data_export"); spec["data_card"] = "card.json"
    assert preflight(spec, make_status(3), tmp_path).passed
    spec["operation"] = "training"
    assert not preflight(spec, make_status(3), tmp_path).passed
