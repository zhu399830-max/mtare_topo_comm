"""Synthetic deterministic registration/pooling to support audit integration."""
import copy
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mtare_topo.data.gse_scoped_model_input import ScopedModelInputs
from mtare_topo.data.primitive_relation_training import PrimitiveRelationStudentInput
from tests.v3.unit.test_gse_coordinate_audit_card import coordinate_card
from tests.v3.unit.test_governance import make_spec
from tests.v3.unit.test_gse_local_teacher_audit_runner import _fixture, _check_seal


def _module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "tools/v3"))
    return importlib.import_module("run_gse_coordinate_support_audit_v1")


def _task_case(tmp_path, monkeypatch, rows_count=1):
    module = _module(monkeypatch)
    diagnostic = module.CoordinateDiagnostic()
    students = []
    for _ in range(rows_count):
        value = np.zeros((5, 2, 16, 720), dtype=np.float32)
        # Two vertically separated returns in every selected azimuth bucket.
        # Registration and token pooling are REAL functions; no model is created.
        value[:, 0, [0, 15], :] = .2
        value[:, 1, [0, 15], :] = 1
        students.append(PrimitiveRelationStudentInput(value, np.zeros((5, 3), dtype=np.float32),
                                                     np.zeros(5, dtype=np.float32)))
    frame_rows = np.arange(rows_count * 5).reshape(rows_count, 5)
    batch = ScopedModelInputs(tuple(students), frame_rows, np.arange(rows_count) + 100)
    calls = []
    def read_task(task):
        assert task == "S01_fixture_C01__c1_mixed"
        calls.append(task)
        return batch
    diagnostic.reader = SimpleNamespace(read_task=read_task)
    diagnostic.sensor_frames_decoded = 0
    diagnostic.additional_reads = {}
    diagnostic.row_log = tmp_path / "row_log.jsonl"
    diagnostic.row_log.touch()
    rows = [{"task": "S01_fixture_C01__c1_mixed", "row_index": i, "unchanged_score": .25}
            for i in range(rows_count)]
    diagnostic.previous = {(r["task"], r["row_index"]): copy.deepcopy(r) for r in rows}
    axis = np.zeros((rows_count, 32, 3, 3), dtype=np.float32)
    axis[:, 5, :, 0] = (-2, 0, 2)
    mask = np.zeros((rows_count, 32), dtype=np.uint8)
    mask[:, 5] = 1
    teacher = {"frame_row": frame_rows.copy(), "primitive_mask": mask,
               "axis_control_current_sensor_m": axis}
    return diagnostic, rows, teacher, calls


@pytest.mark.parametrize("drift", ["duplicate", "wrong_selection"])
def test_start_rejects_nonexact_prior_row_list_before_log_creation(tmp_path, monkeypatch, drift):
    module = _module(monkeypatch)
    monkeypatch.setattr(module.executor, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(module, "ScopedCompositionModelReader", lambda *args, **kwargs: SimpleNamespace(opened={}))
    card = coordinate_card()
    card["diagnostic"] = module.KIND
    previous = copy.deepcopy(card["selected_rows"])
    if drift == "duplicate":
        previous.append(copy.deepcopy(previous[0]))
    else:
        previous[0]["row_index"] = 999
    (tmp_path / "previous.json").write_text(json.dumps(previous))
    spec = make_spec(3, "audit")
    spec.update({"diagnostic": module.KIND, "wall_time_cap_s": 600,
                 "expected_scipy_version": module.scipy.__version__, "source_seals": [],
                 "sensor_root": "unused_sensor", "teacher_root": "unused_teacher",
                 "previous_observation_audit": "previous.json"})
    run = tmp_path / "results/gate3_semantics" / module.build_run_id(spec)
    (run / "logs").mkdir(parents=True)
    with pytest.raises(ValueError, match="inventory|selection|previous"):
        module.CoordinateDiagnostic().start(spec, card, lambda path, loader: loader(path))
    assert not (run / "logs/coordinate_support.jsonl").exists()


def test_task_real_registration_pooling_bounds_preserves_three_controls_and_witnesses(tmp_path, monkeypatch):
    diagnostic, rows, teacher, calls = _task_case(tmp_path, monkeypatch)
    prior = copy.deepcopy(rows)
    diagnostic.task(rows, teacher, {})
    assert calls == ["S01_fixture_C01__c1_mixed"]
    assert diagnostic.sensor_frames_decoded == 5
    result = rows[0].pop("coordinate_support")
    assert rows == prior
    points = result["control_points"]
    assert len(points) == 3
    assert [p["teacher_slot_scoring_only"] for p in points] == [5, 5, 5]
    assert [p["control_index"] for p in points] == [0, 1, 2]
    np.testing.assert_array_equal([p["query_current_sensor_m"] for p in points],
                                  teacher["axis_control_current_sensor_m"][0, 5])
    for pool in ("mean", "raw"):
        bounds = result[pool + "_support"]
        assert len(bounds["witnesses"]) == len(bounds["lower_bound_m"]) == len(bounds["upper_bound_m"]) == 3
        for index, point in enumerate(points):
            assert point[pool + "_status"] == bounds["status"][index]
            assert point[pool + "_lower_bound_m"] == bounds["lower_bound_m"][index]
            assert point[pool + "_upper_bound_m"] == bounds["upper_bound_m"][index]
    progress = [json.loads(line) for line in diagnostic.row_log.read_text().splitlines()]
    assert len(progress) == 1 and progress[0]["controls"] == 3
    assert progress[0]["raw_points"] == 5 * 2 * 720
    assert progress[0]["mean_points"] == 5 * 180


def test_task_rejects_sensor_teacher_frame_order_drift_before_geometry(tmp_path, monkeypatch):
    diagnostic, rows, teacher, _ = _task_case(tmp_path, monkeypatch)
    teacher["frame_row"] = teacher["frame_row"][:, ::-1]
    with pytest.raises(ValueError, match="frame-order drift"):
        diagnostic.task(rows, teacher, {})
    assert diagnostic.sensor_frames_decoded == 0
    assert not diagnostic.row_log.read_text()
    assert "coordinate_support" not in rows[0]


def test_task_rejects_previous_row_changes_without_silently_new_scoring(tmp_path, monkeypatch):
    diagnostic, rows, teacher, _ = _task_case(tmp_path, monkeypatch)
    rows[0]["unchanged_score"] += .001
    with pytest.raises(ValueError, match="exactly reproduced"):
        diagnostic.task(rows, teacher, {})
    assert not diagnostic.row_log.read_text()
    assert "coordinate_support" not in rows[0]


def _summary_rows():
    rows = []
    for index in range(180):
        controls = []
        for slot in range(9 if index < 12 else 8):
            for control in range(3):
                witness = slot == 0 and control == 0
                controls.append({"mean_status": "OUTSIDE_CERTIFIED" if witness else "WITHIN_TOLERANCE_WITNESS",
                                 "raw_status": "WITHIN_TOLERANCE_WITNESS",
                                 "mean_lower_bound_m": 1. if witness else 0.,
                                 "mean_upper_bound_m": 2. if witness else 0.,
                                 "raw_lower_bound_m": 0., "raw_upper_bound_m": 0.,
                                 "pooling_exclusion_with_raw_witness": witness})
        rows.append({"task": f"S{index // 18 + 1:02d}_fixture_C01__c1_mixed",
                     "coordinate_support": {"control_points": controls}})
    return rows


def test_summary_exact4356_controls_900_frames_preserves_parent_population(tmp_path, monkeypatch):
    module = _module(monkeypatch)
    diagnostic = module.CoordinateDiagnostic()
    diagnostic.sensor_frames_decoded = 900
    diagnostic.additional_reads = {"synthetic": "digest"}
    (tmp_path / "artifacts").mkdir()
    (tmp_path / "previews").mkdir()
    result = diagnostic.summarize(_summary_rows(), tmp_path)
    assert result["observations"] == 180 and result["control_points"] == 4356
    assert result["sensor_frames_decoded"] == 900
    assert result["mean_excluded_raw_witness"] == 180
    assert result["both_witness"] == 4176
    assert result["no_model_or_checkpoint"] is True
    assert result["no_optimizer_or_new_targets"] is True
    parents = json.loads((tmp_path / "artifacts/coordinate_support_by_parent.json").read_text())
    assert len(parents) == 10 and sum(p["control_points"] for p in parents) == 4356
    assert (tmp_path / "previews/coordinate_support_by_parent.svg").is_file()


@pytest.mark.parametrize("kind", ["controls", "frames"])
def test_summary_rejects_incomplete_population(tmp_path, monkeypatch, kind):
    diagnostic = _module(monkeypatch).CoordinateDiagnostic()
    diagnostic.sensor_frames_decoded = 899 if kind == "frames" else 900
    diagnostic.additional_reads = {}
    rows = _summary_rows()
    if kind == "controls":
        rows[-1]["coordinate_support"]["control_points"].pop()
    with pytest.raises(ValueError, match="population drift"):
        diagnostic.summarize(rows, tmp_path)


def test_executor_coordinate_hook_reports_sensor_reads_and_selects_600_second_cap(tmp_path, monkeypatch):
    runner, run = _fixture(tmp_path, monkeypatch)
    original_card = json.loads((tmp_path / "card.json").read_text())
    value = coordinate_card()
    for key in ("sealed_sources", "existing_prediction_cache_read_only", "allowed_teacher_fields", "allowed_cache_fields"):
        value[key] = original_card[key]
    extra = tmp_path / "synthetic_sensor_read.json"
    extra.write_text('{"synthetic": true}')
    seal = tmp_path / "source_seal.txt"
    seal.write_text(seal.read_text() + f"{runner.sha(extra)}  {extra.name}\n")
    value["sealed_sources"]["source_seal.txt"] = runner.sha(seal)
    (tmp_path / "card.json").write_text(json.dumps(value))
    (run / "config/data_card.json").write_text(json.dumps(value))
    spec = json.loads((tmp_path / "spec.json").read_text())
    spec["wall_time_cap_s"] = 600
    (tmp_path / "spec.json").write_text(json.dumps(spec))
    (run / "config/run_spec.json").write_text(json.dumps(spec))
    alarms = []
    monkeypatch.setattr(runner.signal, "alarm", lambda seconds: alarms.append(seconds) or 0)

    class RecordingCoordinateDiagnostic:
        coordinate_audit = True
        summary_key = "coordinate_support_expressivity"
        def start(self, spec, card, read_sealed):
            assert read_sealed(extra, lambda p: json.loads(p.read_text())) == {"synthetic": True}
            self.additional_reads = {str(extra): runner.sha(extra)}
            self.sensor_frames_decoded = 0
        def task(self, rows, teacher, prediction):
            self.sensor_frames_decoded += len(np.unique(teacher["frame_row"]))
        def summarize(self, rows, run):
            return {"synthetic_hook_only": True, "observations": len(rows)}

    assert runner.main(diagnostic=RecordingCoordinateDiagnostic()) == 0
    summary = json.loads((run / "metrics/summary.json").read_text())
    assert summary["result"]["sensor_frames_decoded"] == 900
    assert summary["result"]["model_inference"] == 0
    assert summary["result"]["new_targets"] == 0
    assert summary["result"]["coordinate_support_expressivity"]["observations"] == 180
    reads = json.loads((run / "artifacts/source_reads_sha256.json").read_text())
    assert reads[extra.name] == runner.sha(extra)
    assert alarms == [600, 0]
    _check_seal(runner, run, tmp_path)
