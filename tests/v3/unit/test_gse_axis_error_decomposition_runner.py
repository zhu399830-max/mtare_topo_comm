"""Synthetic frozen-row parity, diagnostic aggregation and executor hooks."""
import copy
import importlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from mtare_topo.evaluation.gse_axis_error_decomposition import decompose_axes
from tests.v3.unit.test_gse_local_teacher_audit_runner import _fixture, _check_seal


def _module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "tools/v3"))
    return importlib.import_module("run_gse_axis_error_decomposition_v1")


def _row(index=0):
    return {
        "task": "S01_fixture_C01__c1_mixed", "row_index": index,
        "source_global_sequence_index": 100 + index,
        "target_node_scoring_only": "node_a", "degree": 1,
        "frame_rows": list(range(5 * index, 5 * index + 5)),
        "all_visible_construction_groups_scoring_only": [
            {"node_id_scoring_only": "node_a", "present_members": [{"teacher_slot_scoring_only": 0}]}
        ],
        "oracle_geometry_matches_not_detections": [{
            "prediction_slot": 5, "teacher_slot_scoring_only": 0, "reversed": False,
            "axis_control_point_mean_euclidean_m": 0., "existence_probability": .001,
            "half_axes_mae_m": .3,
        }],
    }


def _start(tmp_path, monkeypatch, previous=None, selected=None):
    module = _module(monkeypatch)
    monkeypatch.setattr(module.executor, "PROJECT_ROOT", tmp_path)
    previous = [_row()] if previous is None else previous
    selected = [_row()] if selected is None else selected
    path = tmp_path / "previous.json"
    path.write_text(json.dumps(previous))
    spec = {"diagnostic": module.KIND, "quadrature_samples": 256,
            "previous_observation_audit": "previous.json"}
    card = {"diagnostic": module.KIND, "new_scoring_thresholds": False,
            "selected_rows": [{k: r[k] for k in ("task", "row_index", "source_global_sequence_index")}
                              for r in selected]}
    calls = []
    def read_sealed(request, loader):
        assert request == path
        calls.append(request)
        return loader(request)
    diagnostic = module.AxisDiagnostic()
    diagnostic.start(spec, card, read_sealed)
    assert calls == [path]
    return diagnostic


def _geometry():
    teacher = np.zeros((1, 32, 3, 3), dtype=np.float32)
    teacher[0, 0, :, 0] = (-2, 0, 2)
    predicted = np.zeros_like(teacher)
    predicted[0, 5] = teacher[0, 0]
    return {"axis_control_current_sensor_m": teacher}, {"axis_control_current_sensor_m": predicted}


def test_start_reads_exact_allowed_previous_rows(tmp_path, monkeypatch):
    diagnostic = _start(tmp_path, monkeypatch)
    assert list(diagnostic.previous) == [(_row()["task"], 0)]


@pytest.mark.parametrize("case", ["duplicate", "missing", "extra", "different_task"])
def test_start_rejects_nonexact_previous_row_inventory(tmp_path, monkeypatch, case):
    previous = [_row()]
    if case == "duplicate":
        previous.append(_row())
    elif case == "missing":
        previous = []
    elif case == "extra":
        previous.append(_row(1))
    else:
        previous[0]["task"] = "S02_fixture_C01__c1_mixed"
    with pytest.raises(ValueError, match="selection drift"):
        _start(tmp_path, monkeypatch, previous=previous)


def test_task_adds_only_diagnostics_after_full_prior_dictionary_parity(tmp_path, monkeypatch):
    previous = _row()
    diagnostic = _start(tmp_path, monkeypatch, previous=[previous])
    rows = [copy.deepcopy(previous)]
    diagnostic.task(rows, *_geometry())
    metrics = rows[0].pop("axis_decomposition_fixed_assignment")
    assert rows[0] == previous
    assert len(metrics) == 1 and metrics[0]["point_mean_m"] == 0
    assert metrics[0]["target_incident_scoring_only"] is True
    assert metrics[0]["existence_probability_unchanged"] == .001


@pytest.mark.parametrize("field", ["source", "frame", "degree", "group", "existence", "score", "slot", "extra"])
def test_task_rejects_any_prior_row_metadata_or_score_drift(tmp_path, monkeypatch, field):
    diagnostic = _start(tmp_path, monkeypatch)
    row = _row()
    if field == "source":
        row["source_global_sequence_index"] += 1
    elif field == "frame":
        row["frame_rows"][0] += 1
    elif field == "degree":
        row["degree"] = 2
    elif field == "group":
        row["all_visible_construction_groups_scoring_only"][0]["node_id_scoring_only"] = "another"
    elif field == "existence":
        row["oracle_geometry_matches_not_detections"][0]["existence_probability"] = .9
    elif field == "score":
        row["oracle_geometry_matches_not_detections"][0]["half_axes_mae_m"] += 1e-7
    elif field == "slot":
        row["oracle_geometry_matches_not_detections"][0]["prediction_slot"] = 6
    else:
        row["new_before_parity"] = True
    with pytest.raises(ValueError, match="reproduce exactly"):
        diagnostic.task([row], *_geometry())
    assert "axis_decomposition_fixed_assignment" not in row


def test_task_rejects_geometry_change_even_when_previous_scores_are_unchanged(tmp_path, monkeypatch):
    diagnostic = _start(tmp_path, monkeypatch)
    teacher, prediction = _geometry()
    prediction["axis_control_current_sensor_m"][0, 5] += 1
    with pytest.raises(ValueError, match="source precision"):
        diagnostic.task([_row()], teacher, prediction)


def test_task_obeys_stored_reversal_not_a_new_best_fit(tmp_path, monkeypatch):
    previous = _row()
    previous["oracle_geometry_matches_not_detections"][0]["reversed"] = True
    diagnostic = _start(tmp_path, monkeypatch, previous=[previous])
    teacher, prediction = _geometry()
    prediction["axis_control_current_sensor_m"][0, 5] = teacher["axis_control_current_sensor_m"][0, 0, ::-1]
    rows = [copy.deepcopy(previous)]
    diagnostic.task(rows, teacher, prediction)
    assert rows[0]["axis_decomposition_fixed_assignment"][0]["point_mean_m"] == 0


def test_summarize_keeps_1452_matches_including_unknowns_and_adds_eleventh_preview(tmp_path, monkeypatch):
    module = _module(monkeypatch)
    (tmp_path / "artifacts").mkdir()
    (tmp_path / "previews").mkdir()
    teacher = np.array([[-2., 0., 0.], [0., 0., 0.], [2., 0., 0.]])
    resolved = decompose_axes(teacher + (1, 2, 0), teacher)
    unknown = decompose_axes(teacher, np.zeros((3, 3)))
    rows = []
    for row_index in range(180):
        family = row_index // 18 + 1
        task = f"S{family:02d}_fixture_C01__c1_mixed"
        matches = []
        for slot in range(9 if row_index < 12 else 8):
            metrics = unknown if (row_index, slot) == (0, 0) else resolved
            matches.append({**metrics, "target_incident_scoring_only": slot == 0})
        rows.append({"task": task, "axis_decomposition_fixed_assignment": matches})
    for family in range(1, 11):
        (tmp_path / "previews" / f"parent_{family}.svg").write_text('<svg xmlns="http://www.w3.org/2000/svg"/>')
    result = module.AxisDiagnostic().summarize(rows, tmp_path)
    assert result["all"]["matches"] == 1452
    assert result["target_incident"]["matches"] == 180
    assert result["other_visible_fragments"]["matches"] == 1272
    assert result["all"]["teacher_direction_unknown"] == 1
    assert result["all"]["metric_unknown_counts"]["axial_rms_m"] == 1
    assert result["all"]["metrics"]["point_mean_m"]["n"] == 1452
    assert result["all"]["metrics"]["axial_rms_m"]["n"] == 1451
    parents = json.loads((tmp_path / "artifacts/axis_decomposition_by_parent.json").read_text())
    assert len(parents) == 10 and sum(p["matches"] for p in parents) == 1452
    assert len(list((tmp_path / "previews").glob("*.svg"))) == 11
    tree = ET.parse(tmp_path / "previews/axis_error_decomposition.svg")
    circles = [n for n in tree.iter() if n.tag.endswith("}circle")]
    assert len(circles) == 1451
    assert sum(c.get("fill") == "#d76516" for c in circles) == 179
    assert sum(c.get("fill") == "#7b8793" for c in circles) == 1272
    texts = [n.text for n in tree.iter() if n.tag.endswith("}text")]
    assert "Plotted: 1451" in texts and "Unknown: 1" in texts
    with pytest.raises(ValueError, match="count drift"):
        module.AxisDiagnostic().summarize(rows[:-1], tmp_path)


class RecordingDiagnostic:
    def __init__(self, failure=None):
        self.calls, self.failure = [], failure

    def _record(self, stage, count):
        self.calls.append((stage, count))
        if stage == self.failure:
            raise ValueError("synthetic diagnostic hook failure " + stage)

    def start(self, spec, card, read_sealed):
        self._record("start", card["observation_count"])

    def task(self, rows, teacher, prediction):
        self._record("task", len(rows))

    def summarize(self, rows, run):
        self._record("summarize", len(rows))
        return {"synthetic_only_recorded_rows": len(rows)}


def test_executor_diagnostic_hook_calls_start_all_tasks_and_summary(tmp_path, monkeypatch):
    runner, run = _fixture(tmp_path, monkeypatch)
    recorder = RecordingDiagnostic()
    assert runner.main(diagnostic=recorder) == 0
    assert recorder.calls == [("start", 180)] + [("task", 18)] * 10 + [("summarize", 180)]
    output = json.loads((run / "metrics/summary.json").read_text())
    assert output["result"]["axis_error_decomposition"] == {"synthetic_only_recorded_rows": 180}
    _check_seal(runner, run, tmp_path)


@pytest.mark.parametrize("stage", ["start", "task", "summarize"])
def test_executor_diagnostic_hook_failures_are_sealed_and_never_retried(tmp_path, monkeypatch, stage):
    runner, run = _fixture(tmp_path, monkeypatch)
    assert runner.main(diagnostic=RecordingDiagnostic(failure=stage)) == 1
    output = json.loads((run / "metrics/summary.json").read_text())
    assert output["status"] == "AUDIT_FAIL" and stage in output["error"]
    assert "synthetic diagnostic hook failure" in (run / "logs/error.log").read_text()
    assert json.loads((run / "RUN_STATE.json").read_text())["state"] == "FAILED"
    _check_seal(runner, run, tmp_path)
    with pytest.raises(RuntimeError, match="no overwrite"):
        runner.main(diagnostic=RecordingDiagnostic())
    _check_seal(runner, run, tmp_path)
