"""Restore existing evidence on synthetic records only; never rerun training."""
import copy
import importlib
from pathlib import Path

import pytest
import torch

from mtare_topo.representation.gse_point_axis_probe import decide


def _module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "tools/v3"))
    return importlib.import_module("run_gse_point_axis_evidence_v1")


def _records():
    coordinate_rows, manifest = [], []
    for index in range(180):
        task = f"S{index // 18 + 1:02d}_fixture_C01__c1_mixed"
        local_row = index % 18
        count = 9 if index < 12 else 8
        manifest.append({"task": task, "row_index": local_row, "visible_fragments": count})
        controls = [{"teacher_slot_scoring_only": slot, "control_index": control,
                     "query_current_sensor_m": [float(slot), float(control), float(index)]}
                    for slot in range(1, count + 1) for control in range(3)]
        coordinate_rows.append({"task": task, "row_index": local_row,
                                "coordinate_support": {"control_points": controls}})
    return coordinate_rows, manifest


def test_restore_teacher_recovers_all4356_controls_and_sparse_slots_without_new_labels(monkeypatch):
    module = _module(monkeypatch)
    rows, manifest = _records()
    original = copy.deepcopy(rows)
    cache = module.restore_teacher(rows, manifest)
    assert rows == original
    assert len(cache) == 180
    assert sum(int(entry["mask"].sum()) for entry in cache) == 1452
    for index, entry in enumerate(cache):
        count = manifest[index]["visible_fragments"]
        assert entry["task"] == manifest[index]["task"]
        assert entry["target"].shape == (1, 32, 3, 3)
        assert entry["mask"].shape == (1, 32) and entry["mask"].dtype == torch.bool
        assert torch.equal(torch.nonzero(entry["mask"][0]).flatten(), torch.arange(1, count + 1))
        assert torch.isfinite(entry["target"]).all()
        assert not entry["target"][~entry["mask"]].any()
        for slot in range(1, count + 1):
            expected = torch.tensor([[slot, control, index] for control in range(3)], dtype=entry["target"].dtype)
            torch.testing.assert_close(entry["target"][0, slot], expected)
        assert "student" not in entry


def test_restore_teacher_uses_control_indices_not_list_position(monkeypatch):
    module = _module(monkeypatch)
    rows, manifest = _records()
    expected = module.restore_teacher(rows, manifest)
    for row in rows:
        row["coordinate_support"]["control_points"].reverse()
    actual = module.restore_teacher(rows, manifest)
    for first, second in zip(expected, actual):
        assert torch.equal(first["target"], second["target"])
        assert torch.equal(first["mask"], second["mask"])


@pytest.mark.parametrize("issue", ["duplicate_row", "missing_row", "wrong_order", "wrong_task",
                                  "missing_control", "duplicate_control", "slot_high", "slot_negative",
                                  "control_high", "control_negative", "nonfinite", "wrong_total",
                                  "manifest_slot_count"])
def test_restore_teacher_rejects_corrupted_or_nonexact_evidence(monkeypatch, issue):
    module = _module(monkeypatch)
    rows, manifest = _records()
    controls = rows[0]["coordinate_support"]["control_points"]
    if issue == "duplicate_row":
        rows[-1] = copy.deepcopy(rows[0])
    elif issue == "missing_row":
        rows.pop()
    elif issue == "wrong_order":
        rows[0], rows[1] = rows[1], rows[0]
    elif issue == "wrong_task":
        rows[0]["task"] = "S01_fixture_C02__c1_mixed"
    elif issue == "missing_control":
        controls.pop()
    elif issue == "duplicate_control":
        controls[-1] = copy.deepcopy(controls[-2])
    elif issue == "slot_high":
        controls[0]["teacher_slot_scoring_only"] = 32
    elif issue == "slot_negative":
        controls[0]["teacher_slot_scoring_only"] = -1
    elif issue == "control_high":
        controls[0]["control_index"] = 3
    elif issue == "control_negative":
        controls[0]["control_index"] = -1
    elif issue == "nonfinite":
        controls[0]["query_current_sensor_m"][0] = float("nan")
    elif issue == "wrong_total":
        del controls[-3:]
        manifest[0]["visible_fragments"] -= 1
    else:
        manifest[0]["visible_fragments"] -= 1
    with pytest.raises((ValueError, RuntimeError)):
        module.restore_teacher(rows, manifest)


def _evaluations(manifest):
    result = {}
    for name, error in (("initial", 5.), ("raw_no_offset", 1.8), ("raw_slot_offset", 2.8), ("legacy_frozen", 3.6)):
        result[name] = [{"coordinate_mae_m": error, "point_mean_euclidean_m": 2. * error,
                         "n_targets": m["visible_fragments"], "n_predictions": 32,
                         "unmatched_predictions": 32 - m["visible_fragments"]} for m in manifest]
    return result


def test_restore_result_keeps_parent_list_and_original_negative_fit_decision(monkeypatch):
    module = _module(monkeypatch)
    _, manifest = _records()
    evaluations = _evaluations(manifest)
    expected = decide(evaluations["initial"], evaluations["raw_no_offset"], evaluations["raw_slot_offset"],
                      evaluations["legacy_frozen"], [m["task"] for m in manifest])
    actual = module.restore_result(evaluations, manifest)
    assert isinstance(actual["parents"], list) and len(actual["parents"]) == 10
    assert actual["parents"] == expected["parents"]
    assert actual["parent_count"] == 10
    assert actual["decision"] == expected["decision"] == "STOP_BEFORE_EXPANSION"
    assert actual["checks"] == expected["checks"]
    assert actual["macro_observation_errors"] == expected["macro_observation_errors"]
    assert actual["scientific_gate_pass"] is False


@pytest.mark.parametrize("issue", ["missing_observation", "parent_imbalance", "nonfinite_metric"])
def test_restore_result_cannot_fix_population_or_change_invalid_metrics(monkeypatch, issue):
    module = _module(monkeypatch)
    _, manifest = _records()
    evaluations = _evaluations(manifest)
    if issue == "missing_observation":
        evaluations["raw_slot_offset"].pop()
    elif issue == "parent_imbalance":
        manifest[0]["task"] = manifest[-1]["task"]
    else:
        evaluations["legacy_frozen"][0]["coordinate_mae_m"] = float("inf")
    with pytest.raises(ValueError):
        module.restore_result(evaluations, manifest)
