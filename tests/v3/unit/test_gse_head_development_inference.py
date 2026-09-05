"""Synthetic frozen-head interfaces, scores and figures; no trained assets."""
import copy
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np
import pytest
import torch
from torch import nn
import zarr

from mtare_topo.evaluation.gse_head_development_inference import (
    GEOMETRY_FIELDS, evaluate_outputs, infer_task, plot_development, read_geometry,
)


def _geometry(tmp_path):
    task = "S01_synthetic_C02__c1_mixed"
    group = zarr.open_group(str(tmp_path / "synthetic.zarr"), mode="w")
    targets = np.zeros((18, 32, 3, 3), np.float32)
    mask = np.zeros((18, 32), np.uint8); mask[:, :8] = 1
    frames = np.arange(90).reshape(18, 5)
    source = np.arange(100, 118)
    for name, value in (("axis_control_current_sensor_m", targets), ("primitive_mask", mask),
                        ("frame_row", frames), ("source_global_sequence_index", source)):
        group.create_dataset(name, data=value)
    records = [{"task": task, "row_index": i, "frame_rows": frames[i].tolist(),
                "source_global_sequence_index": int(source[i]), "visible_fragments": 8} for i in range(18)]
    batch = SimpleNamespace(frame_rows=frames.copy(), source_sequence_indices=source.copy())

    def open_group(root, selected_task, fields):
        assert root == tmp_path and selected_task == task
        assert fields == GEOMETRY_FIELDS
        return group

    reader = SimpleNamespace(teacher_root=tmp_path, _open=open_group)
    return reader, task, batch, records, group


def test_geometry_reads_exact18_targets_and_metadata_parity(tmp_path):
    reader, task, batch, records, group = _geometry(tmp_path)
    group["axis_control_current_sensor_m"][:, 8:] = np.nan
    targets, masks = read_geometry(reader, task, batch, records)
    assert targets.shape == (18, 32, 3, 3) and masks.dtype == bool
    assert masks.sum() == 144 and np.isnan(targets[:, 8:]).all()


@pytest.mark.parametrize("issue", ["17rows", "frames", "source", "fragments", "mask", "active_nan", "shape"])
def test_geometry_population_drift_rejected(tmp_path, issue):
    reader, task, batch, records, group = _geometry(tmp_path)
    if issue == "17rows": records.pop()
    elif issue == "frames": batch.frame_rows[0, 0] += 1
    elif issue == "source": batch.source_sequence_indices[0] += 1
    elif issue == "fragments": records[0]["visible_fragments"] += 1
    elif issue == "mask": group["primitive_mask"][0, 0] = 2
    elif issue == "active_nan": group["axis_control_current_sensor_m"][0, 0] = np.nan
    else:
        del group["axis_control_current_sensor_m"]
        group.create_dataset("axis_control_current_sensor_m", data=np.zeros((18, 31, 3, 3), np.float32))
    with pytest.raises(ValueError):
        read_geometry(reader, task, batch, records)


def _outputs():
    target = np.zeros((180, 32, 3, 3), np.float32)
    masks = np.arange(32)[None] < (8 + (np.arange(180) < 49))[:, None]
    records = [{"task": f"S{i // 18 + 1:02d}_synthetic_C02__c1_mixed", "row_index": i % 18} for i in range(180)]
    axes = {"raw_no_offset": target.copy(), "legacy_frozen": target.copy() + 1}
    return axes, target, masks, records


def test_exact180_two_methods_unknown_directions_and1489_targets_retained():
    axes, target, masks, records = _outputs()
    details, result = evaluate_outputs(axes, target, masks, records)
    assert masks.sum() == 1489
    for method in axes:
        assert len(details[method]) == 180
        macro = result[method]["macro"]
        assert macro["matched_targets"] == 1489
        assert macro["surplus_queries"] == 5760 - 1489
        assert macro["layout"]["undirected_direction_error_deg"]["mean"] is None
        assert macro["layout"]["undirected_direction_error_deg"]["fragments_unresolved"] == 1489
    assert result["comparison"]["parents_raw_better"] == 10
    assert result["comparison"]["old_minus_raw_coordinate_mae_m"] == 1
    assert not any(result["comparison"][key] for key in ("scientific_gate_pass", "whole_model_unseen_claim", "detection_claim"))


@pytest.mark.parametrize("issue", ["method", "179predictions", "31queries", "179targets", "mask_shape", "179records", "nonfinite"])
def test_evaluation_malformed_outputs_rejected(issue):
    axes, target, masks, records = _outputs()
    if issue == "method": axes["extra"] = target.copy()
    elif issue == "179predictions": axes["raw_no_offset"] = axes["raw_no_offset"][:179]
    elif issue == "31queries": axes["raw_no_offset"] = axes["raw_no_offset"][:, :31]
    elif issue == "179targets": target = target[:179]
    elif issue == "mask_shape": masks = masks[:, :31]
    elif issue == "179records": records.pop()
    else: axes["raw_no_offset"][0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        evaluate_outputs(axes, target, masks, records)


def test_real_svg_plots_all18_observations_for_each_parent(tmp_path):
    axes, target, masks, records = _outputs()
    _, result = evaluate_outputs(axes, target, masks, records)
    before = copy.deepcopy(result)
    (tmp_path / "previews").mkdir()
    plot_development(tmp_path, axes, target, masks, records, result)
    files = list((tmp_path / "previews").glob("*.svg"))
    assert len(files) == 11
    for task in result["raw_no_offset"]["parents"]:
        path = tmp_path / f"previews/{task}.svg"
        assert ET.parse(path).getroot().tag == "{http://www.w3.org/2000/svg}svg"
        text = path.read_text()
        for row in range(18):
            assert f"row{row} XY (m)" in text and f"row{row} XZ (m)" in text
    assert result == before


class _Slots(nn.Module):
    def forward(self, query, memory, memory_key_padding_mask):
        assert not memory_key_padding_mask.any()
        return query


class _Model(nn.Module):
    def __init__(self, issue=None):
        super().__init__()
        self.slot_query = nn.Parameter(torch.zeros(32, 128))
        self.slot_decoder = _Slots()
        self.control_query = nn.Linear(128, 384, bias=False)
        nn.init.zeros_(self.control_query.weight)
        self.issue = issue
        self.calls = 0

    def _memory(self, ranges, translation, yaw):
        batch = len(ranges)
        assert not torch.is_grad_enabled()
        return (torch.zeros(batch, 900, 128), torch.ones(batch, 900, dtype=torch.bool),
                torch.zeros(batch, 900, 3), None)

    def forward(self, ranges, translation, yaw):
        self.calls += 1
        value = torch.zeros(len(ranges), 32, 3, 3)
        if self.issue == "memory": value += 1
        if self.issue == "repeat": value += self.calls
        return SimpleNamespace(axis_control_current_sensor_m=value)


class _Head(nn.Module):
    def __init__(self, issue=None):
        super().__init__()
        self.calls = 0
        self.issue = issue

    def forward(self, points, valid, memory, xyz, indices, slots):
        self.calls += 1
        assert not torch.is_grad_enabled()
        assert points.shape == (1, 57600, 3) and valid.shape == (1, 57600)
        assert memory.shape == (1, 900, 128) and slots.shape == (1, 32, 128)
        assert indices.shape == (1, 57600) and indices.min() == 0 and indices.max() == 899
        value = torch.zeros(1, 32, 3, 3)
        if self.issue == "repeat": value += self.calls
        if self.issue == "nan": value[0, 0, 0, 0] = float("nan")
        return SimpleNamespace(votes=SimpleNamespace(axis_control_m=value))


def _student():
    values = np.zeros((5, 2, 16, 720), np.float32)
    values[:, 0] = .1
    values[:, 1] = 1
    row = SimpleNamespace(range_valid=values, relative_translation_current_sensor_m=np.zeros((5, 3), np.float32),
                          relative_yaw_current_sensor_deg=np.zeros(5, np.float32))
    return [row] * 18


def test_infer_task_real_registration_frozen_memory_parity_and_b1_head_no_teacher():
    model, head = _Model(), _Head()
    guards = []
    output = infer_task(model, head, _student(), lambda: guards.append(True), repeat=True)
    assert set(output) == {"raw_no_offset", "legacy_frozen"}
    assert all(value.shape == (18, 32, 3, 3) and value.dtype == np.float32 for value in output.values())
    assert model.calls == 2 and head.calls == 36 and len(guards) == 37
    assert all(parameter.grad is None for parameter in model.parameters())


@pytest.mark.parametrize("issue", ["17rows", "memory", "legacy_repeat", "raw_repeat", "nan"])
def test_inference_rejects_population_parity_and_repeat_drift(issue):
    student = _student()
    if issue == "17rows": student.pop()
    model = _Model("repeat" if issue == "legacy_repeat" else issue)
    head = _Head("repeat" if issue == "raw_repeat" else issue)
    with pytest.raises(ValueError):
        infer_task(model, head, student, lambda: None, repeat=True)
