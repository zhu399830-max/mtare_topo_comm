"""Partial terminal production wiring, not real label certification."""
from copy import deepcopy
import pytest
from mtare_topo.teacher import gse_reference_terminal_targets_v1 as module
from mtare_topo.teacher.gse_surface_target_adapter_v1 import observed_targets


def call(monkeypatch, rows):
    def diagnostic(bundle, *, range_error_bound_m):
        assert range_error_bound_m == 0.
        return dict(terminal_references=deepcopy(rows))
    monkeypatch.setattr(module, 'diagnose_observation', diagnostic)
    return module.produce_terminal_reference_targets(dict(source=dict(frame_rows=[1,2,3,4,5])))


def row():
    return dict(node_id_teacher_only='n', endpoint_key_teacher_only=['p',1],
        anchor_current_sensor_m=[2.,0.,0.], accepted_witnesses=[dict(ray_index=4, stored_t=2.)])


def test_witness_generates_partial_position_not_complete_background(monkeypatch):
    result = call(monkeypatch, [row()])
    target = observed_targets([result['record']])
    assert target.anchor_valid.tolist() == [[True]]
    assert not target.anchor_region_complete.any()
    assert result['record']['openings'] == []
    assert 'node_id_teacher_only' not in str(result['record'])
    assert not result['full_training_gate_eligible']


def test_reference_flag_without_witness_cannot_generate_target(monkeypatch):
    r = row(); r.update(accepted_witnesses=[], terminal_reference_observed=True)
    result = call(monkeypatch, [r])
    assert result['record']['anchors'] == []
    assert len(result['unknown_candidates']) == 1


def test_empty_and_capacity(monkeypatch):
    assert call(monkeypatch, [])['record']['anchors'] == []
    with pytest.raises(OverflowError):
        call(monkeypatch, [dict(row(), anchor_current_sensor_m=[float(i),0.,0.]) for i in range(33)])
