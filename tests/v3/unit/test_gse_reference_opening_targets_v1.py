"""Producer wiring tests; diagnostic geometry is separately tested, not mocked truth."""
from copy import deepcopy
import numpy as np
import pytest

from mtare_topo.teacher import gse_reference_opening_targets_v1 as module
from mtare_topo.teacher.gse_surface_target_adapter_v1 import observed_targets


def run(monkeypatch, rows=None):
    row = dict(reference_position_m=[2., 13., 4.], reference_direction=[0., 1., 0.],
        exclusive_outward_crossing_ray_indices=[1], surface_return_ray_indices=[2],
        primitive_id_teacher_only='secret', reference_arc_m=12.)
    rows = [row] if rows is None else rows
    monkeypatch.setattr(module, 'diagnose_observation', lambda b: dict(proposals=deepcopy(rows)))
    bundle = dict(source=dict(frame_rows=[1, 2, 3, 4, 5]),
        sensor_teacher_only=dict(sensor_xyz_m=np.tile([2., 3., 4.], (5, 1)),
            yaw_deg=np.full(5, 90.)))
    return module.produce_opening_reference_targets(bundle), row


def test_nonempty_position_direction_without_fake_dimensions_or_anchor(monkeypatch):
    result, _ = run(monkeypatch)
    record = result['record']
    np.testing.assert_allclose(record['openings'][0]['position_m'], [10., 0., 0.], atol=1e-14)
    np.testing.assert_allclose(record['openings'][0]['direction'], [1., 0., 0.], atol=1e-14)
    target = observed_targets([record])
    assert target.opening_valid.tolist() == [[True]]
    assert record['anchors'] == [] and record['membership'] == [[]]
    assert record['openings'][0]['width_m'] is None
    assert record['openings'][0]['height_m'] is None
    assert 'secret' not in str(record)
    assert not result['full_training_gate_eligible']
    assert not target.opening_region_complete.any()


@pytest.mark.parametrize('field', ['exclusive_outward_crossing_ray_indices', 'surface_return_ray_indices'])
def test_missing_or_competing_support_stays_unknown(monkeypatch, field):
    _, row = run(monkeypatch)
    row[field] = []
    result, _ = run(monkeypatch, [row])
    assert result['record']['openings'] == []
    assert len(result['unknown_candidates']) == 1
    assert not result['record']['score_region']['openings_complete']


def test_determinism_empty_and_duplicate_rejection(monkeypatch):
    a, row = run(monkeypatch)
    b, _ = run(monkeypatch)
    assert a == b
    empty, _ = run(monkeypatch, [])
    assert empty['record']['openings'] == []
    with pytest.raises(ValueError, match='coincident'):
        run(monkeypatch, [row, row])


def test_capacity_is_not_silent_truncation(monkeypatch):
    _, row = run(monkeypatch)
    rows = [dict(row, reference_position_m=[2. + i, 13., 4.]) for i in range(65)]
    with pytest.raises(OverflowError, match='capacity'):
        run(monkeypatch, rows)
