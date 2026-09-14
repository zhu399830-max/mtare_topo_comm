from types import SimpleNamespace
import numpy as np
import pytest
from mtare_topo.teacher import gse_junction_interface_diagnostic_v1 as module
from test_gse_terminal_diagnostic_v1 import bundle


def test_incidence_and_original_frame_binding(monkeypatch):
    b = bundle()
    origin = b['sensor_teacher_only']['sensor_xyz_m'][-1].tolist()
    monkeypatch.setattr(module, 'construction_incident_paths', lambda d: (
        dict(node_id_teacher_only='junction', anchor_world_m=origin,
             paths=[dict(endpoint_key=('a', 0)), dict(endpoint_key=('a', 1)),
                    dict(endpoint_key=('b', 0))]),))
    monkeypatch.setattr(module, 'load_p1a_realized_construction', lambda d:
                        (None, [SimpleNamespace(primitive_id='a'), SimpleNamespace(primitive_id='b')]))
    monkeypatch.setattr(module, 'mesh_swept_superellipse', lambda *a, **kw: object())
    monkeypatch.setattr(module, 'source_endpoint_cap_faces',
                        lambda mesh, angular_segments, endpoint_index: [endpoint_index])
    def capture(sources, **kw):
        assert len(sources) == 2
        assert sources[0].face_interfaces == {0: 0, 1: 1}
        assert sources[1].face_interfaces == {0: 2}
        np.testing.assert_array_equal(kw['source_frame_indices'], np.repeat(np.arange(5), 11520))
        np.testing.assert_array_equal(kw['origins'][::11520], b['sensor_teacher_only']['sensor_xyz_m'])
        assert kw['first_return'].dtype == np.float32
        return dict(rays=[], raw_interface_intersections=[dict(source_frame_index=3,
                    ray_index=34560)], semantic_label=None, training_eligible=False)
    monkeypatch.setattr(module, 'replay_interfaces', capture)
    out = module.diagnose_observation(b)
    assert len(out['interfaces_teacher_only']) == 3
    assert out['raw_interface_intersections'][0]['source_frame_row'] == 16
    assert out['raw_interface_intersections'][0]['return_source_indices_teacher_only'] == [0, 1]
    assert out['labels_generated'] == 0


def test_source_mismatch_before_geometry():
    b = bundle()
    b['student']['valid_mask'][0, 0, 0] = 0
    with pytest.raises(ValueError, match='validity'):
        module.diagnose_observation(b)


def test_frame_permutation_before_geometry():
    b = bundle()
    b['source']['frame_rows'] = [10, 14, 12, 16, 18]
    with pytest.raises(ValueError, match='increasing'):
        module.diagnose_observation(b)
