import numpy as np
import pytest

from mtare_topo.teacher import gse_terminal_diagnostic_v1 as module
from test_gse_surface_teacher_reader_v1 import inputs


def bundle():
    sensor, student, construction, codebook, source = inputs()
    source['frame_rows'] = [10, 12, 14, 16, 18]
    student['ranges_m'] = np.full((5, 16, 720), 3., dtype=np.float32)
    return dict(sensor_teacher_only=sensor, student=student,
                construction_teacher_only=construction, codebook_teacher_only=codebook, source=source)


def test_real_ray_builder_preserves_frame_and_code_flattening(monkeypatch):
    b = bundle()
    monkeypatch.setattr(module, 'construction_incident_paths', lambda d: ())
    monkeypatch.setattr(module, 'load_p1a_realized_construction', lambda d: (None, ()))
    def capture(groups, primitives, **kw):
        rays = kw['rays']; rays.validate()
        np.testing.assert_array_equal(rays.source_frame_index, np.repeat([10, 12, 14, 16, 18], 11520))
        np.testing.assert_array_equal(rays.origins_m[::11520], b['sensor_teacher_only']['sensor_xyz_m'])
        np.testing.assert_array_equal(kw['membership_codes'], b['sensor_teacher_only']['primitive_membership_code'].reshape(-1))
        assert rays.first_return_m.dtype == np.float32
        assert kw['source_sets'] == [[], [0, 1]]
        return dict(terminal_references=[], training_eligible=False)
    monkeypatch.setattr(module, 'terminal_reference_evidence', capture)
    result = module.diagnose_observation(b, range_error_bound_m=0.)
    assert result['local_terminal_count'] == result['labels_generated'] == 0


def test_frame_permutation_rejected_before_geometry():
    b = bundle(); b['source']['frame_rows'] = [10, 14, 12, 16, 18]
    with pytest.raises(ValueError, match='increasing'):
        module.diagnose_observation(b, range_error_bound_m=0.)


def test_source_return_mismatch_rejected_before_geometry():
    b = bundle(); b['student']['valid_mask'][0, 0, 0] = 0
    with pytest.raises(ValueError, match='validity'):
        module.diagnose_observation(b, range_error_bound_m=0.)
