from types import SimpleNamespace as N
import numpy as np
import pytest
from mtare_topo.data.covered_sensor_adapter import diagnostic_frame, render_covered_frame
from mtare_topo.data.primitive_relation_dataset import PrimitiveMembershipCodebook


def convert(rows, book):
    return diagnostic_frame(rows, sensor_xyz_m=[0, 0, 0],
                            directions_xyz=np.tile([2., 0., 0.], (11520, 1)), codebook=book)


def test_layout_float32_range_and_multisource_roundtrip():
    book = PrimitiveMembershipCodebook(['a', 'b'])
    rows = [dict(status='candidate', distance_m=2., sources=['a']) for _ in range(11520)]
    rows[720] = dict(status='reference_return', distance_m=3., sources=['b', 'a'])
    frame = convert(rows, book)
    assert frame.range_m.dtype == np.float32 and frame.range_m[1, 0] == 3.
    assert frame.valid_mask.dtype == np.uint8 and frame.valid_mask.all()
    assert frame.primitive_membership_code.dtype == np.uint16
    assert book.decode(frame.primitive_membership_code)[720] == (0, 1)
    assert frame.ambiguous_ray_count == 1


@pytest.mark.parametrize('bad', [dict(status='needs_reference'),
    dict(status='candidate', distance_m=float('nan'), sources=['a']),
    dict(status='candidate', distance_m=2., sources=['unknown']),
    dict(status='candidate', distance_m=2., sources=['a', 'a'])])
def test_bad_return_rejects_without_mutating_codebook(bad):
    book = PrimitiveMembershipCodebook(['a'])
    rows = [dict(status='candidate', distance_m=2., sources=['a'])]*11519 + [bad]
    with pytest.raises(ValueError):
        convert(rows, book)
    assert book.source_sets == ((),)


def test_near_and_far_are_invalid_not_qualified_max_range_hits():
    book = PrimitiveMembershipCodebook(['a'])
    rows = [dict(status='out_of_range') for _ in range(11520)]
    rows[0] = dict(status='candidate', distance_m=.2, sources=['a'])
    rows[1] = dict(status='candidate', distance_m=50., sources=['a'])
    frame = convert(rows, book)
    assert frame.valid_mask.sum() == 1 and frame.valid_mask[0, 1] == 1
    assert frame.range_m[0, 0] == 50. and frame.primitive_membership_code[0, 0] == 0


@pytest.mark.parametrize('count', [11519, 11521])
def test_wrong_count_rejected(count):
    with pytest.raises(ValueError):
        convert([dict(status='out_of_range')]*count, PrimitiveMembershipCodebook(['a']))


def test_renderer_calls_diagnostic_without_teacher_or_field():
    seen = []
    def query(origin, direction, *, maximum_m):
        assert maximum_m == 50.
        seen.append(direction)
        return dict(status='candidate', distance_m=2., sources=['a'])
    frame = render_covered_frame(diagnostic=N(query=query), codebook=PrimitiveMembershipCodebook(['a']),
                                 sensor_xyz_m=[0, 0, 0], yaw_deg=0.)
    assert len(seen) == 11520 and frame.valid_mask.all()
