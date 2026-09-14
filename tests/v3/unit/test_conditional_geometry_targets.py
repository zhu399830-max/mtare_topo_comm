import numpy as np
import pytest
from mtare_topo.teacher.gse_conditional_geometry_targets import conditional_geometry_targets as build


def sample():
    c = [dict(source_index=0, component_index=0, source_return_indices=[0, 1], center_m=[0, 0, 0], normal=[1, 0, 0]),
         dict(source_index=0, component_index=1, source_return_indices=[2, 3], center_m=[0, 0, 2], normal=[0, 1, 0])]
    return c, np.array([0, 0, 1, 1]), np.arange(4), np.array([[1], [0]])


def test_partial_reference_no_contour_required_and_signed_height():
    out = build(*sample())
    assert out['axis_known'].all() and not out['axis_abs_dot'].any()
    assert np.array_equal(out['height_difference_m'], [[2], [-2]])
    assert not out['section_known'].any() and not out['correspondence_known'].any()
    assert not out['observability_certified'] and not out['connectivity_certified']


def test_same_source_is_not_same_component():
    out = build(*sample())
    assert not out['same_reference_component'].any()


@pytest.mark.parametrize('kind', ['missing', 'ambiguous', 'outside', 'mixed'])
def test_unresolved_not_background(kind):
    c, assignment, roi, n = sample()
    if kind == 'missing': c[0]['source_return_indices'] = [0]
    if kind == 'ambiguous': c[1]['source_return_indices'].append(0)
    if kind == 'outside': c[0]['center_m'] = [11, 0, 0]
    if kind == 'mixed': assignment = np.array([0, 1, 0, 1])
    out = build(c, assignment, roi, n)
    assert not out['axis_known'].any() and np.isnan(out['axis_abs_dot']).all()


def test_reference_order_and_ray_order_do_not_change_targets():
    c, a, r, n = sample(); one = build(c, a, r, n); two = build(c[::-1], a, r[::-1], n)
    assert np.array_equal(one['axis_abs_dot'], two['axis_abs_dot'])
    assert np.array_equal(one['height_difference_m'], two['height_difference_m'])


def test_no_reference_means_all_unknown():
    _, a, r, n = sample(); out = build([], a, r, n)
    assert not out['axis_known'].any()
