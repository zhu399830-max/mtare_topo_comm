import math

import numpy as np
import pytest

from mtare_topo.teacher.geometry_variant_contract import (
    GeometryRealization,
    cross_section_area,
    realize_primitive,
    guard_vertical_sensor_offset_at_caps,
)
from mtare_topo.teacher.primitive_construction_supervisor import PrimitiveEndpoint, SweptPrimitive
from mtare_topo.teacher.swept_superellipse_field import SweptSuperellipseProvenanceField


def _source():
    endpoints = (
        PrimitiveEndpoint("primitive:e", 0, "a", (0., 0., 0.)),
        PrimitiveEndpoint("primitive:e", 1, "b", (4., 0., 0.)),
    )
    return SweptPrimitive("primitive:e", "e", "t", np.asarray([[0.,0.,0.],[4.,0.,0.]]), 2., (0.,0.), endpoints)


@pytest.mark.parametrize("realization", list(GeometryRealization))
def test_all_realizations_are_deterministic_area_preserving_and_identity_stable(realization):
    source = _source()
    first = realize_primitive("parent", source, realization)
    second = realize_primitive("parent", source, realization)
    assert first.as_dict() == second.as_dict()
    assert first.primitive_id == source.primitive_id
    for axes, exponent in zip(first.endpoint_half_axes_m, first.endpoint_shape_exponent):
        assert cross_section_area(axes, exponent) == pytest.approx(math.pi * source.radius_m**2, rel=1e-12)


def test_ellipse_and_rounded_rectangle_ranges_are_frozen():
    source = _source()
    ellipse = realize_primitive("p", source, GeometryRealization.ELLIPSE)
    rounded = realize_primitive("p", source, GeometryRealization.ROUNDED_RECTANGLE)
    assert ellipse.endpoint_shape_exponent == (2., 2.)
    for axes in ellipse.endpoint_half_axes_m:
        assert 1.0 <= axes[0] / axes[1] <= 1.25
    for axes, exponent in zip(rounded.endpoint_half_axes_m, rounded.endpoint_shape_exponent):
        assert 6.0 <= exponent <= 10.0
        assert .9 <= axes[0] / axes[1] <= 1.1


def test_mixed_has_one_ellipse_and_one_rounded_endpoint_with_c1_interpolation():
    mixed = realize_primitive("p", _source(), GeometryRealization.C1_MIXED)
    exponents = mixed.endpoint_shape_exponent
    assert sum(value == 2. for value in exponents) == 1
    assert sum(value >= 6. for value in exponents) == 1
    epsilon = 1e-5
    start = mixed.parameters_at_fraction(0.)[0]
    near = mixed.parameters_at_fraction(epsilon)[0]
    assert np.max(np.abs((near-start)/epsilon)) < 1e-3


def test_parent_identity_changes_parameters_without_changing_topology_identity():
    source = _source()
    one = realize_primitive("parent-one", source, GeometryRealization.ROUNDED_RECTANGLE)
    two = realize_primitive("parent-two", source, GeometryRealization.ROUNDED_RECTANGLE)
    assert one.primitive_id == two.primitive_id
    assert one.endpoint_half_axes_m != two.endpoint_half_axes_m


def test_vertical_sensor_cap_guard_leaves_flat_sweep_unchanged():
    primitive = realize_primitive("p", _source(), GeometryRealization.ELLIPSE)
    guarded, records = guard_vertical_sensor_offset_at_caps(
        (primitive,), vertical_sensor_offset_m=-1.0, discretization_guard_m=.025,
    )
    assert np.array_equal(guarded[0].centerline_xyz_m, primitive.centerline_xyz_m)
    assert records[0].start_extension_m == records[0].end_extension_m == 0.0


def test_vertical_sensor_cap_guard_extends_only_the_outward_sloped_cap():
    source = _source()
    source = SweptPrimitive(
        source.primitive_id, source.source_edge_id, source.source_tunnel_id,
        np.asarray([[0., 0., 0.], [4., 0., 2.]]), source.radius_m,
        source.endpoint_projection_error_m, source.endpoints,
    )
    primitive = realize_primitive("p", source, GeometryRealization.C1_MIXED)
    guarded, records = guard_vertical_sensor_offset_at_caps(
        (primitive,), vertical_sensor_offset_m=-1.0, discretization_guard_m=.025,
    )
    expected = 2.0 / math.sqrt(20.0) + .025
    assert records[0].start_extension_m == pytest.approx(expected)
    assert records[0].end_extension_m == 0.0
    assert guarded[0].primitive_id == primitive.primitive_id
    assert guarded[0].endpoint_half_axes_m == primitive.endpoint_half_axes_m
    field = SweptSuperellipseProvenanceField(guarded, spacing_m=.025)
    assert field.signed_distance(np.asarray([[0., 0., -1.]]))[0] < 0.0
