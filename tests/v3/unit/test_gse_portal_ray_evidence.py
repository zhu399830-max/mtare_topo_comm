"""Analytic 3-D ray witnesses; no dataset or new training labels."""
from dataclasses import replace

import numpy as np
import pytest

from mtare_topo.teacher.gse_portal_ray_evidence import (
    CausalRaySegments, PortalSections, aperture_ray_evidence, construction_cap_candidates,
)


def rays(origins=((0., -2., 0.),), directions=((0., 1., 0.),), ranges=(8.,)):
    n = len(origins)
    return CausalRaySegments(np.asarray(origins), np.asarray(directions), np.asarray(ranges),
                             np.ones(n, dtype=bool), np.zeros(n, dtype=np.int64), 4, 0.)


def portals(centers=((0., 0., 0.),), normals=((0., 1., 0.),)):
    n = len(centers)
    return PortalSections(np.asarray(centers), np.asarray(normals),
                          np.tile((1., 0., 0.), (n, 1)), np.ones((n, 2)), np.full(n, 2.))


def test_open_branch_positive_without_end_wall_return():
    value = aperture_ray_evidence(rays(), portals())
    assert value.witnessed.tolist() == [True]
    assert value.witness_ray_index.tolist() == [0]


@pytest.mark.parametrize("distance", [1., 2.])
def test_occluded_or_surface_only_is_unknown_not_open(distance):
    value = aperture_ray_evidence(replace(rays(), first_return_m=np.array([distance])), portals())
    assert not value.witnessed.any()
    assert value.witness_ray_index.tolist() == [-1]


def test_range_uncertainty_does_not_extrapolate_free_space():
    value = replace(rays(), first_return_m=np.array([2.1]), range_error_bound_m=.2)
    assert not aperture_ray_evidence(value, portals()).witnessed.any()


def test_stacked_portal_not_joined_by_xy_projection():
    value = portals(((0., 0., 0.), (0., 0., 3.)), ((0., 1., 0.), (0., 1., 0.)))
    assert aperture_ray_evidence(rays(), value).witnessed.tolist() == [True, False]


def test_duplicate_portals_are_unknown_not_two_exits():
    value = portals(((0., 0., 0.), (0., 0., 0.)), ((0., 1., 0.), (0., 1., 0.)))
    result = aperture_ray_evidence(rays(), value)
    assert result.crossing_ray_count.tolist() == [1, 1]
    assert result.exclusive_ray_count.tolist() == [0, 0]
    assert result.ambiguous_ray_count == 1


def test_two_serial_portals_not_independent_witnesses():
    value = portals(((0., 0., 0.), (0., 3., 0.)), ((0., 1., 0.), (0., 1., 0.)))
    assert not aperture_ray_evidence(rays(), value).witnessed.any()


def test_reverse_ray_witnesses_connection_from_inside_passage():
    value = rays(((0., 2., 0.),), ((0., -1., 0.),))
    result = aperture_ray_evidence(value, portals())
    assert result.witnessed.all()
    assert result.inward_ray_count.tolist() == [0]
    assert result.outward_ray_count.tolist() == [1]


@pytest.mark.parametrize("offset", [1., 1.1])
def test_boundary_or_outside_crossing_rejected(offset):
    value = rays(((offset, -2., 0.),))
    assert not aperture_ray_evidence(value, portals()).witnessed.any()


def test_exact_portal_origin_unknown_no_backward_extension():
    assert not aperture_ray_evidence(rays(((0., 0., 0.),)), portals()).witnessed.any()
    assert not aperture_ray_evidence(rays(((0., .1, 0.),)), portals()).witnessed.any()


def test_invalid_no_return_not_fabricated_free_space():
    value = replace(rays(), valid=np.array([False]), first_return_m=np.array([np.nan]))
    assert not aperture_ray_evidence(value, portals()).witnessed.any()


def test_future_ray_rejected_even_if_marked_invalid():
    value = replace(rays(), source_frame_index=np.array([5]), valid=np.array([False]))
    with pytest.raises(ValueError, match="future"):
        aperture_ray_evidence(value, portals())


def test_common_rigid_transform_invariance_and_sloped_section():
    angle = .43
    rotation = np.array([[np.cos(angle), 0., np.sin(angle)], [0., 1., 0.],
                         [-np.sin(angle), 0., np.cos(angle)]])
    yaw = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    rotation = rotation @ yaw
    shift = np.array([3., -4., 2.])
    ray, port = rays(), portals()
    moved_ray = replace(ray, origins_m=ray.origins_m @ rotation.T + shift,
                        directions=ray.directions @ rotation.T)
    moved_port = replace(port, centers_m=port.centers_m @ rotation.T + shift,
                         inward_normals=port.inward_normals @ rotation.T,
                         width_directions=port.width_directions @ rotation.T)
    assert aperture_ray_evidence(moved_ray, moved_port).witnessed.tolist() == [True]


def test_permutation_and_chunk_size_do_not_change_port_decisions():
    port = portals(((0., 0., 0.), (0., 0., 3.)), ((0., 1., 0.), (0., 1., 0.)))
    ray = rays(((0., -2., 0.), (0., -2., 3.)), ((0., 1., 0.),) * 2, (8.,) * 2)
    expected = aperture_ray_evidence(ray, port, chunk_size=1)
    permuted = PortalSections(*(getattr(port, name)[::-1] if getattr(port, name) is not None else None
                               for name in port.__dataclass_fields__))
    actual = aperture_ray_evidence(ray, permuted, chunk_size=2)
    np.testing.assert_array_equal(expected.witnessed, actual.witnessed[::-1])
    np.testing.assert_array_equal(expected.exclusive_ray_count, actual.exclusive_ray_count[::-1])
    np.testing.assert_array_equal(expected.witness_ray_index, actual.witness_ray_index[::-1])


@pytest.mark.parametrize("field,value", [("half_axes_m", np.zeros((1, 2))),
                                         ("exponents", np.array([.5])),
                                         ("inward_normals", np.array([[0., 2., 0.]])),
                                         ("width_directions", np.array([[0., 1., 0.]]))])
def test_invalid_portal_contract_fails(field, value):
    with pytest.raises(ValueError):
        aperture_ray_evidence(rays(), replace(portals(), **{field: value}))


def test_empty_ray_population_is_unknown():
    ray = CausalRaySegments(np.empty((0, 3)), np.empty((0, 3)), np.empty(0),
                            np.empty(0, bool), np.empty(0, np.int64), 4, 0.)
    assert not aperture_ray_evidence(ray, portals()).witnessed.any()


def test_float32_cap_upward_rounding_cannot_become_opening():
    # 2.2 becomes 2.200000047683716 in the actual float32 source format.
    value = replace(rays(((0., -2.2, 0.),)), first_return_m=np.array([2.2], dtype=np.float32))
    assert float(value.first_return_m[0]) > 2.2
    assert not aperture_ray_evidence(value, portals()).witnessed.any()


def test_noncentral_boundary_contact_unknown_under_repeated_rigid_transforms():
    rng = np.random.default_rng(19)
    ray, port = rays(((.4, 0., .3),)), portals()
    for _ in range(50):
        rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        shift = rng.normal(size=3) * 10
        moved_ray = replace(ray, origins_m=ray.origins_m @ rotation.T + shift,
                            directions=ray.directions @ rotation.T)
        moved_port = replace(port, centers_m=port.centers_m @ rotation.T + shift,
                             inward_normals=port.inward_normals @ rotation.T,
                             width_directions=port.width_directions @ rotation.T)
        assert not aperture_ray_evidence(moved_ray, moved_port).witnessed.any()


def test_witness_does_not_certify_robot_width():
    # This intentionally witnesses a tiny visible aperture, NOT traversability.
    tiny = replace(portals(), half_axes_m=np.full((1, 2), .001))
    assert aperture_ray_evidence(rays(), tiny).witnessed.all()


def primitive(identity="one", points=((0., 0., 0.), (4., 2., 1.))):
    from mtare_topo.teacher.swept_superellipse_field import SweptSuperellipsePrimitive
    return SweptSuperellipsePrimitive(identity, np.array(points), ((1., 2.), (2., 3.)), (2., 8.))


def test_cap_adapter_uses_both_ends_and_original_mesh_sample_basis():
    from mtare_topo.teacher.swept_superellipse_field import _sample_operand
    value = primitive()
    expected = _sample_operand(value, .05)
    result = construction_cap_candidates([value], mesh_axial_spacing_m=.05, angular_segments=64)
    assert result.teacher_keys == (("one", 0), ("one", 1))
    np.testing.assert_array_equal(result.sections.centers_m, expected.points[[0, -1]])
    np.testing.assert_array_equal(result.sections.half_axes_m, [[1., 2.], [2., 3.]])
    np.testing.assert_allclose(result.sections.inward_normals, expected.tangents[[0, -1]] * [[1], [-1]])


def test_cap_adapter_does_not_collapse_coincident_or_same_tunnel_operands():
    values = [primitive("edge_a"), primitive("edge_b")]
    result = construction_cap_candidates(values, mesh_axial_spacing_m=.05, angular_segments=64)
    assert len(result.teacher_keys) == 4
    np.testing.assert_array_equal(result.sections.centers_m[:2], result.sections.centers_m[2:])


def test_cap_adapter_preserves_frozen_extensions_not_original_node_position():
    result = construction_cap_candidates([primitive(points=((-.1, 0., 0.), (4., 0., 0.)))],
                                          mesh_axial_spacing_m=.05, angular_segments=64)
    assert result.sections.centers_m[0, 0] == -.1


def test_cap_adapter_near_vertical_basis_matches_source():
    result = construction_cap_candidates([primitive(points=((0., 0., 0.), (0., 0., 4.)))],
                                          mesh_axial_spacing_m=.05, angular_segments=64)
    result.sections.validate()
    np.testing.assert_allclose(result.sections.inward_normals, [[0, 0, 1], [0, 0, -1]])


@pytest.mark.parametrize("values", [[], [primitive(), primitive()]])
def test_cap_adapter_empty_or_duplicate_inventory_rejected(values):
    with pytest.raises(ValueError):
        construction_cap_candidates(values, mesh_axial_spacing_m=.05, angular_segments=64)


def test_mesh_spacing_not_sdf_spacing_on_short_bending_segment():
    value = primitive(points=((0., 0., 0.), (.02, 0., 0.), (.02, 1., 0.)))
    mesh = construction_cap_candidates([value], mesh_axial_spacing_m=.05, angular_segments=64)
    finer = construction_cap_candidates([value], mesh_axial_spacing_m=.025, angular_segments=64)
    assert np.linalg.norm(mesh.sections.inward_normals[0] - finer.sections.inward_normals[0]) > .1


def test_mesh_polygon_rejects_ideal_ellipse_sliver_outside_actual_cap():
    from mtare_topo.teacher.swept_superellipse_field import SweptSuperellipsePrimitive
    value = SweptSuperellipsePrimitive("ellipse", np.array([[0., 0., 0.], [0., 2., 0.]]),
                                       ((1., 1.), (1., 1.)), (2., 2.))
    cap = construction_cap_candidates([value], mesh_axial_spacing_m=.05, angular_segments=12).sections
    # Halfway between vertices: unit circle is larger than the polygon chord.
    angle = np.pi / 12
    origin = ((-.99 * np.cos(angle), -1., .99 * np.sin(angle)),)
    ray = rays(origin, ranges=(1.5,))
    ideal = replace(cap, polygon_normalized_xy=None)
    assert aperture_ray_evidence(ray, ideal).witnessed[0]
    assert not aperture_ray_evidence(ray, cap).witnessed[0]
