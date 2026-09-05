"""Finite-ray section evidence only; no actual worlds or teacher generation."""
import copy
import inspect

import numpy as np
import pytest

from mtare_topo.teacher.gse_local_opening import section_positive_evidence
from mtare_topo.teacher.gse_portal_ray_evidence import CausalRaySegments, PortalSections


def _sections(centers, normals=None):
    centers = np.asarray(centers, dtype=np.float64)
    normals = np.asarray(normals if normals is not None else [[1., 0., 0.]] * len(centers), dtype=np.float64)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    width = np.cross(np.broadcast_to([0., 0., 1.], normals.shape), normals)
    width /= np.linalg.norm(width, axis=1, keepdims=True)
    polygon = np.tile(np.array([[.5, .5], [-.5, .5], [-.5, -.5], [.5, -.5]])[None], (len(centers), 1, 1))
    return PortalSections(centers, normals, width, np.ones((len(centers), 2)), np.full(len(centers), 2.), polygon)


def _rays(targets, origins=None, returns=10., valid=None):
    targets = np.asarray(targets, dtype=np.float64)
    origins = np.zeros_like(targets) if origins is None else np.asarray(origins, dtype=np.float64)
    direction = targets - origins; direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    return CausalRaySegments(origins, direction, np.full(len(targets), returns, np.float32),
                            np.ones(len(targets), bool) if valid is None else np.asarray(valid, bool),
                            np.zeros(len(targets), np.int64), 0, 0.)


def _assert_not_region_teacher(result):
    assert result.region_membership_available is False
    assert result.complete_event_labels_available is False
    assert result.independent_opening_claim is False


@pytest.mark.parametrize("angles", [[0., 90., 180.], [0., 120., 240.]])
def test_t_and_y_positive_sections_do_not_automatically_label_junction(angles):
    radians = np.deg2rad(angles)
    normals = np.stack((np.cos(radians), np.sin(radians), np.zeros(3)), axis=1)
    centers = normals * 2
    result = section_positive_evidence(_rays(centers), _sections(centers, normals))
    assert result.section_status == ("SECTION_POSITIVE",) * 3
    _assert_not_region_teacher(result)


def test_exact_duplicate_caps_count_once_and_preserve_original_ambiguity():
    section = _sections([[2., 0., 0.], [2., 0., 0.]])
    original = copy.deepcopy(section)
    result = section_positive_evidence(_rays([[2., 0., 0.]]), section)
    assert result.source_candidate_groups == ((0, 1),)
    assert result.candidate_to_section.tolist() == [0, 0]
    assert result.original_evidence.ambiguous_ray_count == 1
    assert not result.original_evidence.witnessed.any()
    assert result.unique_section_evidence.witnessed.tolist() == [True]
    assert result.unique_section_evidence.exclusive_ray_count.tolist() == [1]
    for name in vars(section): np.testing.assert_array_equal(getattr(section, name), getattr(original, name))
    _assert_not_region_teacher(result)


def test_opposite_normal_identical_world_polygon_is_duplicate_not_second_exit():
    result = section_positive_evidence(_rays([[2., 0., 0.]]),
                                       _sections([[2., 0., 0.], [2., 0., 0.]], [[1., 0., 0.], [-1., 0., 0.]]))
    assert result.source_candidate_groups == ((0, 1),)
    assert result.unique_section_evidence.witnessed.tolist() == [True]


def test_nearby_caps_not_collapsed_by_a_chosen_tolerance():
    sections = _sections([[2., 0., 0.], [2. + 1.e-9, 0., 0.]])
    result = section_positive_evidence(_rays([[3., 0., 0.]]), sections)
    assert result.source_candidate_groups == ((0,), (1,))
    assert result.section_status == ("UNKNOWN_NO_EXCLUSIVE_RAY",) * 2


def test_serial_sections_kept_unknown_not_selected_by_incident_identity():
    result = section_positive_evidence(_rays([[6., 0., 0.]]), _sections([[2., 0., 0.], [4., 0., 0.]]))
    assert result.unique_section_evidence.crossing_ray_count.tolist() == [1, 1]
    assert result.unique_section_evidence.ambiguous_ray_count == 1
    assert not result.unique_section_evidence.witnessed.any()
    _assert_not_region_teacher(result)


def test_stacked_and_two_region_evidence_preserved_without_region_merging():
    centers = [[2., 0., -3.], [2., 0., 3.]]
    origins = [[0., 0., -3.], [0., 0., 3.]]
    result = section_positive_evidence(_rays(centers, origins), _sections(centers))
    assert result.unique_section_evidence.witnessed.tolist() == [True, True]
    assert result.source_candidate_groups == ((0,), (1,))
    _assert_not_region_teacher(result)


def test_same_observation_two_spatial_regions_not_forced_into_single_event():
    centers = [[3., -2., 0.], [3., 2., 0.]]
    result = section_positive_evidence(_rays(centers), _sections(centers))
    assert result.section_status == ("SECTION_POSITIVE", "SECTION_POSITIVE")
    _assert_not_region_teacher(result)


@pytest.mark.parametrize("case", ["no_return", "occluded", "first_return_on_cap"])
def test_absence_is_unknown_never_closed_or_terminal(case):
    rays = _rays([[2., 0., 0.]], returns=2. if case == "first_return_on_cap" else 1.,
                 valid=[False] if case == "no_return" else [True])
    result = section_positive_evidence(rays, _sections([[2., 0., 0.]]))
    assert result.section_status == ("UNKNOWN_NO_EXCLUSIVE_RAY",)
    _assert_not_region_teacher(result)


def test_observation_does_not_determine_unobserved_separator_or_region_membership():
    # Scene A has an undivided area; scene B has a divider y=0, 1<=x<=7.
    # Both finite measured rays miss the divider. Thus these supplied sections
    # and first returns are identical despite different unobserved geometry.
    centers = [[3., -1., 0.], [3., 1., 0.]]
    rays = _rays(centers, returns=8.)
    for direction in rays.directions:
        # At every point of the potential divider's x extent, the ray has
        # nonzero y; the only y=0 crossing is its origin, outside that extent.
        assert direction[0] > 0 and direction[1] != 0
        assert direction[1] / direction[0] != 0
    scene_a = section_positive_evidence(rays, _sections(centers))
    scene_b = section_positive_evidence(copy.deepcopy(rays), _sections(centers))
    assert scene_a.section_status == scene_b.section_status == ("SECTION_POSITIVE",) * 2
    np.testing.assert_array_equal(scene_a.unique_section_evidence.crossing_ray_count,
                                  scene_b.unique_section_evidence.crossing_ray_count)
    _assert_not_region_teacher(scene_a); _assert_not_region_teacher(scene_b)


def test_identical_section_evidence_has_two_distinct_geometric_region_references():
    # One observed straight passage, cuts x=2 and x=4, sensor x=3. A query
    # referring to the section ending at x=2 has only the first cut as a
    # boundary; a query referring to the portion 2<x<4 has both. This is not
    # an alternative map or a required region-mesh input: it demonstrates
    # why an evidence API with no region reference cannot output membership.
    sections = _sections([[2., 0., 0.], [4., 0., 0.]])
    rays = _rays([[2., 0., 0.], [4., 0., 0.]], [[3., 0., 0.], [3., 0., 0.]])
    result = section_positive_evidence(rays, sections)
    assert result.section_status == ("SECTION_POSITIVE",) * 2
    reference_a_boundary = np.isin(sections.centers_m[:, 0], [-2., 2.])
    reference_b_boundary = np.isin(sections.centers_m[:, 0], [2., 4.])
    assert reference_a_boundary.tolist() == [True, False]
    assert reference_b_boundary.tolist() == [True, True]
    _assert_not_region_teacher(result)


def test_old_target_node_and_composition_identity_have_no_interface():
    parameters = inspect.signature(section_positive_evidence).parameters
    assert set(parameters) == {"rays", "sections", "chunk_size"}
    rays, sections = _rays([[2., 0., 0.]]), _sections([[2., 0., 0.]])
    with pytest.raises(TypeError): section_positive_evidence(rays, sections, target_node="old_sample_identity")
    assert section_positive_evidence(rays, sections).section_status == ("SECTION_POSITIVE",)


def test_missing_source_polygon_is_rejected_not_replaced_by_ideal_ellipse():
    sections = _sections([[2., 0., 0.]])
    sections = PortalSections(**{**vars(sections), "polygon_normalized_xy": None})
    with pytest.raises(ValueError, match="source mesh section polygon"):
        section_positive_evidence(_rays([[2., 0., 0.]]), sections)
