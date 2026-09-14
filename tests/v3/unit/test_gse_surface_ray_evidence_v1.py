"""Exact finite-ray software fixtures, not physical clearance labels."""
from dataclasses import replace
from fractions import Fraction
import numpy as np
import pytest

from mtare_topo.representation.gse_surface_ray_evidence_v1 import (
    FREE, OCCUPIED, UNKNOWN, SHAPE, build_surface_ray_grid, query_patch_gaps,
    _segment_cells,
)


def grid(origins=((.125, .125, .125),), endpoints=((1.125, .125, .125),), frames=None, valid=None):
    o, e = np.asarray(origins, dtype=np.float64).reshape(-1, 3), np.asarray(endpoints, dtype=np.float64).reshape(-1, 3)
    n = len(o)
    return build_surface_ray_grid(o, e, np.ones(n, dtype=bool) if valid is None else valid,
                                  np.zeros(n, dtype=np.int64) if frames is None else frames)


def at(value):
    return tuple(np.floor((np.asarray(value) + 10.) / .25).astype(int))


def gap(g, left=(.125, .125, .125), right=(1.125, .125, .125)):
    index = np.full((2, 8), -1, dtype=np.int64); index[0, 0] = 1; index[1, 0] = 0
    return query_patch_gaps(g, np.asarray([left, right], dtype=np.float64), index, index >= 0)


def test_axis_aligned_interior_traversal_and_real_return():
    g = grid()
    assert g.state.shape == SHAPE and g.state.dtype == np.uint8
    for x in (.125, .375, .625, .875):
        assert g.state[at((x, .125, .125))] == FREE
    assert g.state[at((1.125, .125, .125))] == OCCUPIED
    assert g.state[at((1.375, .125, .125))] == UNKNOWN
    assert (g.free_frame_bits != 0).sum() == 4
    assert (g.occupied_frame_bits != 0).sum() == 1
    assert g.state.nbytes + g.free_frame_bits.nbytes + g.occupied_frame_bits.nbytes == 1536000
    assert not g.physical_connectivity and not g.state.flags.writeable


def test_gap_excludes_endpoint_surface_cells_and_is_symmetric():
    g = grid(origins=((.125, .125, .125), (1.125, .125, .125)),
             endpoints=((1.125, .125, .125), (.125, .125, .125)))
    evidence = gap(g)
    np.testing.assert_array_equal(evidence.counts[0, 0], [3, 0, 0])
    np.testing.assert_array_equal(evidence.counts[1, 0], evidence.counts[0, 0])
    assert evidence.endpoint_cell_count[0, 0] == 2
    assert evidence.gap_defined[0, 0] and not evidence.gap_defined[0, 1]
    assert evidence.fractions[0, 0, 0] == 1
    assert not evidence.physical_connectivity


def test_occupied_precedes_free_across_five_frames_and_order_invariant():
    origins = np.repeat([[.125, .125, .125]], 5, axis=0)
    endpoints = np.asarray([[1.125, .125, .125], [.625, .125, .125], [1.125, .125, .125],
                            [1.125, .125, .125], [1.125, .125, .125]])
    frames = np.arange(5)
    a = grid(origins, endpoints, frames)
    b = grid(origins[::-1], endpoints[::-1], frames[::-1])
    assert a.content_sha256 == b.content_sha256
    assert a.source_geometry_sha256 != b.source_geometry_sha256
    assert a.free_frame_bits[at((.375, .125, .125))] == 31
    assert a.occupied_frame_bits[at((.625, .125, .125))] == 2
    assert a.state[at((.625, .125, .125))] == OCCUPIED
    np.testing.assert_array_equal(gap(a).counts[0, 0], [2, 1, 0])


@pytest.mark.parametrize("end", [(11., .125, .125), (10., .125, .125), (100., .125, .125)])
def test_roi_external_return_preserves_free_but_never_creates_clipped_hit(end):
    g = grid(endpoints=(end,))
    assert not g.occupied_frame_bits.any()
    assert g.state[79, 40, 40] == FREE
    assert g.state[39, 40, 40] == UNKNOWN


def test_segment_entering_roi_from_outside_and_reverse_exact_cells():
    a, b = (-11., .125, .125), (11., .125, .125)
    left = grid((a,), (b,)); right = grid((b,), (a,))
    np.testing.assert_array_equal(left.state, right.state)
    assert (left.state == FREE).sum() == 80
    cells, ambiguity = _segment_cells(a, b, 1e-12)
    reverse, reverse_ambiguity = _segment_cells(b, a, 1e-12)
    assert cells == reverse and ambiguity == reverse_ambiguity == set()


def test_no_return_invalid_nan_is_not_maxrange_free():
    g = grid(((float("nan"),) * 3,), ((float("nan"),) * 3,), valid=np.array([False]))
    assert not g.state.any() and g.ignored_ray_count == 1
    assert g.first_return_count == 0
    assert gap(g).counts[0, 0].tolist() == [0, 0, 3]


def test_zero_length_has_only_real_hit_no_free_volume():
    g = grid(endpoints=((.125, .125, .125),))
    assert not g.free_frame_bits.any() and (g.state == OCCUPIED).sum() == 1
    e = gap(g, right=(.125, .125, .125))
    assert not e.gap_defined.any() and not e.fractions.any()


def test_internal_plane_ray_is_ambiguous_not_arbitrary_one_sided_free():
    g = grid(((.125, 0., .125),), ((1.125, 0., .125),))
    assert not g.free_frame_bits.any()
    assert g.ambiguous_ray_count == 1
    assert g.state[44, 39, 40] == g.state[44, 40, 40] == OCCUPIED
    e = gap(g, (.125, 0., .125), (1.125, 0., .125))
    assert e.counts[0, 0].tolist() == [0, 0, 6]
    assert e.ambiguous_cell_count[0, 0] == 6


def test_corner_crossings_do_not_clear_zero_measure_neighbor_cells():
    g = grid(((.125, .125, .125),), ((1.125, 1.125, 1.125),))
    assert (g.state == FREE).sum() == 4
    assert g.state[40, 41, 40] == UNKNOWN
    forward, ambiguous = _segment_cells((.125,) * 3, (1.125,) * 3, 1e-12)
    backward, ambiguous_back = _segment_cells((1.125,) * 3, (.125,) * 3, 1e-12)
    assert forward == backward and ambiguous == ambiguous_back == set()


def test_hit_exactly_on_internal_boundary_conservatively_marks_incident_cells():
    g = grid(endpoints=((1., .125, .125),))
    assert g.state[43, 40, 40] == OCCUPIED and g.state[44, 40, 40] == OCCUPIED
    assert g.state[42, 40, 40] == FREE


def test_outer_face_ray_does_not_clear_cube_interior():
    g = grid(((10., .125, .125),), ((10., 1.125, .125),))
    assert not g.state.any()


def test_stacked_free_ray_never_clears_unseen_lower_layer():
    g = grid(((.125, .125, 3.125),), ((1.125, .125, 3.125),))
    assert gap(g).counts[0, 0].tolist() == [0, 0, 3]
    assert gap(g, (.125, .125, 3.125), (1.125, .125, 3.125)).counts[0, 0].tolist() == [3, 0, 0]


def test_empty_grid_empty_patches_and_repeat_hash():
    a, b = grid((), ()), grid((), ())
    assert a.content_sha256 == b.content_sha256 and a.source_geometry_sha256 == b.source_geometry_sha256
    e = query_patch_gaps(a, np.empty((0, 3)), np.empty((0, 8), dtype=np.int64), np.empty((0, 8), dtype=bool))
    assert e.counts.shape == (0, 8, 3)


@pytest.mark.parametrize("change", ["valid_dtype", "frame5", "negative_frame", "float_frame", "nan_origin", "nan_hit", "shape"])
def test_invalid_input_rejected(change):
    o, p = np.array([[.125] * 3]), np.array([[1.125, .125, .125]])
    valid, frame = np.array([True]), np.array([0])
    if change == "valid_dtype": valid = np.array([1])
    elif change == "frame5": frame = np.array([5])
    elif change == "negative_frame": frame = np.array([-1])
    elif change == "float_frame": frame = np.array([0.])
    elif change == "nan_origin": o[0, 0] = np.nan
    elif change == "nan_hit": p[0, 0] = np.nan
    else: p = p[:, :2]
    with pytest.raises(ValueError):
        build_surface_ray_grid(o, p, valid, frame)


@pytest.mark.parametrize("change", ["state", "bits", "numerical_bound", "physical"])
def test_grid_tampering_rejected(change):
    g = grid()
    if change == "state": g = replace(g, state=np.zeros(SHAPE, dtype=np.uint8))
    elif change == "bits": g = replace(g, free_frame_bits=np.full(SHAPE, 255, dtype=np.uint8))
    elif change == "numerical_bound": g = replace(g, numerical_bound_m=1.)
    else: g = replace(g, physical_connectivity=True)
    with pytest.raises(ValueError): gap(g)


@pytest.mark.parametrize("change", ["self", "duplicate", "oob", "mask", "nan", "outside"])
def test_bad_gap_neighbors_or_centers_rejected(change):
    centers = np.array([[.125] * 3, [1.125, .125, .125]])
    index = np.full((2, 8), -1, dtype=np.int64); index[0, 0] = 1
    if change == "self": index[0, 0] = 0
    elif change == "duplicate": index[0, 1] = 1
    elif change == "oob": index[0, 0] = 2
    elif change == "nan": centers[0, 0] = np.nan
    elif change == "outside": centers[0, 0] = 11.
    valid = index >= 0
    if change == "mask": valid[:] = False
    with pytest.raises(ValueError): query_patch_gaps(grid(), centers, index, valid)


@pytest.mark.parametrize("origin,endpoint", [
    ((-9.875, -9.625, .125), (9.625, 8.875, .625)),
    ((.125, .125, .125), (9.875, 9.875, 9.875)),
    ((8.625, -7.875, .375), (-6.125, 4.875, .375)),
])
def test_crossed_cells_equal_independent_exact_rational_plane_oracle(origin, endpoint):
    # Independent Python Fraction arithmetic, not the implementation's numpy
    # midpoint computations. Endpoints are strictly within the grid.
    p = [Fraction(x) for x in origin]; q = [Fraction(x) for x in endpoint]
    d = [b - a for a, b in zip(p, q)]
    times = {Fraction(0), Fraction(1)}
    for axis in range(3):
        if d[axis]:
            for k in range(81):
                crossing = (Fraction(k, 4) - 10 - p[axis]) / d[axis]
                if 0 < crossing < 1:
                    times.add(crossing)
    times = sorted(times); exact_cells = set()
    for a, b in zip(times, times[1:]):
        middle = (a + b) / 2
        indices = [int((4 * (x + middle * delta + 10)) // 1) for x, delta in zip(p, d)]
        exact_cells.add((indices[0] * 80 + indices[1]) * 80 + indices[2])
    actual, ambiguous = _segment_cells(origin, endpoint, 1e-12)
    reverse, reverse_ambiguous = _segment_cells(endpoint, origin, 1e-12)
    assert actual == reverse == exact_cells and ambiguous == reverse_ambiguous == set()


def test_float32_boundary_uncertainty_never_clears_an_arbitrary_side():
    origin = np.array([[.125, 1e-7, .125]], dtype=np.float32)
    endpoint = np.array([[1.125, 1e-7, .125]], dtype=np.float32)
    g = build_surface_ray_grid(origin, endpoint, np.array([True]), np.array([0]))
    assert g.numerical_bound_m >= float(np.spacing(np.float32(10.)))
    assert not g.free_frame_bits.any() and g.ambiguous_ray_count == 1


def test_maximum_population_and_finite_bound_are_explicit():
    with pytest.raises(ValueError, match="maximum"):
        build_surface_ray_grid(np.zeros((57601, 3)), np.zeros((57601, 3)),
                               np.zeros(57601, dtype=bool), np.zeros(57601, dtype=np.int64))
    o = np.zeros((1, 3), dtype=np.float32)
    p = np.full((1, 3), np.finfo(np.float32).max, dtype=np.float32)
    with pytest.raises(ValueError, match="numerical bound"):
        build_surface_ray_grid(o, p, np.array([True]), np.array([0]))
