"""Synthetic patches only; no research payloads or checkpoints."""
from dataclasses import fields

import numpy as np
import pytest

from mtare_topo.representation.gse_surface_patches_v1 import extract_surface_patches, build_patch_neighbors


def cloud():
    points = np.array([[x, y, 1.1] for x in (.05, .15, .25) for y in (.05, .15, .25)]
                      + [[x, y, 2.1] for x in (1.05, 1.15, 1.25) for y in (.05, .15, .25)])
    return points, np.ones(len(points), dtype=bool), np.arange(len(points), dtype=np.int64) % 5


def test_exact_permutation_invariance_and_provenance_mapping():
    xyz, valid, frame = cloud(); p = extract_surface_patches(xyz, valid, frame)
    perm = np.random.default_rng(0).permutation(len(xyz)); q = extract_surface_patches(xyz[perm], valid[perm], frame[perm])
    for name in ("centers_m", "normals", "normal_valid", "normal_uncertainty", "bounds_min_m", "bounds_max_m", "roughness_m", "point_count", "frame_support"):
        np.testing.assert_array_equal(getattr(p, name), getattr(q, name))
    np.testing.assert_array_equal(q.point_patch_index, p.point_patch_index[perm])
    assert len(p.centers_m) == 2 and p.normal_valid.all()
    np.testing.assert_allclose(np.abs(p.normals[:, 2]), 1.)
    assert p.frame_support.sum() == len(xyz)


def test_roi_return_not_clipped_and_unknown_ray_origin_not_invented():
    xyz = np.array([[1., 0., 0.], [15., 0., 0.], [np.nan, 0., 0.]])
    valid = np.array([True, True, False]); frame = np.array([0, 4, 2])
    p = extract_surface_patches(xyz, valid, frame)
    assert len(p.centers_m) == 1 and p.point_patch_index.tolist() == [0, -1, -1]
    assert p.ray_origins_m is None and p.ray_endpoints_m[:, 0].tolist() == [1., 15.]
    assert p.ray_frame_index.tolist() == [0, 4]
    origins = np.zeros_like(xyz); origins[1, 1] = 2.
    p = extract_surface_patches(xyz, valid, frame, ray_origins_m=origins)
    np.testing.assert_array_equal(p.ray_origins_m, origins[:2])
    relations = build_patch_neighbors(p)
    assert not relations.physical_connectivity
    assert relations.ray_evidence[..., 2].all()


def test_degenerate_cells_kept_and_empty_observation_safe():
    xyz = np.array([[.1, .1, .1], [.2, .1, .1], [.3, .1, .1]])
    p = extract_surface_patches(xyz, np.ones(3, dtype=bool), np.arange(3))
    assert len(p.centers_m) == 1 and not p.normal_valid[0]
    assert p.normal_uncertainty[0] == 1 and not p.normals.any()
    empty = extract_surface_patches(xyz, np.zeros(3, dtype=bool), np.arange(3))
    assert empty.centers_m.shape == (0, 3)
    assert build_patch_neighbors(empty).neighbor_index.shape == (0, 8)


def test_capacity_never_drops_patches():
    xyz, valid, frame = cloud()
    with pytest.raises(OverflowError, match="never truncate"):
        extract_surface_patches(xyz, valid, frame, max_patches=1)
    with pytest.raises(ValueError): extract_surface_patches(xyz, valid, frame, max_patches=True)


def test_actual4096_capacity_overflow_and_sparse_memory_shape():
    xyz = np.array([(x, y, z) for x in np.arange(-4.9, 5., .5)
                    for y in np.arange(-4.9, 5., .5) for z in np.arange(-4.9, 5., .5)])[:4097]
    with pytest.raises(OverflowError):
        extract_surface_patches(xyz, np.ones(len(xyz), dtype=bool), np.zeros(len(xyz), dtype=np.int64))
    p = extract_surface_patches(xyz[:4096], np.ones(4096, dtype=bool), np.zeros(4096, dtype=np.int64))
    r = build_patch_neighbors(p)
    assert len(p.centers_m) == 4096 and r.neighbor_index.shape == (4096, 8)
    assert r.valid.all() and (r.neighbor_index != np.arange(4096)[:, None]).all()


def test_sparse_neighbors_are_3d_and_layout_sensitive():
    xyz = np.array([[.1, .1, .1], [1.1, .1, .1], [.1, .1, 3.1]])
    p = extract_surface_patches(xyz, np.ones(3, dtype=bool), np.arange(3))
    r = build_patch_neighbors(p)
    assert r.neighbor_index.shape == (3, 8) and (r.valid.sum(1) == 2).all()
    nearest = r.neighbor_index[0, 0]
    np.testing.assert_array_equal(p.centers_m[nearest], xyz[1])
    assert r.relative_xyz_m[0, 1, 2] == 3.
    assert not r.normal_pair_valid.any()  # singletons never pretend a normal
    q = extract_surface_patches(xyz + np.array([0., 0., 1.]), np.ones(3, dtype=bool), np.arange(3))
    np.testing.assert_allclose(build_patch_neighbors(q).relative_xyz_m, r.relative_xyz_m)


def test_rotation_of_points_changes_plane_normals_not_equivariance_claim():
    xyz, valid, frame = cloud(); p = extract_surface_patches(xyz, valid, frame)
    rotation = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
    q = extract_surface_patches(xyz @ rotation.T, valid, frame)
    np.testing.assert_allclose(q.centers_m, p.centers_m @ rotation.T)
    np.testing.assert_allclose(np.abs(q.normals), np.abs(p.normals @ rotation.T), atol=1e-12)
    # A generic small rotation crosses fixed voxel boundaries: never advertise
    # exact rotation-equivariance of the grouping algorithm.
    near = np.array([[.49, .01, 1.], [.49, .04, 1.], [.49, .18, 1.]])
    angle = -.2; yaw = np.array([[np.cos(angle), -np.sin(angle), 0.], [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]])
    a = extract_surface_patches(near, np.ones(3, dtype=bool), np.arange(3))
    b = extract_surface_patches(near @ yaw.T, np.ones(3, dtype=bool), np.arange(3))
    assert len(a.centers_m) != len(b.centers_m)


@pytest.mark.parametrize("mutation", ["nonfinite", "frame_future", "frame_float", "mask_int"])
def test_invalid_inputs_fail(mutation):
    xyz, valid, frame = cloud()
    if mutation == "nonfinite": xyz[0, 0] = np.nan
    if mutation == "frame_future": frame[0] = 5
    if mutation == "frame_float": frame = frame.astype(float)
    if mutation == "mask_int": valid = valid.astype(int)
    with pytest.raises(ValueError): extract_surface_patches(xyz, valid, frame)
