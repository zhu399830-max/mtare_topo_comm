from __future__ import annotations

import math

import numpy as np

from execute_gse_structured_polar_teacher_feasibility_v1 import (
    assign_azimuth_bin,
    assign_elevation_band,
    circular_bin_distance,
)


def _xyz(azimuth_deg: float, elevation_deg: float = 0.0):
    azimuth = math.radians(azimuth_deg)
    elevation = math.radians(elevation_deg)
    return np.asarray([
        math.cos(elevation) * math.cos(azimuth),
        math.cos(elevation) * math.sin(azimuth),
        math.sin(elevation),
    ], dtype=np.float64)


def test_nearest_two_degree_bin_has_circular_boundary():
    xyz = np.stack((_xyz(0.75), _xyz(2.75), _xyz(359.75)))
    assert assign_azimuth_bin(xyz).tolist() == [0, 1, 0]
    assert circular_bin_distance(0, 179) == 1
    assert circular_bin_distance(2, 179) == 3


def test_integer_bin_rotation_is_exact_away_from_boundary():
    xyz = np.stack([_xyz(value) for value in (0.75, 91.1, 180.4, 271.3)])
    original = assign_azimuth_bin(xyz)
    angle = math.radians(34.0)
    rotation = np.asarray([
        [math.cos(angle), -math.sin(angle), 0.0],
        [math.sin(angle), math.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
    ])
    assert np.array_equal(assign_azimuth_bin(xyz @ rotation.T), (original + 17) % 180)


def test_elevation_band_centers_are_fixed():
    xyz = np.stack([_xyz(0.0, value) for value in (-15.0, -6.0, 2.0, 14.0)])
    assert assign_elevation_band(xyz).tolist() == [0, 1, 2, 3]
