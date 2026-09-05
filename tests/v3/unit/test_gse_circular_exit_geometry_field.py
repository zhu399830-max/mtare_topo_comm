from __future__ import annotations

import numpy as np
import pytest

from mtare_topo.representation.gse_circular_exit_geometry_field import (
    circular_component_count,
    circular_roll_field,
    decode_exit_geometry_peaks,
    encode_exit_geometry_peaks,
    heading_unit_to_bearing_bins,
)
from execute_gse_circular_exit_geometry_field_teacher_export_v1 import _partition


def _inputs(headings: tuple[float, ...]) -> dict[str, np.ndarray]:
    slots = 6
    angle = np.radians(np.asarray(headings, dtype=np.float64))
    unit = np.zeros((1, slots, 2), dtype=np.float32)
    unit[0, :len(angle), 0] = np.sin(angle)
    unit[0, :len(angle), 1] = np.cos(angle)
    mask = np.zeros((1, slots), dtype=bool)
    mask[0, :len(angle)] = True
    width = np.zeros((1, slots), dtype=np.float32)
    width[0, :len(angle)] = np.arange(1, len(angle) + 1)
    width_mask = mask.copy()
    profile = np.zeros((1, slots, 4), dtype=np.float32)
    for slot in range(len(angle)):
        profile[0, slot] = slot + np.arange(4)
    return {"exit_mask": mask, "heading_unit": unit, "opening_width_m": width, "width_valid_mask": width_mask, "vertical_profile_m": profile}


def test_peak_field_round_trip_preserves_geometry() -> None:
    field = encode_exit_geometry_peaks(**_inputs((359.7, 90.4, 181.0)))
    decoded = decode_exit_geometry_peaks(field)[0]
    assert np.allclose(np.sort(decoded["heading_deg"]), np.asarray([90.4, 181.0, 359.7]), atol=1e-5)
    assert sorted(decoded["opening_width_m"].tolist()) == [1.0, 2.0, 3.0]
    assert circular_component_count(field.presence).tolist() == [3]


def test_peak_field_rejects_same_bin_collision() -> None:
    with pytest.raises(RuntimeError, match="collide"):
        encode_exit_geometry_peaks(**_inputs((10.1, 10.8)))


def test_integer_rotation_is_exact_circular_roll() -> None:
    field = encode_exit_geometry_peaks(**_inputs((10.4, 120.2)))
    rolled = circular_roll_field(field, 10)
    assert np.array_equal(rolled.presence, np.roll(field.presence, 10, axis=1))
    original = decode_exit_geometry_peaks(field)[0]["heading_deg"]
    rotated = decode_exit_geometry_peaks(rolled)[0]["heading_deg"]
    assert np.allclose(np.sort(rotated), np.sort((original + 20.0) % 360.0), atol=1e-5)


def test_circular_component_count_merges_wraparound() -> None:
    mask = np.zeros((2, 180), dtype=bool)
    mask[0, [179, 0, 1, 20]] = True
    mask[1] = True
    assert circular_component_count(mask).tolist() == [2, 1]


def test_half_bin_boundary_uses_one_canonical_float64_conversion() -> None:
    value = _inputs((357.0,))
    angle, bins, residual = heading_unit_to_bearing_bins(value["heading_unit"])
    field = encode_exit_geometry_peaks(**value)
    active = np.flatnonzero(field.presence[0])
    assert active.tolist() == [int(bins[0, 0])]
    recovered = (active[0] * 2.0 + field.heading_residual_deg[0, active[0]]) % 360.0
    error = abs((recovered - angle[0, 0] + 180.0) % 360.0 - 180.0)
    assert error < 1e-5
    assert abs(float(field.heading_residual_deg[0, active[0]]) - residual[0, 0]) < 1e-5


def test_teacher_export_partition_rejects_test_worlds() -> None:
    assert _partition("S01_flat_tree_small_C01") == "fit"
    assert _partition("S01_flat_tree_small_C08") == "selection"
    with pytest.raises(RuntimeError, match="forbidden"):
        _partition("S01_flat_tree_small_C09")
