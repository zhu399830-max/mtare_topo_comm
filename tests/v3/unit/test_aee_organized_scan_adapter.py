from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from mtare_topo.integration.aee_organized_scan_adapter import (
    aee_organized_pointcloud2_to_range_image,
    validate_aee_organized_pointcloud2,
)


@dataclass
class Field:
    name: str
    offset: int
    datatype: int


class Message:
    width = 16
    height = 350
    point_step = 22
    row_step = 352
    is_bigendian = False
    is_dense = False
    fields = [Field("x", 0, 7), Field("y", 4, 7), Field("z", 8, 7), Field("ring", 16, 4)]
    data = bytes(350 * 16 * 22)


class PointCloud2:
    @staticmethod
    def read_points(message, *, field_names, skip_nans):
        assert message is not None
        assert field_names == ("x", "y", "z", "ring")
        assert skip_nans is False
        azimuth = np.linspace(-np.pi, np.pi, 350)
        elevation = np.deg2rad(np.arange(-15, 16, 2))
        for angle in azimuth:
            for ring, el in enumerate(elevation):
                radius = 10.0
                yield (
                    radius * np.cos(el) * np.cos(angle),
                    radius * np.cos(el) * np.sin(angle),
                    radius * np.sin(el),
                    ring,
                )


def test_exact_organized_adapter_produces_frozen_model_shape() -> None:
    ranges, valid, audit = aee_organized_pointcloud2_to_range_image(Message(), PointCloud2)
    assert ranges.shape == valid.shape == (16, 720)
    assert ranges.dtype == np.float32 and valid.dtype == np.uint8
    assert np.all(valid == 1)
    assert np.allclose(ranges, 10.0, atol=1e-5)
    assert audit.input_points == 5600
    assert audit.accepted_points == 5600
    assert audit.unique_valid_cells == 16 * 720


def test_layout_and_ring_drift_fail_closed() -> None:
    bad = Message()
    bad.width = 15
    with pytest.raises(ValueError, match="layout drift"):
        validate_aee_organized_pointcloud2(bad)

    class WrongRings(PointCloud2):
        @staticmethod
        def read_points(message, *, field_names, skip_nans):
            for index, row in enumerate(PointCloud2.read_points(message, field_names=field_names, skip_nans=skip_nans)):
                yield (*row[:3], 15 - row[3]) if index == 0 else row

    with pytest.raises(ValueError, match="ring order"):
        aee_organized_pointcloud2_to_range_image(Message(), WrongRings)
