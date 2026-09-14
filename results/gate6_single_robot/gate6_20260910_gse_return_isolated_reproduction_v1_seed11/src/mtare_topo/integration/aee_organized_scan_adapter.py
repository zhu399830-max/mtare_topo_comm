"""Exact online adapter for the frozen AEE organized Gazebo LiDAR."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from mtare_topo.data.aee_sensor_operator_parity import (
    organized_aee_gazebo_points_to_ranges,
    resample_aee_gazebo_organized_azimuth,
)


@dataclass(frozen=True)
class AeeOrganizedScanAudit:
    input_points: int
    accepted_points: int
    unique_valid_cells: int
    raw_finite_returns: int
    raw_no_returns: int
    model_near_rejections: int
    model_far_rejections: int
    source_rings: int
    source_azimuth_samples: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def validate_aee_organized_pointcloud2(message: Any) -> None:
    """Reject any PointCloud2 layout that differs from the frozen plugin."""

    fields = {field.name: field for field in message.fields}
    required = {"x": (0, 7), "y": (4, 7), "z": (8, 7), "ring": (16, 4)}
    observed = {
        key: (int(fields[key].offset), int(fields[key].datatype))
        for key in required
        if key in fields
    }
    if int(message.width) != 16 or int(message.height) != 350:
        raise ValueError(
            f"AEE organized layout drift: width={message.width}, height={message.height}"
        )
    if int(message.point_step) != 22 or int(message.row_step) != 352:
        raise ValueError("AEE organized PointCloud2 stride drift")
    if observed != required or bool(message.is_bigendian) or bool(message.is_dense):
        raise ValueError(f"AEE organized PointCloud2 field contract drift: {observed}")
    if len(message.data) != 350 * 16 * 22:
        raise ValueError("AEE organized PointCloud2 byte-count drift")


def aee_organized_pointcloud2_to_range_image(
    message: Any,
    point_cloud2: Any,
) -> tuple[np.ndarray, np.ndarray, AeeOrganizedScanAudit]:
    """Decode physical beams and apply the exact frozen 350-to-720 operator."""

    validate_aee_organized_pointcloud2(message)
    records = list(
        point_cloud2.read_points(
            message,
            field_names=("x", "y", "z", "ring"),
            skip_nans=False,
        )
    )
    if len(records) != 350 * 16:
        raise ValueError("AEE organized PointCloud2 record-count drift")
    points = np.asarray([row[:3] for row in records], dtype=np.float64).reshape(350, 16, 3)
    rings = np.asarray([row[3] for row in records], dtype=np.int64).reshape(350, 16)
    _, _, source_range, source_valid, source_audit = organized_aee_gazebo_points_to_ranges(
        points, rings
    )
    model_range, model_valid = resample_aee_gazebo_organized_azimuth(
        source_range, source_valid
    )
    audit = AeeOrganizedScanAudit(
        input_points=source_audit.raw_records,
        accepted_points=source_audit.model_valid_returns,
        unique_valid_cells=int(np.count_nonzero(model_valid)),
        raw_finite_returns=source_audit.raw_finite_returns,
        raw_no_returns=source_audit.raw_no_returns,
        model_near_rejections=source_audit.model_near_rejections,
        model_far_rejections=source_audit.model_far_rejections,
        source_rings=source_audit.rings,
        source_azimuth_samples=source_audit.azimuth_samples,
    )
    return model_range, model_valid, audit


__all__ = [
    "AeeOrganizedScanAudit",
    "aee_organized_pointcloud2_to_range_image",
    "validate_aee_organized_pointcloud2",
]
