"""Frozen geometry, SDF and metric helpers for CPU--Gazebo LiDAR parity."""

from __future__ import annotations

import math
import struct
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from mtare_topo.data.cano_sensor_smoke import (
    ELEVATION_DEG,
    MAX_RANGE_M,
    NEAR_RANGE_M,
    _node_maps,
    _nearest_spline_location,
    _polyline_distances,
    interpolate_polyline,
)


AZIMUTH_DEG = np.arange(720, dtype=np.float64) * 0.5
ROLE_COUNTS = {
    "S01_flat_tree_small_C01": {"tunnel_interior": 3, "junction_transition": 2, "terminal_approach": 3},
    "S06_3d_branch_medium_C01": {"tunnel_interior": 3, "junction_transition": 3, "terminal_approach": 2},
    "S10_3d_complex_C01": {"tunnel_interior": 2, "junction_transition": 3, "terminal_approach": 3},
}


def _yaw_deg(tangent: np.ndarray) -> float:
    return float(math.degrees(math.atan2(float(tangent[1]), float(tangent[0]))) % 360.0)


def _record(
    parent_id: str,
    role: str,
    index: int,
    axis_xyz: np.ndarray,
    tangent: np.ndarray,
    fta_distance_m: float,
    detail: str,
) -> dict[str, Any]:
    sensor = np.asarray(axis_xyz, dtype=np.float64).copy()
    sensor[2] += float(fta_distance_m) + 1.0
    return {
        "pose_id": f"{parent_id}__{role}_{index:02d}",
        "parent_id": parent_id,
        "role": role,
        "axis_xyz_m": np.asarray(axis_xyz, dtype=np.float64).tolist(),
        "sensor_xyz_m": sensor.tolist(),
        "yaw_deg": _yaw_deg(tangent),
        "selection_detail": detail,
    }


def select_parity_poses(
    parent_id: str,
    graph: Mapping[str, Any],
    spline_document: Mapping[str, Any],
    fta_distance_m: float,
) -> list[dict[str, Any]]:
    """Select the predeclared eight train-only poses without sensor feedback."""

    if parent_id not in ROLE_COUNTS:
        raise ValueError(f"unapproved parity parent: {parent_id}")
    wanted = ROLE_COUNTS[parent_id]
    splines = {
        int(item["tunnel_id"]): np.asarray(item["points"], dtype=np.float64)
        for item in spline_document["tunnels"]
    }
    nodes, adjacency = _node_maps(graph)
    event_xyz = np.asarray(
        [node["xyz"] for node in graph["nodes"] if int(node["degree"]) != 2],
        dtype=np.float64,
    )
    result: list[dict[str, Any]] = []

    # Deterministic coverage-first interior candidates across tunnel IDs.
    candidates: list[tuple[float, int, float, np.ndarray, np.ndarray]] = []
    for tunnel_id in sorted(splines):
        points = splines[tunnel_id]
        total = float(_polyline_distances(points)[-1])
        for fraction in (0.25, 0.5, 0.75):
            distance = fraction * total
            point, tangent = interpolate_polyline(points, distance)
            clearance = float(np.min(np.linalg.norm(event_xyz - point, axis=1)))
            if clearance >= 12.0:
                candidates.append((-clearance, tunnel_id, distance, point, tangent))
    candidates.sort(key=lambda value: (value[0], value[1], value[2]))
    chosen: list[np.ndarray] = []
    for negative_clearance, tunnel_id, distance, point, tangent in candidates:
        if any(float(np.linalg.norm(point - prior)) < 20.0 for prior in chosen):
            continue
        index = len(chosen)
        chosen.append(point)
        result.append(
            _record(
                parent_id,
                "tunnel_interior",
                index,
                point,
                tangent,
                fta_distance_m,
                f"tunnel={tunnel_id};arc_m={distance:.6f};event_clearance_m={-negative_clearance:.6f}",
            )
        )
        if len(chosen) == wanted["tunnel_interior"]:
            break
    if len(chosen) != wanted["tunnel_interior"]:
        raise ValueError(f"{parent_id}: insufficient frozen interior poses")

    junctions = sorted(
        (node for node in graph["nodes"] if int(node["degree"]) >= 3),
        key=lambda value: value["id"],
    )
    if len(junctions) < wanted["junction_transition"]:
        raise ValueError(f"{parent_id}: insufficient junction nodes")
    for index, node in enumerate(junctions[: wanted["junction_transition"]]):
        center = np.asarray(node["xyz"], dtype=np.float64)
        tunnel_id, distance = _nearest_spline_location(center, splines, node["incident_tunnel_ids"])
        point, tangent = interpolate_polyline(splines[tunnel_id], distance)
        neighbour = np.asarray(nodes[adjacency[node["id"]][0]]["xyz"], dtype=np.float64)
        if float(np.dot(tangent, neighbour - point)) < 0.0:
            tangent = -tangent
        result.append(
            _record(parent_id, "junction_transition", index, point, tangent, fta_distance_m, f"node={node['id']};degree={node['degree']}")
        )

    terminals = sorted(
        (node for node in graph["nodes"] if int(node["degree"]) == 1),
        key=lambda value: value["id"],
    )
    if len(terminals) < wanted["terminal_approach"]:
        raise ValueError(f"{parent_id}: insufficient terminal nodes")
    for index, node in enumerate(terminals[: wanted["terminal_approach"]]):
        endpoint = np.asarray(node["xyz"], dtype=np.float64)
        tunnel_id, endpoint_distance = _nearest_spline_location(endpoint, splines, node["incident_tunnel_ids"])
        total = float(_polyline_distances(splines[tunnel_id])[-1])
        sign = 1.0 if endpoint_distance < total / 2.0 else -1.0
        point, tangent = interpolate_polyline(splines[tunnel_id], endpoint_distance + sign * 3.0)
        tangent *= sign
        result.append(
            _record(parent_id, "terminal_approach", index, point, tangent, fta_distance_m, f"node={node['id']};inward_m=3")
        )

    if len(result) != 8 or len({item["pose_id"] for item in result}) != 8:
        raise ValueError(f"{parent_id}: parity pose count/identity failure")
    return result


def gazebo_world_sdf(mesh_uri: str | None, poses: Sequence[Mapping[str, Any]], *, analytic_box: bool = False) -> str:
    """Create a headless Gazebo Classic world with static organized CPU ray sensors."""

    if analytic_box == (mesh_uri is not None):
        raise ValueError("choose exactly one of analytic_box or mesh_uri")
    geometry = ""
    if analytic_box:
        walls = [
            ("xp", "10.05 0 0", "0.1 24 16"), ("xn", "-10.05 0 0", "0.1 24 16"),
            ("yp", "0 12.05 0", "20 0.1 16"), ("yn", "0 -12.05 0", "20 0.1 16"),
            ("zp", "0 0 8.05", "20 24 0.1"), ("zn", "0 0 -8.05", "20 24 0.1"),
        ]
        models = []
        for name, pose, size in walls:
            models.append(f'<model name="wall_{name}"><static>true</static><pose>{pose} 0 0 0</pose><link name="link"><collision name="collision"><geometry><box><size>{size}</size></box></geometry></collision></link></model>')
        geometry = "\n".join(models)
    else:
        geometry = f'<model name="cano_mesh"><static>true</static><link name="mesh_link"><collision name="mesh_collision"><geometry><mesh><uri>{mesh_uri}</uri><scale>1 1 1</scale></mesh></geometry></collision></link></model>'

    sensors = []
    for index, pose in enumerate(poses):
        x, y, z = (float(value) for value in pose["sensor_xyz_m"])
        yaw = math.radians(float(pose["yaw_deg"]))
        topic = f"/parity/lidar_{index:02d}"
        sensors.append(f'''
<model name="parity_sensor_{index:02d}"><static>true</static><pose>{x:.12g} {y:.12g} {z:.12g} 0 0 {yaw:.12g}</pose><link name="lidar_link">
<sensor type="ray" name="lidar"><always_on>true</always_on><update_rate>5</update_rate><visualize>false</visualize>
<ray><scan><horizontal><samples>720</samples><resolution>1</resolution><min_angle>0</min_angle><max_angle>6.274458660919615</max_angle></horizontal><vertical><samples>16</samples><resolution>1</resolution><min_angle>-0.2617993877991494</min_angle><max_angle>0.2617993877991494</max_angle></vertical></scan><range><min>0.3</min><max>51.0</max><resolution>0.001</resolution></range><noise><type>gaussian</type><mean>0</mean><stddev>0</stddev></noise></ray>
<plugin name="parity_velodyne_{index:02d}" filename="libgazebo_ros_velodyne_laser.so"><topicName>{topic}</topicName><frameName>parity_sensor_{index:02d}/lidar_link</frameName><organize_cloud>true</organize_cloud><min_range>0.3</min_range><max_range>50.0</max_range><gaussianNoise>0</gaussianNoise></plugin>
</sensor></link></model>''')
    return f'''<?xml version="1.0"?><sdf version="1.6"><world name="parity"><gravity>0 0 -9.81</gravity><physics type="ode"><max_step_size>0.001</max_step_size><real_time_update_rate>1000</real_time_update_rate></physics>{geometry}{''.join(sensors)}</world></sdf>\n'''


def decode_organized_cloud(data: bytes, width: int, height: int, point_step: int) -> tuple[np.ndarray, np.ndarray]:
    """Decode the frozen Velodyne plugin's azimuth-major organized cloud."""

    if (width, height, point_step) != (16, 720, 22):
        raise ValueError(f"unexpected PointCloud2 layout {(width, height, point_step)}")
    if len(data) != width * height * point_step:
        raise ValueError("PointCloud2 byte count mismatch")
    ranges = np.full((16, 720), MAX_RANGE_M, dtype=np.float32)
    valid = np.zeros((16, 720), dtype=np.uint8)
    for azimuth_index in range(720):
        for ring_index in range(16):
            offset = (azimuth_index * 16 + ring_index) * point_step
            x, y, z = struct.unpack_from("<fff", data, offset)
            ring = struct.unpack_from("<H", data, offset + 16)[0]
            if ring != ring_index:
                raise ValueError(f"ring mismatch at azimuth {azimuth_index}: {ring} != {ring_index}")
            if math.isfinite(x) and math.isfinite(y) and math.isfinite(z):
                distance = math.sqrt(x * x + y * y + z * z)
                if NEAR_RANGE_M < distance < MAX_RANGE_M:
                    observed_bin = int(round((math.degrees(math.atan2(y, x)) % 360.0) / 0.5)) % 720
                    if observed_bin != azimuth_index:
                        raise ValueError(f"azimuth mismatch {observed_bin} != {azimuth_index}")
                    ranges[ring_index, azimuth_index] = distance
                    valid[ring_index, azimuth_index] = 1
    return ranges, valid


def analytic_box_ranges() -> tuple[np.ndarray, np.ndarray]:
    """Closed-form ranges from the origin to box x=+-10,y=+-12,z=+-8."""

    azimuth, elevation = np.meshgrid(np.radians(AZIMUTH_DEG), np.radians(ELEVATION_DEG))
    directions = np.stack((np.cos(elevation) * np.cos(azimuth), np.cos(elevation) * np.sin(azimuth), np.sin(elevation)), axis=-1)
    limits = np.asarray([10.0, 12.0, 8.0], dtype=np.float64)
    candidates = np.full_like(directions, np.inf, dtype=np.float64)
    np.divide(
        limits,
        np.abs(directions),
        out=candidates,
        where=np.abs(directions) > 1e-12,
    )
    ranges = np.min(candidates, axis=-1).astype(np.float32)
    valid = ((ranges >= NEAR_RANGE_M) & (ranges <= MAX_RANGE_M)).astype(np.uint8)
    return ranges, valid


def parity_metrics(cpu_range: np.ndarray, cpu_valid: np.ndarray, gazebo_range: np.ndarray, gazebo_valid: np.ndarray) -> dict[str, Any]:
    if any(array.shape != (16, 720) for array in (cpu_range, cpu_valid, gazebo_range, gazebo_valid)):
        raise ValueError("parity arrays must all be 16x720")
    agreement = float(np.mean(cpu_valid == gazebo_valid))
    common = cpu_valid.astype(bool) & gazebo_valid.astype(bool)
    errors = np.abs(cpu_range[common].astype(np.float64) - gazebo_range[common].astype(np.float64))
    if errors.size == 0:
        raise ValueError("no common valid rays")
    horizontal_rows = np.flatnonzero((ELEVATION_DEG >= -5.0) & (ELEVATION_DEG <= 5.0))
    cpu_horizontal = float(np.min(cpu_range[horizontal_rows][cpu_valid[horizontal_rows].astype(bool)]))
    gazebo_horizontal = float(np.min(gazebo_range[horizontal_rows][gazebo_valid[horizontal_rows].astype(bool)]))
    metrics = {
        "valid_mask_agreement": agreement,
        "common_valid_count": int(errors.size),
        "range_mae_m": float(np.mean(errors)),
        "range_error_p95_m": float(np.quantile(errors, 0.95)),
        "range_error_p99_m": float(np.quantile(errors, 0.99)),
        "horizontal_minimum_range_difference_m": abs(cpu_horizontal - gazebo_horizontal),
    }
    metrics["passed"] = bool(
        metrics["valid_mask_agreement"] >= 0.98
        and metrics["range_mae_m"] <= 0.05
        and metrics["range_error_p95_m"] <= 0.10
        and metrics["range_error_p99_m"] <= 0.25
        and metrics["horizontal_minimum_range_difference_m"] <= 0.10
    )
    return metrics
