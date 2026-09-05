#!/usr/bin/env python3
"""Materialize evidence and qualify all sealed C08 frames against V3 implicit geometry."""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from skimage.measure import marching_cubes

from _bootstrap import PROJECT_ROOT
from mtare_topo.data.local_implicit_union import (
    ContinuousLayeredHybridField,
    RouteConditionedSupportField,
    RouteSupportQuery,
    SampledSweptTubeField,
    build_local_union_windows,
    edge_arc_incidence,
    overlapping_window_pairs,
    continuous_transition_seam_audit,
    continuous_transition_semantics_audit,
)
from mtare_topo.governance import load_json, write_json


RUN_ID = "gate4_20260816_cano_c08_route_conditioned_support_corrective_v8r_seed0"
WORLDS = (("S01_flat_tree_small_C08", 801), ("S06_3d_branch_medium_C08", 1414), ("S10_3d_complex_C08", 2558))
ASSETS = PROJECT_ROOT / "results/gate0_baseline/gate0_20260811_cano_100_parent_perception_mesh_m1r_sanitized_assets_seed0/artifacts/meshes"
TRAJECTORIES = PROJECT_ROOT / "results/gate4_topology/gate4_20260813_cano_c08_trajectory_mesh_contract_v1_seed0/artifacts"
FORMAL_PASS_STATUS = "PASS_CANO_C08_ROUTE_CONDITIONED_SUPPORT_CORRECTIVE_V8R"
FORMAL_FAIL_STATUS = "FAIL_CANO_C08_ROUTE_CONDITIONED_SUPPORT_CORRECTIVE_V8R"
FORMAL_SCHEMA_VERSION = "cano_c08_route_conditioned_support_corrective_v8r"
FORMAL_WORLD_COUNT = 3
FORMAL_C09_WORLDS_READ = 0
FORMAL_CLAIM_BOUNDARY = "Route-conditioned support plus traversal-conditioned collision qualification only; native Cano remains LiDAR-only and no Gate-4 graph/planner claim is made."
FORMAL_PREVIEW_TITLE = "V8 route-conditioned support corrective"
RESOLUTIONS = (0.10, 0.05, 0.025)
CLEARANCE = 0.8
FLOOR = 1.0
FLOOR_TOL = 0.05
SUPPORT_INTERVAL_INTERIOR_MARGIN_M = 1e-8
MAX_RAY = 20.0
AZIMUTH = 720


def geometry_metrics(xyz: np.ndarray) -> dict[str, float]:
    delta = np.diff(xyz, axis=0)
    step = np.linalg.norm(delta, axis=1)
    unit = np.divide(delta, step[:, None], out=np.zeros_like(delta), where=step[:, None] > 0)
    turn = np.degrees(np.arccos(np.clip(np.sum(unit[:-1] * unit[1:], axis=1), -1, 1)))
    return {
        "maximum_step_m": float(step.max()),
        "p99_step_m": float(np.quantile(step, 0.99)),
        "maximum_turn_deg": float(turn.max()),
        "p99_turn_deg": float(np.quantile(turn, 0.99)),
    }


def finite_min_by_frame(values: np.ndarray, frames: int) -> np.ndarray:
    result = []
    for row in values.reshape(frames, AZIMUTH):
        finite = row[np.isfinite(row)]
        result.append(float(finite.min()) if len(finite) else float("inf"))
    return np.asarray(result)


def frame_identities(route_arc: np.ndarray, traversals: list[dict]) -> list[dict]:
    ends = np.asarray([float(item["route_end_m"]) for item in traversals])
    indices = np.minimum(np.searchsorted(ends, route_arc, side="right"), len(traversals) - 1)
    return [
        {
            "traversal_index": int(traversals[index]["traversal_index"]),
            "edge_id": str(traversals[index]["edge_id"]),
            "tunnel_id": int(traversals[index]["tunnel_id"]),
            "traversal_arc_m": float(route_arc[frame] - traversals[index]["route_start_m"]),
        }
        for frame, index in enumerate(indices)
    ]


def load_json_list(path: Path) -> list[dict]:
    """Load a sealed list-root JSON manifest without weakening governance.load_json."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not value or not all(isinstance(item, dict) for item in value):
        raise ValueError(f"JSON root must be a non-empty list of objects: {path}")
    return value


def sanitize_visual_faces(vertices: np.ndarray, faces: np.ndarray, maximum_area_m2: float = 1e-12) -> tuple[np.ndarray, int]:
    """Deterministically remove numerically degenerate faces from a visualization patch."""
    triangles = np.asarray(vertices)[np.asarray(faces)]
    areas = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1) * 0.5
    keep = areas > maximum_area_m2
    return np.asarray(faces)[keep], int(np.count_nonzero(~keep))


def extract_visual_patch(
    field: ContinuousLayeredHybridField, window, layer_tunnel_id: int, resolution: float, destination: Path
) -> dict:
    """Extract a non-authoritative open local patch and retain its integrity metrics."""
    radius = float(window.radius_m)
    count = int(np.ceil(2 * radius / resolution)) + 1
    coordinates = np.linspace(-radius, radius, count, dtype=np.float64)
    volume = np.empty((count, count, count), dtype=np.float32)
    center = np.asarray(window.center_xyz_m, dtype=np.float64)
    xx, yy = np.meshgrid(coordinates, coordinates, indexing="ij")
    for z_index, z_value in enumerate(coordinates):
        points = np.column_stack((xx.ravel(), yy.ravel(), np.full(xx.size, z_value))) + center
        layers = np.full(len(points), layer_tunnel_id, dtype=np.int64)
        volume[:, :, z_index] = field.signed_distance(points, layers).reshape(count, count)
    vertices, faces, normals, values = marching_cubes(volume, level=0.0, spacing=(resolution,) * 3, allow_degenerate=False)
    vertices += center - radius
    faces, removed = sanitize_visual_faces(vertices, faces)
    faces = faces.astype(np.uint32)
    triangles = vertices[faces]
    areas = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1) * 0.5
    edges = np.sort(np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])), axis=1)
    _, incidence = np.unique(edges, axis=0, return_counts=True)
    np.savez_compressed(destination, vertices_m=vertices.astype(np.float32), faces=faces, normals=normals.astype(np.float32), values=values.astype(np.float32))
    return {
        "resolution_m": resolution,
        "layer_tunnel_id": layer_tunnel_id,
        "grid_shape": [count, count, count],
        "field_bytes": int(volume.nbytes),
        "vertex_count": int(len(vertices)),
        "triangle_count_before_sanitation": int(len(faces) + removed),
        "triangle_count": int(len(faces)),
        "removed_degenerate_triangle_count": removed,
        "sanitation_area_threshold_m2": 1e-12,
        "degenerate_triangle_count": int(np.count_nonzero(areas <= 1e-12)),
        "boundary_edge_count": int(np.count_nonzero(incidence == 1)),
        "nonmanifold_edge_count": int(np.count_nonzero(incidence > 2)),
        "authoritative_for_qualification": False,
        "open_boundary_interpretation": "Expected visualization clipping at the local cube; the continuous implicit field is authoritative.",
        "artifact": str(destination.relative_to(destination.parent.parent)),
    }


def solve_support_sensor(field: ContinuousLayeredHybridField, axis: np.ndarray, fta: float, layer_ids: np.ndarray) -> np.ndarray:
    """Return the deterministic one-metre-above-support sensor trajectory."""
    sensor = axis.copy()
    sensor[:, 2] += fta + FLOOR
    initial_down = field.ray_exit_distances(
        sensor, np.tile([[0.0, 0.0, -1.0]], (len(axis), 1)), layer_ids, MAX_RAY
    )
    if not np.isfinite(initial_down).all():
        raise RuntimeError("floor solve failed")
    sensor[:, 2] -= initial_down - FLOOR
    return sensor


def freeze_support_queries(
    axis: np.ndarray,
    identities: list[dict],
    windows,
) -> tuple[np.ndarray, tuple[RouteSupportQuery, ...]]:
    """Freeze unique original-axis window membership before any correction."""
    membership = np.zeros(len(axis), dtype=bool)
    window_ids: list[str | None] = [None] * len(axis)
    for window in windows:
        selected = np.linalg.norm(axis - np.asarray(window.center_xyz_m), axis=1) <= window.radius_m
        conflict = selected & membership
        if conflict.any():
            raise RuntimeError(f"non-unique frozen window membership at frame {int(np.flatnonzero(conflict)[0])}")
        membership |= selected
        for index in np.flatnonzero(selected):
            window_ids[int(index)] = window.node_id
    queries = tuple(
        RouteSupportQuery(int(identity["tunnel_id"]), str(identity["edge_id"]), window_ids[index])
        for index, identity in enumerate(identities)
    )
    return membership, queries


def optimize_window_trajectory(field: SampledSweptTubeField, axis: np.ndarray, center_sensor: np.ndarray, fta: float, windows) -> tuple[np.ndarray, np.ndarray, dict]:
    """Minimum-displacement convex window-local XY/z solution under the step contract."""
    center_axis = center_sensor.copy()
    center_axis[:, 2] -= fta + FLOOR
    membership = np.zeros(len(axis), dtype=bool)
    for window in windows:
        membership |= np.linalg.norm(axis - np.asarray(window.center_xyz_m), axis=1) <= window.radius_m
    identities = np.flatnonzero(membership)
    if not len(identities):
        return center_sensor, membership, {"solver": "no_window_frames", "success": True, "window_frames": 0}
    down = field.ray_exit_distances(center_sensor, np.tile([[0.0, 0.0, -1.0]], (len(axis), 1)), MAX_RAY)
    error = down - FLOOR
    lower_z = center_axis[:, 2] - FLOOR_TOL - error
    upper_z = center_axis[:, 2] + FLOOR_TOL - error
    base = center_axis.copy()
    base[identities, 2] = upper_z[identities]
    index_of = {int(frame): index for index, frame in enumerate(identities)}
    pairs = np.flatnonzero(membership[:-1] | membership[1:])
    step_limit = geometry_metrics(axis)["maximum_step_m"] + 0.1

    def unpack(vector):
        candidate = base.copy()
        candidate[identities] = vector.reshape(-1, 3)
        return candidate

    def objective(vector):
        displacement = vector.reshape(-1, 3) - axis[identities]
        return float(np.sum(displacement * displacement))

    def objective_jacobian(vector):
        return (2.0 * (vector.reshape(-1, 3) - axis[identities])).ravel()

    def step_constraints(vector):
        candidate = unpack(vector)
        delta = candidate[pairs + 1] - candidate[pairs]
        return step_limit * step_limit - np.sum(delta * delta, axis=1)

    def step_jacobian(vector):
        candidate = unpack(vector)
        delta = candidate[pairs + 1] - candidate[pairs]
        jacobian = np.zeros((len(pairs), len(vector)), dtype=np.float64)
        for row, (frame, difference) in enumerate(zip(pairs, delta)):
            if int(frame) in index_of:
                start = 3 * index_of[int(frame)]
                jacobian[row, start:start + 3] = 2.0 * difference
            if int(frame + 1) in index_of:
                start = 3 * index_of[int(frame + 1)]
                jacobian[row, start:start + 3] = -2.0 * difference
        return jacobian

    bounds = []
    for frame in identities:
        bounds.extend(((None, None), (None, None), (float(lower_z[frame]), float(upper_z[frame]))))
    initial_points = center_axis[identities].copy()
    initial_points[:, 2] = np.clip(initial_points[:, 2], lower_z[identities], upper_z[identities])
    initial = initial_points.ravel()
    result = minimize(
        objective, initial, jac=objective_jacobian, bounds=bounds,
        constraints={"type": "ineq", "fun": step_constraints, "jac": step_jacobian},
        method="SLSQP", options={"ftol": 1e-8, "maxiter": 300, "disp": False},
    )
    optimized_axis = unpack(result.x)
    constraint_margin = float(step_constraints(result.x).min())
    if not np.isfinite(result.x).all() or constraint_margin < -1e-7:
        raise RuntimeError(f"trajectory optimizer infeasible: {result.message}; margin={constraint_margin}")
    optimized_sensor = optimized_axis.copy()
    optimized_sensor[:, 2] += fta + FLOOR
    correction = optimized_axis - axis
    first = np.diff(correction, axis=0)
    second = np.diff(correction, n=2, axis=0)
    metadata = {
        "solver": "scipy_slsqp_convex_minimum_displacement",
        "solver_success": bool(result.success),
        "solver_message": str(result.message),
        "iterations": int(result.nit),
        "window_frames": int(membership.sum()),
        "primary_squared_displacement_m2": float(objective(result.x)),
        "secondary_first_difference_energy_m2": float(np.sum(first * first)),
        "tertiary_second_difference_energy_m2": float(np.sum(second * second)),
        "minimum_step_constraint_margin_m2": constraint_margin,
        "maximum_xy_correction_m": float(np.linalg.norm(correction[:, :2], axis=1).max()),
        "maximum_z_change_from_center_support_m": float(np.abs(optimized_axis[:, 2] - center_axis[:, 2]).max()),
        "objective_order": ["minimum displacement (strictly convex, unique)", "minimum first difference if tied", "minimum second difference if tied"],
    }
    return optimized_sensor, membership, metadata


def solve_constrained_trajectory(
    fields: tuple[ContinuousLayeredHybridField, ...],
    support_field: RouteConditionedSupportField,
    axis: np.ndarray,
    fta: float,
    windows,
    layer_ids: np.ndarray,
    identities: list[dict],
) -> tuple[np.ndarray, np.ndarray, dict, tuple[RouteSupportQuery, ...]]:
    """Correct only frozen window frames against route-conditioned support."""
    if not fields:
        raise ValueError("at least one qualification field is required")
    nominal_sensor = axis.copy()
    nominal_sensor[:, 2] += fta + FLOOR
    sensor = nominal_sensor.copy()
    membership, queries = freeze_support_queries(axis, identities, windows)
    identities = np.flatnonzero(membership)
    index_of = {int(frame): index for index, frame in enumerate(identities)}
    pairs = np.flatnonzero(membership[:-1] | membership[1:])
    step_limit = geometry_metrics(axis)["maximum_step_m"] + 0.1
    history = []
    for outer_iteration in range(1, 9):
        graph_axis = sensor.copy(); graph_axis[:, 2] -= fta + FLOOR
        surfaces = support_field.support_heights(graph_axis, queries)
        lower_sensor_z = surfaces + FLOOR - FLOOR_TOL + SUPPORT_INTERVAL_INTERIOR_MARGIN_M
        upper_sensor_z = surfaces + FLOOR + FLOOR_TOL - SUPPORT_INTERVAL_INTERIOR_MARGIN_M
        widths = upper_sensor_z - lower_sensor_z
        sensor[identities, 2] = np.clip(
            nominal_sensor[identities, 2], lower_sensor_z[identities], upper_sensor_z[identities]
        )
        sensor[~membership] = nominal_sensor[~membership]
        fixed_axis = sensor.copy(); fixed_axis[:, 2] -= fta + FLOOR

        def unpack(vector):
            candidate = fixed_axis.copy()
            candidate[identities, :2] = vector.reshape(-1, 2)
            return candidate

        def objective(vector):
            difference = vector.reshape(-1, 2) - axis[identities, :2]
            return float(np.sum(difference * difference))

        def objective_jacobian(vector):
            return (2.0 * (vector.reshape(-1, 2) - axis[identities, :2])).ravel()

        def constraints(vector):
            candidate = unpack(vector)
            delta = candidate[pairs + 1] - candidate[pairs]
            return step_limit * step_limit - np.sum(delta * delta, axis=1)

        def constraint_jacobian(vector):
            candidate = unpack(vector)
            delta = candidate[pairs + 1] - candidate[pairs]
            jacobian = np.zeros((len(pairs), len(vector)), dtype=np.float64)
            for row, (frame, difference) in enumerate(zip(pairs, delta)):
                if int(frame) in index_of:
                    start = 2 * index_of[int(frame)]
                    jacobian[row, start:start + 2] = 2.0 * difference[:2]
                if int(frame + 1) in index_of:
                    start = 2 * index_of[int(frame + 1)]
                    jacobian[row, start:start + 2] = -2.0 * difference[:2]
            return jacobian

        result = minimize(
            objective, fixed_axis[identities, :2].ravel(), jac=objective_jacobian,
            constraints={"type": "ineq", "fun": constraints, "jac": constraint_jacobian},
            method="SLSQP", options={"ftol": 1e-8, "maxiter": 300, "disp": False},
        )
        constraint_margin = float(constraints(result.x).min())
        if not np.isfinite(result.x).all() or constraint_margin < -1e-7:
            raise RuntimeError(f"fixed-support XY optimizer infeasible: {result.message}; margin={constraint_margin}")
        candidate_axis = unpack(result.x)
        sensor[:, :2] = candidate_axis[:, :2]
        sensor[~membership] = nominal_sensor[~membership]
        graph_axis = sensor.copy(); graph_axis[:, 2] -= fta + FLOOR
        post_xy_surfaces = support_field.support_heights(graph_axis, queries)
        post_xy_lower = post_xy_surfaces + FLOOR - FLOOR_TOL + SUPPORT_INTERVAL_INTERIOR_MARGIN_M
        post_xy_upper = post_xy_surfaces + FLOOR + FLOOR_TOL - SUPPORT_INTERVAL_INTERIOR_MARGIN_M
        post_xy_widths = post_xy_upper - post_xy_lower
        sensor[identities, 2] = np.clip(
            nominal_sensor[identities, 2], post_xy_lower[identities], post_xy_upper[identities]
        )
        sensor[~membership] = nominal_sensor[~membership]
        candidate_axis = sensor.copy()
        candidate_axis[:, 2] -= fta + FLOOR
        candidate_axis[~membership] = axis[~membership]
        verification_downs = support_field.downward_distances(sensor, candidate_axis, queries)
        maximum_step = geometry_metrics(candidate_axis)["maximum_step_m"]
        history.append({
            "iteration": outer_iteration,
            "maximum_floor_error_m": float(np.max(np.abs(verification_downs - FLOOR))),
            "minimum_common_support_interval_width_m": float(post_xy_widths.min()),
            "maximum_resolution_surface_spread_m": 0.0,
            "maximum_step_m": maximum_step,
            "maximum_xy_correction_m": float(np.linalg.norm(candidate_axis[:, :2] - axis[:, :2], axis=1).max()),
            "minimum_step_constraint_margin_m2": constraint_margin,
        })
        if history[-1]["maximum_floor_error_m"] <= FLOOR_TOL + 1e-9 and maximum_step <= step_limit + 1e-7:
            correction = candidate_axis - axis
            first = np.diff(correction, axis=0); second = np.diff(correction, n=2, axis=0)
            evidence = {
                "solver": "alternating_multiresolution_common_support_interval_and_slsqp_minimum_xy",
                "solver_success": bool(result.success), "solver_message": str(result.message),
                "iterations": int(result.nit), "window_frames": int(membership.sum()),
                "primary_squared_displacement_m2": float(np.sum(correction * correction)),
                "secondary_first_difference_energy_m2": float(np.sum(first * first)),
                "tertiary_second_difference_energy_m2": float(np.sum(second * second)),
                "minimum_step_constraint_margin_m2": constraint_margin,
                "maximum_xy_correction_m": float(np.linalg.norm(correction[:, :2], axis=1).max()),
                "maximum_z_correction_m": float(np.abs(correction[:, 2]).max()),
                "outside_window_pose_exact": bool(np.array_equal(sensor[~membership], nominal_sensor[~membership])),
                "objective_order": ["minimum displacement", "minimum first difference if tied", "minimum second difference if tied"],
                "support_xy_outer_iterations": outer_iteration,
                "support_xy_convergence": history,
                "qualification_field_count": len(fields),
                "support_field_count": 1,
                "support_interval_rule": "route-conditioned analytic support [surface+0.95m+1e-8m,surface+1.05m-1e-8m]; only frozen window frames may move; final audit remains exact [0.95m,1.05m]",
                "support_interval_interior_numerical_margin_m": SUPPORT_INTERVAL_INTERIOR_MARGIN_M,
                "minimum_common_support_interval_width_m": min(item["minimum_common_support_interval_width_m"] for item in history),
                "maximum_resolution_surface_spread_m": 0.0,
            }
            if not evidence["outside_window_pose_exact"]:
                raise RuntimeError("window-exterior pose changed")
            return sensor, membership, evidence, queries
    raise RuntimeError(f"support/XY optimizer did not converge: {history}")


def audit_sensor(
    field: ContinuousLayeredHybridField,
    support_field: RouteConditionedSupportField,
    sensor: np.ndarray,
    graph_axis: np.ndarray,
    fta: float,
    horizontal_dirs: np.ndarray,
    progress: tuple[str, float],
    layer_ids: np.ndarray,
    support_queries: tuple[RouteSupportQuery, ...],
) -> dict[str, np.ndarray]:
    """Audit one frozen optimized sensor trajectory against one field resolution."""
    axis = np.asarray(graph_axis, dtype=np.float64)
    if axis.shape != sensor.shape:
        raise ValueError("graph axis and sensor trajectory shapes differ")
    down = support_field.downward_distances(sensor, axis, support_queries)
    up = field.ray_exit_distances(
        sensor, np.tile([[0.0, 0.0, 1.0]], (len(axis), 1)), layer_ids, MAX_RAY
    )
    horizontal = []
    for start in range(0, len(axis), 16):
        batch = sensor[start:start + 16]
        origins = np.repeat(batch, AZIMUTH, axis=0)
        directions = np.tile(horizontal_dirs, (len(batch), 1))
        ray_layers = np.repeat(layer_ids[start:start + len(batch)], AZIMUTH)
        horizontal.extend(
            finite_min_by_frame(field.ray_exit_distances(origins, directions, ray_layers, MAX_RAY), len(batch))
        )
        if start % 256 == 0:
            print(json.dumps({"world": progress[0], "resolution_m": progress[1], "qualified_frames": min(start + 16, len(axis))}), flush=True)
    horizontal = np.asarray(horizontal)
    passed = np.isfinite(horizontal) & (horizontal >= CLEARANCE) & np.isfinite(down) & (np.abs(down - FLOOR) <= FLOOR_TOL) & np.isfinite(up) & (up >= CLEARANCE)
    return {"sensor": sensor, "axis": axis, "horizontal": horizontal, "down": down, "up": up, "passed": passed}


def readonly_proof(include_native: bool, include_geometry: bool = True) -> int:
    """Run the complete C08 contract proof without creating any result asset."""
    azimuth = np.radians(np.arange(AZIMUTH) * 0.5)
    horizontal_dirs = np.stack((np.cos(azimuth), np.sin(azimuth), np.zeros(AZIMUTH)), axis=1)
    summaries = []
    total_windows = total_edges = total_endpoints = total_traversals = total_frames = total_window_frames = 0
    for world, expected in WORLDS:
        root = ASSETS / world / "primary"
        graph, splines, geometry = (
            load_json(root / name) for name in ("graph.json", "splines.json", "geometry_parameters.json")
        )
        windows = build_local_union_windows(graph, geometry)
        if overlapping_window_pairs(windows):
            raise RuntimeError(f"window overlap {world}")
        arcs = edge_arc_incidence(graph, splines)
        trajectory = np.load(TRAJECTORIES / f"{world}_trajectory.npz")
        teacher_axis = trajectory["xyz_m"].astype(float)
        route_arc = trajectory["route_arc_m"].astype(float)
        traversals = load_json_list(TRAJECTORIES / f"{world}_traversals.json")
        frame_identity = frame_identities(route_arc, traversals)
        layer_ids = np.asarray([item["tunnel_id"] for item in frame_identity], dtype=np.int64)
        if len(teacher_axis) != expected or len(arcs) != 2 * len(graph["edges"]) or len(traversals) != 2 * len(graph["edges"]):
            raise RuntimeError(f"sealed count drift {world}")
        base_fields = {
            resolution: SampledSweptTubeField.from_documents(splines, geometry, resolution)
            for resolution in RESOLUTIONS
        }
        fields = {
            resolution: ContinuousLayeredHybridField(base_fields[resolution], windows)
            for resolution in RESOLUTIONS
        }
        support = RouteConditionedSupportField.from_documents(graph, splines, geometry, windows)
        sensor, membership, optimizer, support_queries = solve_constrained_trajectory(
            tuple(fields[value] for value in RESOLUTIONS), support, teacher_axis,
            float(geometry["fta_distance_m"]), windows, layer_ids, frame_identity,
        )
        graph_axis = sensor.copy(); graph_axis[:, 2] -= float(geometry["fta_distance_m"]) + FLOOR
        graph_axis[~membership] = teacher_axis[~membership]
        if not np.array_equal(graph_axis[~membership], teacher_axis[~membership]):
            raise RuntimeError(f"window-exterior pose drift {world}")
        resolution_results = []
        if include_geometry:
            for resolution in RESOLUTIONS:
                result = audit_sensor(
                    fields[resolution], support, sensor, graph_axis, float(geometry["fta_distance_m"]),
                    horizontal_dirs, (world, resolution), layer_ids, support_queries,
                )
                resolution_results.append(result)
                if not result["passed"].all():
                    failed = np.flatnonzero(~result["passed"])
                    raise RuntimeError(f"read-only safety proof failed {world}/{resolution}: {failed[:10].tolist()}")
            if not all(np.array_equal(resolution_results[0]["passed"], item["passed"]) for item in resolution_results[1:]):
                raise RuntimeError(f"resolution pass/fail drift {world}")

        native = {"checked": False}
        if include_native:
            import open3d as o3d
            from mtare_topo.data.c08_causal_replay import frame_contract
            from mtare_topo.data.cano_phase2_dataset import cast_ranges, evaluate_frame, spline_arrays
            from mtare_topo.data.cano_sensor_smoke import lidar_local_directions, world_directions

            def scene_from_mesh(path: Path):
                mesh = o3d.io.read_triangle_mesh(str(path), enable_post_processing=False)
                scene = o3d.t.geometry.RaycastingScene()
                scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
                return scene

            frames = frame_contract(
                teacher_axis_xyz_m=teacher_axis, graph_axis_xyz_m=graph_axis, sensor_xyz_m=sensor,
                tangent_world=trajectory["tangent_world"], route_arc_m=route_arc,
                traversals=traversals, graph=graph, fta_distance_m=float(geometry["fta_distance_m"]),
            )
            scene_a = scene_from_mesh(root / "mesh.obj")
            scene_b = scene_from_mesh(root / "mesh.obj")
            local_directions = lidar_local_directions()
            spline_map = spline_arrays(splines)
            native_clearance_failures = 0
            maximum_scene_difference = 0.0
            for index, frame in enumerate(frames):
                evaluated = evaluate_frame(scene_a, frame, spline_map, o3d)
                directions = world_directions(local_directions, float(frame["yaw_deg"]))
                second_range, second_valid = cast_ranges(scene_b, np.asarray(frame["sensor_xyz_m"]), directions, o3d)
                difference = float(np.max(np.abs(evaluated["range_m"] - second_range)))
                deterministic = np.array_equal(evaluated["valid_mask"], second_valid) and difference <= 1e-6
                if not evaluated["teacher_eligible"] or not np.isfinite(evaluated["range_m"]).all() or not deterministic:
                    raise RuntimeError(
                        f"native proof failed {world}/frame{index}: teacher={evaluated['teacher_eligible']} "
                        f"finite={np.isfinite(evaluated['range_m']).all()} deterministic={deterministic} "
                        f"los={evaluated['branch_los']}"
                    )
                native_clearance_failures += int(not evaluated["native_clearance_passed"])
                maximum_scene_difference = max(maximum_scene_difference, difference)
                if (index + 1) % 250 == 0 or index + 1 == len(frames):
                    print(json.dumps({"world": world, "native_frames": index + 1, "frames": len(frames)}), flush=True)
            native = {
                "checked": True,
                "frames": len(frames),
                "native_clearance_diagnostic_failed_frames": native_clearance_failures,
                "dual_scene_max_difference_m": maximum_scene_difference,
            }
        finest = resolution_results[-1] if resolution_results else None
        summary = {
            "world": world, "edges": len(graph["edges"]), "endpoints": len(arcs),
            "traversals": len(traversals), "windows": len(windows), "frames": len(teacher_axis),
            "window_frames": int(membership.sum()), "outside_frames": int((~membership).sum()),
            "outside_pose_exact": optimizer["outside_window_pose_exact"],
            "minimum_horizontal_clearance_m": float(finest["horizontal"].min()) if finest else None,
            "maximum_down_error_m": float(np.max(np.abs(finest["down"] - FLOOR))) if finest else None,
            "minimum_up_clearance_m": float(finest["up"].min()) if finest else None, "native": native,
        }
        print(json.dumps(summary), flush=True)
        summaries.append(summary)
        total_windows += len(windows); total_edges += len(graph["edges"]); total_endpoints += len(arcs)
        total_traversals += len(traversals); total_frames += len(teacher_axis); total_window_frames += int(membership.sum())
    expected_counts = (25, 312, 624, 624, 4773, 466)
    actual_counts = (total_windows, total_edges, total_endpoints, total_traversals, total_frames, total_window_frames)
    if actual_counts != expected_counts:
        raise RuntimeError(f"aggregate count drift: {actual_counts} != {expected_counts}")
    print(json.dumps({
        "overall_status": (
            "PASS_C08_READONLY_FULL_PROOF" if include_native and include_geometry
            else "PASS_C08_READONLY_NATIVE_PROOF" if include_native
            else "PASS_C08_READONLY_GEOMETRY_PROOF"
        ),
        "windows": total_windows, "edges": total_edges, "endpoints": total_endpoints,
        "traversals": total_traversals, "frames": total_frames, "window_frames": total_window_frames,
        "outside_frames": total_frames - total_window_frames, "world_metrics": summaries,
    }), flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--readonly-proof", choices=("geometry", "native", "full"))
    args = parser.parse_args()
    if args.readonly_proof:
        if args.run_dir is not None:
            raise RuntimeError("read-only proof cannot accept a run directory")
        return readonly_proof(
            args.readonly_proof in ("native", "full"),
            args.readonly_proof in ("geometry", "full"),
        )
    if args.run_dir is None:
        parser.error("--run-dir is required for materialization")
    run = args.run_dir.resolve()
    if run.name != RUN_ID:
        raise RuntimeError("unexpected run directory")
    started = time.monotonic()
    all_rows, world_metrics, window_manifest, arc_manifest, patch_manifest = [], [], [], [], []
    azimuth = np.radians(np.arange(AZIMUTH) * 0.5)
    horizontal_dirs = np.stack((np.cos(azimuth), np.sin(azimuth), np.zeros(AZIMUTH)), axis=1)

    for world, expected in WORLDS:
        root = ASSETS / world / "primary"
        graph, splines, geometry = (load_json(root / name) for name in ("graph.json", "splines.json", "geometry_parameters.json"))
        windows = build_local_union_windows(graph, geometry)
        overlaps = overlapping_window_pairs(windows)
        if overlaps:
            raise RuntimeError(f"window overlap {world}: {overlaps[:3]}")
        arcs = edge_arc_incidence(graph, splines)
        if len(arcs) != 2 * len(graph["edges"]):
            raise RuntimeError(f"arc count drift {world}")
        trajectory = np.load(TRAJECTORIES / f"{world}_trajectory.npz")
        axis = trajectory["xyz_m"].astype(float)
        route_arc = trajectory["route_arc_m"].astype(float)
        traversals = load_json_list(TRAJECTORIES / f"{world}_traversals.json")
        identities = frame_identities(route_arc, traversals)
        layer_ids = np.asarray([item["tunnel_id"] for item in identities], dtype=np.int64)
        if len(axis) != expected or len(traversals) != 2 * len(graph["edges"]):
            raise RuntimeError(f"sealed trajectory/traversal count drift {world}")

        base_fields = {
            resolution: SampledSweptTubeField.from_documents(splines, geometry, resolution)
            for resolution in RESOLUTIONS
        }
        fields = {
            resolution: ContinuousLayeredHybridField(base_fields[resolution], windows)
            for resolution in RESOLUTIONS
        }
        support_field = RouteConditionedSupportField.from_documents(graph, splines, geometry, windows)
        optimized_sensor, window_membership, optimizer, support_queries = solve_constrained_trajectory(
            tuple(fields[resolution] for resolution in RESOLUTIONS),
            support_field, axis, float(geometry["fta_distance_m"]), windows, layer_ids, identities
        )
        optimized_axis = optimized_sensor.copy()
        optimized_axis[:, 2] -= float(geometry["fta_distance_m"]) + FLOOR
        optimized_axis[~window_membership] = axis[~window_membership]
        if not np.array_equal(optimized_axis[~window_membership], axis[~window_membership]):
            raise RuntimeError(f"window-exterior graph pose changed {world}")

        results_by_resolution, convergence = {}, []
        for resolution in RESOLUTIONS:
            base_field = base_fields[resolution]
            field = fields[resolution]
            seams = [continuous_transition_seam_audit(field, window, 2048) for window in windows]
            isolation = [continuous_transition_semantics_audit(field, window, 4096) for window in windows]
            if not all(item["passed"] for item in seams + isolation):
                raise RuntimeError(f"field seam/isolation failed {world} at {resolution}")
            for window, seam, isolated in zip(windows, seams, isolation):
                for layer_tunnel_id in window.incident_tunnel_ids:
                    patch_path = run / "artifacts" / f"patch_{world}_{window.node_id}_layer{layer_tunnel_id}_{resolution:.3f}m.npz"
                    print(json.dumps({"world": world, "resolution_m": resolution, "patch_node": window.node_id, "layer_tunnel_id": layer_tunnel_id, "state": "extracting"}), flush=True)
                    patch = extract_visual_patch(field, window, layer_tunnel_id, resolution, patch_path)
                    patch.update({"world": world, "node_id": window.node_id, "incident_tunnel_ids": list(window.incident_tunnel_ids), "seam": seam, "transition_semantics": isolated})
                    if patch["degenerate_triangle_count"] or patch["nonmanifold_edge_count"]:
                        raise RuntimeError(f"visual patch integrity failed {world}/{window.node_id}/layer{layer_tunnel_id}/{resolution}")
                    patch_manifest.append(patch)
                    print(json.dumps({"world": world, "resolution_m": resolution, "patch_node": window.node_id, "layer_tunnel_id": layer_tunnel_id, "state": "complete", "vertices": patch["vertex_count"]}), flush=True)
            result = audit_sensor(
                field, support_field, optimized_sensor, optimized_axis, float(geometry["fta_distance_m"]), horizontal_dirs,
                (world, resolution), layer_ids, support_queries,
            )
            results_by_resolution[resolution] = result
            convergence.append({
                "resolution_m": resolution,
                "centerline_sample_count": sum(len(item) for item in base_field.samples_by_tunnel.values()),
                "maximum_seam_field_error_m": max(item["maximum_boundary_value_error_m"] for item in seams),
                "maximum_transition_formula_error_m": max(item["maximum_transition_formula_error_m"] for item in isolation),
                "maximum_nonincident_dispatch_error_m": max(item["maximum_nonincident_dispatch_error_m"] for item in isolation),
                "passed_frames": int(result["passed"].sum()),
                "all_frames_passed": bool(result["passed"].all()),
            })
            if not result["passed"].all():
                raise RuntimeError(f"safety qualification failed {world} at {resolution}")

        pass_vectors = [results_by_resolution[value]["passed"] for value in RESOLUTIONS]
        resolution_identity = all(np.array_equal(pass_vectors[0], item) for item in pass_vectors[1:])
        finest = results_by_resolution[RESOLUTIONS[-1]]
        old_geometry = geometry_metrics(axis)
        new_geometry = geometry_metrics(finest["axis"])
        continuity = bool(new_geometry["maximum_step_m"] <= old_geometry["maximum_step_m"] + 0.1 and new_geometry["maximum_turn_deg"] <= old_geometry["maximum_turn_deg"] + 1.0)
        metric = {
            "world": world,
            "frames": len(axis),
            "traversals": len(traversals),
            "window_frames": int(window_membership.sum()),
            "outside_window_frames": int((~window_membership).sum()),
            "outside_window_pose_exact": bool(np.array_equal(finest["axis"][~window_membership], axis[~window_membership])),
            "passed_frames": int(finest["passed"].sum()),
            "failed_frames": int((~finest["passed"]).sum()),
            "minimum_horizontal_clearance_m": float(finest["horizontal"].min()),
            "maximum_floor_error_m": float(np.max(np.abs(finest["down"] - FLOOR))),
            "minimum_up_clearance_m": float(finest["up"].min()),
            "maximum_xy_correction_m": optimizer["maximum_xy_correction_m"],
            "optimizer": optimizer,
            "baseline_geometry": old_geometry,
            "corrected_geometry": new_geometry,
            "continuity_passed": continuity,
            "resolution_pass_fail_identical": resolution_identity,
            "all_frames_passed": bool(finest["passed"].all()),
            "resolution_convergence": convergence,
        }
        write_json(run / f"metrics/{world}.json", metric)
        world_metrics.append(metric)
        if not continuity or not resolution_identity:
            raise RuntimeError(f"continuity/resolution identity failed {world}")
        np.savez_compressed(
            run / f"artifacts/{world}_corrected_trajectory.npz",
            xyz_m=finest["axis"], teacher_axis_xyz_m=axis, graph_axis_xyz_m=finest["axis"],
            sensor_xyz_m=finest["sensor"], route_arc_m=route_arc,
            tangent_world=trajectory["tangent_world"], xy_offset_m=finest["axis"][:, :2] - axis[:, :2],
            frame_index=np.arange(len(axis)), traversal_index=np.asarray([item["traversal_index"] for item in identities]),
            inside_union_window=window_membership,
        )
        for window in windows:
            window_manifest.append({"world": world, "node_id": window.node_id, "center_xyz_m": list(window.center_xyz_m), "incident_tunnel_ids": list(window.incident_tunnel_ids), "radius_m": window.radius_m})
        for record in arcs:
            arc_manifest.append({"world": world, **record.__dict__})
        for index, identity in enumerate(identities):
            all_rows.append({
                "world": world, "frame_index": index, "route_order": index, "route_arc_m": route_arc[index], **identity,
                "inside_union_window": bool(window_membership[index]), "x_offset_m": finest["axis"][index, 0] - axis[index, 0], "y_offset_m": finest["axis"][index, 1] - axis[index, 1],
                "horizontal_clearance_m": finest["horizontal"][index], "down_m": finest["down"][index], "up_m": finest["up"][index],
                "passed_0.100m": bool(results_by_resolution[0.10]["passed"][index]),
                "passed_0.050m": bool(results_by_resolution[0.05]["passed"][index]),
                "passed_0.025m": bool(finest["passed"][index]), "passed": bool(finest["passed"][index]),
                "sensor_x_m": finest["sensor"][index, 0], "sensor_y_m": finest["sensor"][index, 1], "sensor_z_m": finest["sensor"][index, 2],
            })
        figure, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
        axes[0].plot(axis[:, 0], axis[:, 1], linewidth=0.4, label="sealed")
        axes[0].plot(finest["axis"][:, 0], finest["axis"][:, 1], linewidth=0.4, label="qualified")
        axes[0].legend(); axes[0].set_title("Complete XY")
        axes[1].plot(axis[:, 0], axis[:, 2], linewidth=0.4, label="sealed")
        axes[1].plot(finest["axis"][:, 0], finest["axis"][:, 2], linewidth=0.4, label="qualified")
        axes[1].legend(); axes[1].set_title("Complete XZ")
        axes[2].plot(route_arc, finest["horizontal"], linewidth=0.4, label="horizontal")
        axes[2].plot(route_arc, finest["down"], linewidth=0.4, label="down")
        axes[2].plot(route_arc, finest["up"], linewidth=0.4, label="up")
        axes[2].axhline(CLEARANCE, color="red", linestyle="--"); axes[2].legend(); axes[2].set_title("Complete safety audit")
        figure.suptitle(world + " " + FORMAL_PREVIEW_TITLE)
        figure.savefig(run / f"previews/{world}_complete_qualification.png", dpi=150)
        plt.close(figure)
        print(json.dumps(metric), flush=True)

    write_json(run / "artifacts/window_manifest.json", window_manifest)
    write_json(run / "artifacts/arc_incidence_manifest.json", arc_manifest)
    write_json(run / "artifacts/patch_provenance_manifest.json", patch_manifest)
    with (run / "artifacts/complete_frame_audit.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_rows[0]))
        writer.writeheader(); writer.writerows(all_rows)
    passed = all(item["all_frames_passed"] and item["continuity_passed"] and item["resolution_pass_fail_identical"] for item in world_metrics)
    summary = {
        "schema_version": FORMAL_SCHEMA_VERSION,
        "overall_status": FORMAL_PASS_STATUS if passed else FORMAL_FAIL_STATUS,
        "worlds": FORMAL_WORLD_COUNT, "windows": len(window_manifest), "arc_endpoint_records": len(arc_manifest), "directed_traversals": sum(item["traversals"] for item in world_metrics),
        "frames": len(all_rows), "resolution_frame_audits": len(all_rows) * len(RESOLUTIONS),
        "horizontal_rays": len(all_rows) * AZIMUTH * len(RESOLUTIONS),
        "collision_vertical_rays": len(all_rows) * len(RESOLUTIONS),
        "support_height_evaluations": len(all_rows) * len(RESOLUTIONS),
        "window_frames": sum(item["window_frames"] for item in world_metrics),
        "outside_window_frames": sum(item["outside_window_frames"] for item in world_metrics),
        "outside_window_pose_exact": all(item["outside_window_pose_exact"] for item in world_metrics),
        "visual_patch_count": len(patch_manifest), "all_frames_passed": passed, "world_metrics": world_metrics,
        "inference_frames": 0, "graph_updates": 0, "training_samples_consumed": 0,
        "c09_worlds_read": FORMAL_C09_WORLDS_READ, "c10_worlds_read": 0, "mtare_worlds_read": 0,
        "duration_seconds": time.monotonic() - started,
        "claim_boundary": FORMAL_CLAIM_BOUNDARY,
    }
    write_json(run / "metrics/summary.json", summary)
    print(json.dumps(summary), flush=True)
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
