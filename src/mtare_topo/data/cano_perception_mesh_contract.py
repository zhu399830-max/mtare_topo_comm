"""Pure audits for the Cano 100-parent perception-mesh M0 contract."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence

import numpy as np


SENTINEL_PARENT_IDS = (
    "S01_flat_tree_small_C01",
    "S02_3d_tree_small_C01",
    "S03_flat_unicyclic_small_C01",
    "S04_3d_unicyclic_small_C01",
    "S05_flat_branch_medium_C01",
    "S06_3d_branch_medium_C01",
    "S07_flat_loop_rich_C01",
    "S08_3d_loop_rich_C01",
    "S09_flat_complex_C01",
    "S10_3d_complex_C01",
)


def floor_support_contract(
    expected_floor_z_m: Sequence[float],
    observed_floor_z_m: Sequence[float],
    tolerance_m: float,
) -> dict[str, Any]:
    """Audit mesh floor first hits against generator-expected floor heights."""

    expected = np.asarray(expected_floor_z_m, dtype=np.float64).reshape(-1)
    observed = np.asarray(observed_floor_z_m, dtype=np.float64).reshape(-1)
    if expected.shape != observed.shape or len(expected) == 0 or tolerance_m <= 0:
        raise ValueError("floor support arrays/tolerance are invalid")
    finite = np.isfinite(observed)
    errors = np.where(finite, np.abs(observed - expected), np.inf)
    passed = finite & (errors <= float(tolerance_m))
    return {
        "sample_count": len(expected),
        "supported_count": int(np.count_nonzero(passed)),
        "unsupported_count": int(np.count_nonzero(~passed)),
        "support_fraction": float(np.mean(passed)),
        "maximum_supported_height_error_m": (
            float(np.max(errors[passed])) if np.any(passed) else None
        ),
        "unsupported_indices": np.flatnonzero(~passed).astype(int).tolist(),
        "passed": bool(np.all(passed)),
    }


def canonical_json_hash(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def select_train_sentinels(parents: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Select the predeclared rank-1 train parent from every recipe."""

    selected = [
        dict(parent)
        for parent in parents
        if parent.get("accepted_rank_in_recipe") == 1
    ]
    selected.sort(key=lambda item: str(item.get("recipe_stratum_id", "")))
    observed_ids = tuple(str(item.get("parent_id")) for item in selected)
    if observed_ids != SENTINEL_PARENT_IDS:
        raise ValueError(f"M0 sentinel identity mismatch: {observed_ids!r}")
    if any(item.get("split") != "train" for item in selected):
        raise ValueError("M0 sentinel selection must be train-only")
    recipes = [str(item.get("recipe_stratum_id")) for item in selected]
    if len(set(recipes)) != 10:
        raise ValueError("M0 requires exactly one sentinel per recipe")
    geometry_seeds = [item.get("reserved_geometry_seed") for item in selected]
    if (
        any(not isinstance(seed, int) or isinstance(seed, bool) for seed in geometry_seeds)
        or len(set(geometry_seeds)) != 10
    ):
        raise ValueError("M0 reserved geometry seeds must be ten unique integers")
    return selected


def mesh_array_audit(
    vertices: np.ndarray,
    triangles: np.ndarray,
    component_triangle_counts: Sequence[int],
    axis: np.ndarray,
    source_tunnel_ids: Sequence[int],
    spline_points: np.ndarray,
    *,
    aabb_margin_m: float = 10.0,
    minimum_largest_component_fraction: float = 0.999,
) -> dict[str, Any]:
    """Audit one mesh without changing or repairing its geometry."""

    vertices = np.asarray(vertices, dtype=np.float64)
    triangles = np.asarray(triangles)
    axis = np.asarray(axis, dtype=np.float64)
    spline_points = np.asarray(spline_points, dtype=np.float64)
    counts = np.asarray(component_triangle_counts, dtype=np.int64).reshape(-1)

    vertex_shape_valid = vertices.ndim == 2 and vertices.shape[1:] == (3,) and len(vertices) > 0
    triangle_shape_valid = triangles.ndim == 2 and triangles.shape[1:] == (3,) and len(triangles) > 0
    finite_vertices = bool(vertex_shape_valid and np.isfinite(vertices).all())
    integer_triangles = bool(triangle_shape_valid and np.issubdtype(triangles.dtype, np.integer))
    triangle_indices_valid = bool(
        integer_triangles
        and int(np.min(triangles)) >= 0
        and int(np.max(triangles)) < len(vertices)
    )
    degenerate_triangle_count = -1
    if finite_vertices and triangle_indices_valid:
        first = vertices[triangles[:, 1]] - vertices[triangles[:, 0]]
        second = vertices[triangles[:, 2]] - vertices[triangles[:, 0]]
        doubled_area = np.linalg.norm(np.cross(first, second), axis=1)
        degenerate_triangle_count = int(np.count_nonzero(doubled_area <= 1e-12))

    component_counts_valid = bool(
        len(counts) > 0 and np.all(counts > 0) and int(np.sum(counts)) == len(triangles)
    )
    largest_component_fraction = (
        float(np.max(counts) / np.sum(counts)) if component_counts_valid else 0.0
    )

    axis_shape_valid = axis.ndim == 2 and axis.shape[1:] == (9,) and len(axis) > 0
    finite_axis = bool(axis_shape_valid and np.isfinite(axis).all())
    positive_axis_radius = bool(finite_axis and np.all(axis[:, 6] > 0.0))
    observed_tunnel_ids = (
        {int(round(value)) for value in axis[:, 8]} if finite_axis else set()
    )
    expected_tunnel_ids = {int(value) for value in source_tunnel_ids}
    every_tunnel_has_axis = bool(expected_tunnel_ids and expected_tunnel_ids <= observed_tunnel_ids)

    spline_shape_valid = (
        spline_points.ndim == 2 and spline_points.shape[1:] == (3,) and len(spline_points) > 0
    )
    finite_splines = bool(spline_shape_valid and np.isfinite(spline_points).all())
    splines_inside_expanded_aabb = False
    if finite_vertices and finite_splines:
        lower = np.min(vertices, axis=0) - float(aabb_margin_m)
        upper = np.max(vertices, axis=0) + float(aabb_margin_m)
        splines_inside_expanded_aabb = bool(
            np.all(spline_points >= lower) and np.all(spline_points <= upper)
        )

    checks = {
        "nonempty_finite_vertices": finite_vertices,
        "nonempty_integer_index_valid_triangles": triangle_indices_valid,
        "zero_degenerate_triangles": degenerate_triangle_count == 0,
        "component_counts_cover_triangles": component_counts_valid,
        "largest_component_fraction_at_least_0_999": (
            largest_component_fraction >= minimum_largest_component_fraction
        ),
        "finite_axis_with_positive_radius": finite_axis and positive_axis_radius,
        "every_source_tunnel_has_axis": every_tunnel_has_axis,
        "finite_splines_inside_mesh_aabb_plus_10m": (
            finite_splines and splines_inside_expanded_aabb
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "vertex_count": int(len(vertices)) if vertices.ndim else 0,
        "triangle_count": int(len(triangles)) if triangles.ndim else 0,
        "degenerate_triangle_count": degenerate_triangle_count,
        "triangle_component_count": int(len(counts)),
        "largest_component_fraction": largest_component_fraction,
        "expected_tunnel_ids": sorted(expected_tunnel_ids),
        "axis_tunnel_ids": sorted(observed_tunnel_ids),
        "axis_row_count": int(len(axis)) if axis.ndim else 0,
        "spline_point_count": int(len(spline_points)) if spline_points.ndim else 0,
        "aabb_margin_m": float(aabb_margin_m),
    }


def zero_area_triangle_sanitation(
    vertices: np.ndarray,
    triangles: np.ndarray,
    *,
    maximum_doubled_area: float = 1e-12,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return a deterministic keep mask and evidence for zero-area sanitation."""

    vertices = np.asarray(vertices, dtype=np.float64)
    triangles = np.asarray(triangles)
    if vertices.ndim != 2 or vertices.shape[1:] != (3,) or not np.isfinite(vertices).all():
        raise ValueError("sanitation requires finite Nx3 vertices")
    if (
        triangles.ndim != 2
        or triangles.shape[1:] != (3,)
        or not np.issubdtype(triangles.dtype, np.integer)
        or len(triangles) == 0
        or int(np.min(triangles)) < 0
        or int(np.max(triangles)) >= len(vertices)
    ):
        raise ValueError("sanitation requires nonempty valid integer triangles")
    first = vertices[triangles[:, 1]] - vertices[triangles[:, 0]]
    second = vertices[triangles[:, 2]] - vertices[triangles[:, 0]]
    doubled_area = np.linalg.norm(np.cross(first, second), axis=1)
    remove = doubled_area <= float(maximum_doubled_area)
    keep = ~remove
    removed = doubled_area[remove]
    evidence = {
        "predicate": f"float64_doubled_area <= {float(maximum_doubled_area):.17g}",
        "maximum_doubled_area": float(maximum_doubled_area),
        "triangle_count_before": int(len(triangles)),
        "removed_triangle_count": int(np.count_nonzero(remove)),
        "triangle_count_after": int(np.count_nonzero(keep)),
        "removed_surface_area_sum_m2": float(np.sum(removed) / 2.0),
        "maximum_removed_doubled_area": float(np.max(removed)) if len(removed) else 0.0,
        "minimum_kept_doubled_area": float(np.min(doubled_area[keep])) if np.any(keep) else None,
    }
    return keep, evidence


def replay_pair_audit(
    primary: Mapping[str, Any], replay: Mapping[str, Any]
) -> dict[str, Any]:
    checks = {
        "parent_id_equal": primary.get("parent_id") == replay.get("parent_id"),
        "graph_identity_equal": primary.get("graph_identity") == replay.get("graph_identity"),
        "spline_identity_equal": primary.get("spline_identity") == replay.get("spline_identity"),
        "effective_geometry_parameters_equal": (
            primary.get("effective_geometry_parameter_sha256")
            == replay.get("effective_geometry_parameter_sha256")
        ),
        "mesh_sha256_equal": primary.get("mesh_sha256") == replay.get("mesh_sha256"),
        "primary_mesh_passed": primary.get("mesh_audit", {}).get("passed") is True,
        "replay_mesh_passed": replay.get("mesh_audit", {}).get("passed") is True,
    }
    return {"passed": all(checks.values()), "checks": checks}


def geometric_replay_pair_audit(
    primary: Mapping[str, Any],
    replay: Mapping[str, Any],
    *,
    axis_exact_equal: bool,
    primary_vertex_count: int,
    replay_vertex_count: int,
    primary_triangle_count: int,
    replay_triangle_count: int,
    primary_to_replay_nearest_vertex_maximum_m: float,
    replay_to_primary_nearest_vertex_maximum_m: float,
    aabb_endpoint_maximum_coordinate_difference_m: float,
    surface_area_relative_difference: float,
    nearest_vertex_limit_m: float = 0.75,
    aabb_limit_m: float = 0.5,
    surface_area_relative_limit: float = 0.01,
    triangle_count_relative_limit: float | None = None,
    vertex_count_relative_limit: float | None = None,
    primary_to_replay_surface_maximum_m: float | None = None,
    replay_to_primary_surface_maximum_m: float | None = None,
    point_to_surface_limit_m: float = 0.75,
) -> dict[str, Any]:
    """Audit native-mesh replay under Cano's documented post-Poisson noise bound."""

    triangle_count_denominator = max(
        abs(int(primary_triangle_count)), abs(int(replay_triangle_count)), 1
    )
    triangle_count_relative_difference = (
        abs(int(primary_triangle_count) - int(replay_triangle_count))
        / triangle_count_denominator
    )
    if triangle_count_relative_limit is None:
        triangle_count_check_name = "triangle_count_equal"
        triangle_count_passed = int(primary_triangle_count) == int(replay_triangle_count)
    else:
        triangle_count_check_name = (
            "triangle_count_relative_difference_within_0_01_percent"
        )
        triangle_count_passed = (
            triangle_count_relative_difference <= float(triangle_count_relative_limit)
        )

    vertex_count_denominator = max(
        abs(int(primary_vertex_count)), abs(int(replay_vertex_count)), 1
    )
    vertex_count_relative_difference = (
        abs(int(primary_vertex_count) - int(replay_vertex_count))
        / vertex_count_denominator
    )
    if vertex_count_relative_limit is None:
        vertex_count_check_name = "vertex_count_equal"
        vertex_count_passed = int(primary_vertex_count) == int(replay_vertex_count)
    else:
        vertex_count_check_name = (
            "vertex_count_relative_difference_within_0_01_percent"
        )
        vertex_count_passed = (
            vertex_count_relative_difference <= float(vertex_count_relative_limit)
        )

    checks = {
        "parent_id_equal": primary.get("parent_id") == replay.get("parent_id"),
        "graph_identity_equal": primary.get("graph_identity") == replay.get("graph_identity"),
        "spline_identity_equal": primary.get("spline_identity") == replay.get("spline_identity"),
        "operation_trace_equal": (
            primary.get("operation_trace_sha256") == replay.get("operation_trace_sha256")
        ),
        "effective_geometry_parameters_equal": (
            primary.get("effective_geometry_parameter_sha256")
            == replay.get("effective_geometry_parameter_sha256")
        ),
        "axis_exact_equal": bool(axis_exact_equal),
        vertex_count_check_name: vertex_count_passed,
        triangle_count_check_name: triangle_count_passed,
        "aabb_endpoint_maximum_coordinate_difference_within_0_5m": (
            float(aabb_endpoint_maximum_coordinate_difference_m) <= aabb_limit_m
        ),
        "surface_area_relative_difference_within_1_percent": (
            float(surface_area_relative_difference) <= surface_area_relative_limit
        ),
        "primary_mesh_passed": primary.get("mesh_audit", {}).get("passed") is True,
        "replay_mesh_passed": replay.get("mesh_audit", {}).get("passed") is True,
    }
    point_to_surface_enabled = (
        primary_to_replay_surface_maximum_m is not None
        and replay_to_primary_surface_maximum_m is not None
    )
    if point_to_surface_enabled:
        checks.update(
            {
                "primary_vertices_to_replay_surface_maximum_within_0_75m": (
                    float(primary_to_replay_surface_maximum_m)
                    <= point_to_surface_limit_m
                ),
                "replay_vertices_to_primary_surface_maximum_within_0_75m": (
                    float(replay_to_primary_surface_maximum_m)
                    <= point_to_surface_limit_m
                ),
            }
        )
    else:
        checks.update(
            {
                "primary_to_replay_nearest_vertex_maximum_within_0_75m": (
                    float(primary_to_replay_nearest_vertex_maximum_m)
                    <= nearest_vertex_limit_m
                ),
                "replay_to_primary_nearest_vertex_maximum_within_0_75m": (
                    float(replay_to_primary_nearest_vertex_maximum_m)
                    <= nearest_vertex_limit_m
                ),
            }
        )
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "metrics": {
            "primary_vertex_count": int(primary_vertex_count),
            "replay_vertex_count": int(replay_vertex_count),
            "vertex_count_absolute_difference": abs(
                int(primary_vertex_count) - int(replay_vertex_count)
            ),
            "vertex_count_relative_difference": float(
                vertex_count_relative_difference
            ),
            "primary_triangle_count": int(primary_triangle_count),
            "replay_triangle_count": int(replay_triangle_count),
            "triangle_count_absolute_difference": abs(
                int(primary_triangle_count) - int(replay_triangle_count)
            ),
            "triangle_count_relative_difference": float(
                triangle_count_relative_difference
            ),
            "primary_to_replay_nearest_vertex_maximum_m": float(
                primary_to_replay_nearest_vertex_maximum_m
            ),
            "replay_to_primary_nearest_vertex_maximum_m": float(
                replay_to_primary_nearest_vertex_maximum_m
            ),
            "primary_vertices_to_replay_surface_maximum_m": (
                None
                if primary_to_replay_surface_maximum_m is None
                else float(primary_to_replay_surface_maximum_m)
            ),
            "replay_vertices_to_primary_surface_maximum_m": (
                None
                if replay_to_primary_surface_maximum_m is None
                else float(replay_to_primary_surface_maximum_m)
            ),
            "aabb_endpoint_maximum_coordinate_difference_m": float(
                aabb_endpoint_maximum_coordinate_difference_m
            ),
            "surface_area_relative_difference": float(surface_area_relative_difference),
            "primary_obj_sha256": primary.get("mesh_sha256"),
            "replay_obj_sha256": replay.get("mesh_sha256"),
            "obj_sha256_equal_recorded_not_required": (
                primary.get("mesh_sha256") == replay.get("mesh_sha256")
            ),
        },
        "thresholds": {
            "nearest_vertex_maximum_m": float(nearest_vertex_limit_m),
            "point_to_surface_maximum_m": (
                float(point_to_surface_limit_m) if point_to_surface_enabled else None
            ),
            "pass_fail_distance_metric": (
                "vertex_to_opposite_triangle_surface_maximum"
                if point_to_surface_enabled
                else "nearest_vertex_maximum"
            ),
            "aabb_endpoint_maximum_coordinate_difference_m": float(aabb_limit_m),
            "surface_area_relative_difference": float(surface_area_relative_limit),
            "triangle_count_relative_difference": (
                None
                if triangle_count_relative_limit is None
                else float(triangle_count_relative_limit)
            ),
            "vertex_count_relative_difference": (
                None
                if vertex_count_relative_limit is None
                else float(vertex_count_relative_limit)
            ),
            "nearest_vertex_threshold_basis": (
                "sqrt(3)*0.4=0.692820 m upstream paired noise bound plus "
                "0.057180 m numerical allowance"
            ),
        },
    }


def batch_mesh_audit(parent_records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    parent_ids = [str(item.get("parent_id")) for item in parent_records]
    checks = {
        "exactly_ten_predeclared_sentinels": tuple(parent_ids) == SENTINEL_PARENT_IDS,
        "all_primary_meshes_passed": all(
            item.get("primary", {}).get("mesh_audit", {}).get("passed") is True
            for item in parent_records
        ),
        "all_replay_meshes_passed": all(
            item.get("replay", {}).get("mesh_audit", {}).get("passed") is True
            for item in parent_records
        ),
        "all_replay_pairs_passed": all(
            item.get("replay_audit", {}).get("passed") is True for item in parent_records
        ),
        "all_sources_are_train": all(item.get("split") == "train" for item in parent_records),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "parent_count": len(parent_records),
        "primary_mesh_count": len(parent_records),
        "replay_mesh_count": len(parent_records),
    }


def immutable_asset_batch_audit(
    parent_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Audit ten single-pass primary assets and explicitly reject replay records."""

    parent_ids = tuple(str(item.get("parent_id")) for item in parent_records)
    hashes = [item.get("primary", {}).get("mesh_sha256") for item in parent_records]
    checks = {
        "exactly_ten_predeclared_train_sentinels": parent_ids == SENTINEL_PARENT_IDS,
        "all_train_split": all(item.get("split") == "train" for item in parent_records),
        "all_primary_meshes_passed": all(
            item.get("primary", {}).get("mesh_audit", {}).get("passed") is True
            for item in parent_records
        ),
        "all_source_identity_checks_passed": all(
            all(item.get("primary", {}).get("identity_checks", {}).values())
            for item in parent_records
        ),
        "all_assets_have_unique_recorded_obj_sha256": (
            len(hashes) == 10
            and all(isinstance(value, str) and len(value) == 64 for value in hashes)
            and len(set(hashes)) == 10
        ),
        "zero_replay_records": all("replay" not in item for item in parent_records),
    }
    return {"passed": all(checks.values()), "checks": checks, "parent_ids": list(parent_ids)}


def immutable_full_parent_asset_batch_audit(
    parent_records: Sequence[Mapping[str, Any]],
    expected_parents: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Audit the full frozen 100-parent immutable-asset batch.

    The expected manifest is supplied explicitly so this helper verifies identity,
    order and split without importing mutable project result paths.
    """

    parent_ids = tuple(str(item.get("parent_id")) for item in parent_records)
    expected_ids = tuple(str(item.get("parent_id")) for item in expected_parents)
    observed_splits = [str(item.get("split")) for item in parent_records]
    expected_splits = [str(item.get("split")) for item in expected_parents]
    observed_recipes = [str(item.get("recipe_stratum_id")) for item in parent_records]
    expected_recipes = [str(item.get("recipe_stratum_id")) for item in expected_parents]
    hashes = [item.get("primary", {}).get("mesh_sha256") for item in parent_records]
    identity_keys = {
        "graph_matches_sealed_source",
        "operation_trace_matches_sealed_source",
        "parent_identity_matches_v2r",
        "splines_match_sealed_source",
    }
    split_counts = {name: observed_splits.count(name) for name in ("train", "validation", "development_test")}
    recipe_counts = {name: observed_recipes.count(name) for name in sorted(set(expected_recipes))}
    checks = {
        "exact_frozen_parent_identity_and_order": len(parent_records) == 100 and parent_ids == expected_ids,
        "exact_frozen_split_assignment": observed_splits == expected_splits and split_counts == {"train": 80, "validation": 10, "development_test": 10},
        "exact_ten_by_ten_recipe_coverage": len(recipe_counts) == 10 and all(value == 10 for value in recipe_counts.values()) and observed_recipes == expected_recipes,
        "all_primary_meshes_passed": all(item.get("primary", {}).get("mesh_audit", {}).get("passed") is True for item in parent_records),
        "all_source_identity_checks_passed": all(set(item.get("primary", {}).get("identity_checks", {})) == identity_keys and all(item.get("primary", {}).get("identity_checks", {}).values()) for item in parent_records),
        "all_assets_have_unique_recorded_obj_sha256": len(hashes) == 100 and all(isinstance(value, str) and len(value) == 64 for value in hashes) and len(set(hashes)) == 100,
        "zero_replay_records": all("replay" not in item for item in parent_records),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "parent_ids": list(parent_ids),
        "split_counts": split_counts,
        "recipe_counts": recipe_counts,
    }


__all__ = [
    "SENTINEL_PARENT_IDS",
    "batch_mesh_audit",
    "canonical_json_hash",
    "geometric_replay_pair_audit",
    "immutable_asset_batch_audit",
    "immutable_full_parent_asset_batch_audit",
    "mesh_array_audit",
    "replay_pair_audit",
    "select_train_sentinels",
    "zero_area_triangle_sanitation",
]
