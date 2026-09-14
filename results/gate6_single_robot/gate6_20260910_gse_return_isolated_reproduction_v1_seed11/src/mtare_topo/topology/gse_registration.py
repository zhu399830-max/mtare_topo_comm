"""CPU Open3D point-to-plane ICP evidence, not a node-merge decision.

All data-dependent thresholds are required in RegistrationConfig. This module
does not read files, retrieve places, use teacher identities, or mutate graphs.
T_target_source maps source-local metre coordinates into target-local metres.
"""
from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class RegistrationConfig:
    max_correspondence_distance_m: float
    normal_radius_m: float
    normal_max_nn: int
    normal_min_neighbors: int
    max_normal_surface_variation: float
    min_normal_second_eigenvalue_ratio: float
    min_normal_coverage: float
    max_iterations: int
    relative_fitness: float
    relative_rmse: float
    point_to_point_bootstrap_iterations: int
    min_correspondences: int
    min_fitness: float
    max_inlier_rmse_m: float
    information_eigenvalue_ratio: float
    max_information_condition: float

    def __post_init__(self):
        for name in ("max_correspondence_distance_m", "normal_radius_m", "max_inlier_rmse_m",
                     "max_information_condition"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"invalid {name}")
        for name in ("max_normal_surface_variation", "min_normal_second_eigenvalue_ratio",
                     "min_normal_coverage", "min_fitness", "information_eigenvalue_ratio"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or not 0 < value <= 1:
                raise ValueError(f"invalid {name}")
        for name in ("relative_fitness", "relative_rmse"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or not 0 <= value < 1:
                raise ValueError(f"invalid {name}")
        for name in ("normal_max_nn", "normal_min_neighbors", "max_iterations", "min_correspondences"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"invalid {name}")
        if (self.normal_min_neighbors < 3 or self.normal_max_nn < self.normal_min_neighbors
                or self.min_correspondences < 6 or self.max_normal_surface_variation > 1 / 3
                or self.max_information_condition < 1 or self.information_eigenvalue_ratio >= 1
                or type(self.point_to_point_bootstrap_iterations) is not int
                or self.point_to_point_bootstrap_iterations < 0):
            raise ValueError("inconsistent registration configuration")


@dataclass(frozen=True)
class InformationDiagnostic:
    scale_m: float | None
    eigenvalues: tuple[float, ...]
    rank: int
    condition: float | None


@dataclass(frozen=True)
class DirectionalRegistrationEvidence:
    source_input_points: int
    target_input_points: int
    source_normal_points: int
    target_normal_points: int
    correspondences: int
    fitness: float                  # denominator: ALL original source points
    filtered_fitness: float         # Open3D denominator: retained source points
    inlier_rmse_m: float | None     # None, never zero, when no correspondences
    point_to_plane_rmse_m: float | None
    information: InformationDiagnostic


@dataclass(frozen=True)
class RegistrationEvidence:
    transformation_target_from_source: tuple[tuple[float, ...], ...] | None
    forward: DirectionalRegistrationEvidence | None
    reverse: DirectionalRegistrationEvidence | None
    accepted: bool
    rejection_reasons: tuple[str, ...]
    backend: str
    source_normal_indices: tuple[int, ...]
    target_normal_indices: tuple[int, ...]
    bootstrap_iterations_requested: int
    runtime_error: str | None = None


def _cloud(value, name):
    try:
        points = np.array(value, dtype=np.float64, copy=True)
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid {name} cloud") from error
    if points.ndim != 2 or points.shape[1:] != (3,) or len(points) < 3 or not np.isfinite(points).all():
        raise ValueError(f"{name} must contain at least three finite 3D points")
    return points


def _se3(value):
    try:
        transform = np.array(value, dtype=np.float64, copy=True)
    except (TypeError, ValueError) as error:
        raise ValueError("invalid initial SE3") from error
    if (transform.shape != (4, 4) or not np.isfinite(transform).all()
            or not np.allclose(transform[3], (0., 0., 0., 1.), rtol=0., atol=1e-9)
            or not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), rtol=0., atol=1e-8)
            or abs(float(np.linalg.det(transform[:3, :3])) - 1.) > 1e-8):
        raise ValueError("initial/estimated transform must be proper finite SE3")
    return transform


def _tuple_matrix(array):
    return tuple(tuple(float(v) for v in row) for row in array)


def _normals(points, config, cpu):
    cloud = cpu.geometry.PointCloud()
    cloud.points = cpu.utility.Vector3dVector(points)
    cloud.estimate_normals(cpu.geometry.KDTreeSearchParamHybrid(
        radius=config.normal_radius_m, max_nn=config.normal_max_nn), fast_normal_computation=False)
    normals = np.asarray(cloud.normals)
    tree = cpu.geometry.KDTreeFlann(cloud)
    keep = []
    for i, point in enumerate(points):
        count, neighbors, _ = tree.search_hybrid_vector_3d(point, config.normal_radius_m, config.normal_max_nn)
        if count < config.normal_min_neighbors:
            continue
        neighborhood = points[np.asarray(neighbors)]
        centered = neighborhood - neighborhood.mean(axis=0)
        eigenvalues = np.linalg.eigvalsh(centered.T @ centered / count)
        total = float(eigenvalues.sum())
        if (total <= np.finfo(float).tiny or eigenvalues[-1] <= 0
                or eigenvalues[1] / eigenvalues[-1] < config.min_normal_second_eigenvalue_ratio
                or max(0., eigenvalues[0]) / total > config.max_normal_surface_variation
                or not np.isfinite(normals[i]).all() or abs(float(np.linalg.norm(normals[i])) - 1.) > 1e-8):
            continue
        keep.append(i)
    # Explicitly report every excluded point; it remains in the overlap denominator.
    filtered = cloud.select_by_index(keep)
    return filtered, tuple(keep)


def point_to_plane_information(points, normals, *, eigenvalue_ratio: float) -> InformationDiagnostic:
    """Dimensionless local J^T J / N diagnostic, not a pose covariance.

    J=[n, ((p-centroid)/RMS_radius) cross n]. Parameters correspond to
    translation/RMS_radius and rotation, so units, common translation and scale
    do not manufacture conditioning. Normal signs do not affect J^T J.
    Full rank only means local observability for these correspondences; it does
    not establish global place uniqueness or protect against repeated rooms.
    """
    points = np.asarray(points, dtype=np.float64)
    normals = np.asarray(normals, dtype=np.float64)
    if (points.ndim != 2 or points.shape[1:] != (3,) or normals.shape != points.shape
            or not np.isfinite(points).all() or not np.isfinite(normals).all()
            or not math.isfinite(eigenvalue_ratio) or not 0 < eigenvalue_ratio < 1):
        raise ValueError("invalid point-to-plane information inputs")
    if len(points) == 0:
        return InformationDiagnostic(None, (0.,) * 6, 0, None)
    if not np.allclose(np.linalg.norm(normals, axis=1), 1., rtol=0., atol=1e-8):
        raise ValueError("information normals must be unit vectors")
    centered = points - points.mean(axis=0)
    scale = float(np.sqrt(np.mean(np.sum(centered * centered, axis=1))))
    if scale <= np.finfo(float).tiny:
        return InformationDiagnostic(scale, (0.,) * 6, 0, None)
    jacobian = np.concatenate((normals, np.cross(centered / scale, normals)), axis=1)
    values = np.maximum(np.linalg.eigvalsh(jacobian.T @ jacobian / len(points)), 0.)
    rank = int(np.count_nonzero(values > values[-1] * eigenvalue_ratio))
    condition = float(values[-1] / values[0]) if rank == 6 else None
    return InformationDiagnostic(scale, tuple(float(v) for v in values), rank, condition)


def _direction(source, target, transform, source_count, target_count, config, cpu):
    evaluated = cpu.pipelines.registration.evaluate_registration(
        source, target, config.max_correspondence_distance_m, transform)
    pairs = np.asarray(evaluated.correspondence_set, dtype=np.int64).reshape(-1, 2)
    source_points = np.asarray(source.points)
    target_points = np.asarray(target.points)
    transformed = source_points @ transform[:3, :3].T + transform[:3, 3]
    if len(pairs):
        a, b = transformed[pairs[:, 0]], target_points[pairs[:, 1]]
        normals = np.asarray(target.normals)[pairs[:, 1]]
        rmse = float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))
        plane_rmse = float(np.sqrt(np.mean(np.sum((a - b) * normals, axis=1) ** 2)))
        information = point_to_plane_information(a, normals, eigenvalue_ratio=config.information_eigenvalue_ratio)
    else:
        rmse, plane_rmse = None, None
        information = InformationDiagnostic(None, (0.,) * 6, 0, None)
    if not math.isclose(float(evaluated.fitness), len(pairs) / max(1, len(source_points)), abs_tol=1e-12):
        raise RuntimeError("Open3D fitness/correspondence contract mismatch")
    return DirectionalRegistrationEvidence(source_count, target_count, len(source_points), len(target_points),
                                           len(pairs), len(pairs) / source_count, float(evaluated.fitness),
                                           rmse, plane_rmse, information)


def register_local_clouds(source_xyz_m, target_xyz_m, initial_target_from_source,
                          config: RegistrationConfig) -> RegistrationEvidence:
    """One explicitly configured local ICP attempt; no global/retry fallback.

    Point-to-point initialization is used only if its requested iteration count
    is nonzero. Final metrics evaluate the estimated transform in both directions;
    the reverse evaluation uses its inverse, not a separately optimized transform.
    Invalid inputs raise ValueError; valid but unusable geometry returns rejection.
    """
    if not isinstance(config, RegistrationConfig):
        raise ValueError("complete RegistrationConfig required")
    source_points, target_points = _cloud(source_xyz_m, "source"), _cloud(target_xyz_m, "target")
    initial = _se3(initial_target_from_source)
    import open3d
    if open3d.__version__ != "0.19.0":
        raise RuntimeError("adapter qualified only against Open3D 0.19.0")
    if open3d.__DEVICE_API__ != "cpu":
        raise RuntimeError("CPU adapter requires CUDA_VISIBLE_DEVICES='' before process launch; do not mix pybind backends")
    cpu = open3d
    backend = "open3d-0.19.0-cpu-legacy-point-to-plane"
    source, source_indices = _normals(source_points, config, cpu)
    target, target_indices = _normals(target_points, config, cpu)
    coverage_reasons = []
    for name, indices, points in (("source", source_indices, source_points), ("target", target_indices, target_points)):
        if len(indices) / len(points) < config.min_normal_coverage:
            coverage_reasons.append(name + "_insufficient_normal_coverage")
        if len(indices) < config.min_correspondences:
            coverage_reasons.append(name + "_insufficient_normal_points")
    if coverage_reasons:
        return RegistrationEvidence(None, None, None, False, tuple(coverage_reasons), backend,
                                    source_indices, target_indices, config.point_to_point_bootstrap_iterations)
    try:
        transform = initial
        if config.point_to_point_bootstrap_iterations:
            bootstrap = cpu.pipelines.registration.registration_icp(
                source, target, config.max_correspondence_distance_m, transform,
                cpu.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=False),
                cpu.pipelines.registration.ICPConvergenceCriteria(
                    config.relative_fitness, config.relative_rmse, config.point_to_point_bootstrap_iterations))
            transform = _se3(bootstrap.transformation)
        estimated = cpu.pipelines.registration.registration_icp(
            source, target, config.max_correspondence_distance_m, transform,
            cpu.pipelines.registration.TransformationEstimationPointToPlane(),
            cpu.pipelines.registration.ICPConvergenceCriteria(
                config.relative_fitness, config.relative_rmse, config.max_iterations))
        transform = _se3(estimated.transformation)
        forward = _direction(source, target, transform, len(source_points), len(target_points), config, cpu)
        reverse = _direction(target, source, np.linalg.inv(transform), len(target_points), len(source_points), config, cpu)
    except (RuntimeError, ValueError, np.linalg.LinAlgError) as error:
        return RegistrationEvidence(None, None, None, False, ("icp_backend_failure",), backend,
                                    source_indices, target_indices, config.point_to_point_bootstrap_iterations, str(error))
    reasons = []
    for name, direction in (("forward", forward), ("reverse", reverse)):
        if direction.correspondences < config.min_correspondences:
            reasons.append(name + "_insufficient_correspondences")
        if direction.fitness < config.min_fitness:
            reasons.append(name + "_low_overlap")
        if direction.inlier_rmse_m is None:
            reasons.append(name + "_rmse_undefined")
        elif direction.inlier_rmse_m > config.max_inlier_rmse_m:
            reasons.append(name + "_high_rmse")
        if direction.information.rank != 6:
            reasons.append(name + "_degenerate_geometry")
        elif direction.information.condition > config.max_information_condition:
            reasons.append(name + "_ill_conditioned_geometry")
    return RegistrationEvidence(_tuple_matrix(transform), forward, reverse, not reasons, tuple(reasons), backend,
                                source_indices, target_indices, config.point_to_point_bootstrap_iterations)
