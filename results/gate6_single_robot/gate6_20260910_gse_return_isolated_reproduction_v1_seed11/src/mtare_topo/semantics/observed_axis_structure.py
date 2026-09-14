"""Finite-axis structure proposals, not verified openings or graph edges."""
from dataclasses import dataclass
import numpy as np

from .axis_intersection_candidates import axis_intersection_candidates
from .observed_primitive_candidates import observed_primitive_candidates


@dataclass(frozen=True)
class AxisStructureProposal:
    position_m: tuple
    directions: tuple
    direction_primitive_indices: tuple
    primitive_indices: tuple
    connectivity_verified: bool = False


@dataclass(frozen=True)
class ObservedAxisStructure:
    primitives: tuple
    rejected_axes: tuple
    structures: tuple


def structure_from_observed_primitives(primitives, *, max_residual_m,
        min_crossing_sine, endpoint_tolerance_m, maximum_candidates):
    """Chord approximation only when all three fitted controls agree with it.

    No endpoint is relocated to a teacher anchor. No segment is extended to
    meet another. Outward directions follow observed finite axis extents, not
    graph edges; duplicate/ambiguous physical port identities remain unresolved.
    """
    if len(primitives) > 32:
        raise ValueError('primitive capacity exceeded')
    segments, mapping, rejected = [], [], []
    # Validate numerical policy even when no primitive survives.
    policy = dict(max_residual_m=max_residual_m, min_crossing_sine=min_crossing_sine,
                  endpoint_tolerance_m=endpoint_tolerance_m, maximum_candidates=maximum_candidates)
    axis_intersection_candidates(np.empty((0, 2, 3)), **policy)
    for index, primitive in enumerate(primitives):
        fit = primitive.fit
        if fit.axis_controls_m is None:
            rejected.append((index, fit.reason)); continue
        controls = np.asarray(fit.axis_controls_m, float)
        if controls.shape != (3, 3) or not np.isfinite(controls).all():
            raise ValueError('finite three-control fit required')
        delta = controls[-1]-controls[0]
        length = np.linalg.norm(delta)
        if length <= np.finfo(float).eps:
            rejected.append((index, 'degenerate_fitted_chord')); continue
        direction = delta/length
        along = (controls-controls[0]) @ direction
        residual = np.linalg.norm(controls-controls[0]-along[:, None]*direction, axis=1)
        if (np.max(residual) > max_residual_m or np.any(along < -endpoint_tolerance_m)
                or np.any(along > length+endpoint_tolerance_m)):
            rejected.append((index, 'curved_fit_not_represented_by_chord')); continue
        segments.append(controls[[0, 2]]); mapping.append(index)
    segments = np.asarray(segments).reshape(-1, 2, 3)
    structures = []
    for group in axis_intersection_candidates(segments, **policy):
        directions, sources = [], []
        for i in group.segment_indices:
            a, b = segments[i]
            length = np.linalg.norm(b-a); d = (b-a)/length
            along = float((np.asarray(group.position_m)-a) @ d)
            if along > endpoint_tolerance_m:
                directions.append(tuple(-d)); sources.append(mapping[i])
            if length-along > endpoint_tolerance_m:
                directions.append(tuple(d)); sources.append(mapping[i])
        structures.append(AxisStructureProposal(group.position_m, tuple(directions),
                          tuple(sources), tuple(mapping[i] for i in group.segment_indices)))
    return ObservedAxisStructure(tuple(primitives), tuple(rejected), tuple(structures))


def observed_axis_structure(points, ray_indices, translation, yaw, *, fit_config,
                            exit_config, max_residual_m, min_crossing_sine,
                            endpoint_tolerance_m, maximum_candidates):
    primitives = observed_primitive_candidates(points, ray_indices, translation, yaw,
                                              fit_config=fit_config, exit_config=exit_config)
    return structure_from_observed_primitives(primitives, max_residual_m=max_residual_m,
        min_crossing_sine=min_crossing_sine, endpoint_tolerance_m=endpoint_tolerance_m,
        maximum_candidates=maximum_candidates)
