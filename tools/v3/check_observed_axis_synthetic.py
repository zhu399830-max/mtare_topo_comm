"""Synthetic analytic T-room counterexample; no dataset or teacher files.

Prints diagnostic outcomes, not a research benchmark pass. Intersections below
raycast an air-volume union of boxes; no physical support/safety claim.
"""
import json
import numpy as np
from mtare_topo.data.cano_sensor_smoke import lidar_local_directions
from mtare_topo.semantics.observed_axis_structure import observed_axis_structure
from mtare_topo.semantics.primitive_relation_nonlearning import PrimitiveRelationBaselineConfig
from mtare_topo.semantics.range_exit_baseline import RangeExitBaselineConfig


def scene_returns(origin, radius_m=10.):
    directions = np.asarray(lidar_local_directions()).reshape(-1, 3)
    boxes = [(np.array([-12., -2., -2.]), np.array([12., 2., 2.])),
             (np.array([-2., 0., -2.]), np.array([2., 12., 2.]))]
    points, ids = [], []
    for i, d in enumerate(directions):
        intervals = []
        for low, high in boxes:
            near, far = -np.inf, np.inf
            for axis in range(3):
                if abs(d[axis]) < 1e-14:
                    if not low[axis] <= origin[axis] <= high[axis]:
                        near, far = np.inf, -np.inf; break
                else:
                    t = sorted([(low[axis]-origin[axis])/d[axis], (high[axis]-origin[axis])/d[axis]])
                    near, far = max(near, t[0]), min(far, t[1])
            if near <= far and far > 0:
                intervals.append((max(0., near), far))
        intervals.sort(); end = 0.
        for start, stop in intervals:
            if start > end + 1e-12: break
            end = max(end, stop)
        if not end > 0: raise ValueError('synthetic sensor not in air union')
        p = d*end
        if np.linalg.norm(p) <= radius_m:
            points.append(p); ids.append(4*11520+i)
    return np.asarray(points), np.asarray(ids)


def main():
    rows = []
    for x in (-4., 0., 4.):
        origin = np.array([x, 0., 0.]); points, ids = scene_returns(origin)
        result = observed_axis_structure(points, ids, np.zeros((5, 3)), np.zeros(5),
            fit_config=PrimitiveRelationBaselineConfig(), exit_config=RangeExitBaselineConfig(),
            max_residual_m=.01, min_crossing_sine=.1, endpoint_tolerance_m=1e-8, maximum_candidates=32)
        rows.append(dict(sensor_x=x, returns=len(points), primitive_count=len(result.primitives),
            fits=[dict(heading=p.heading_deg, reason=p.fit.reason, controls=p.fit.axis_controls_m)
                  for p in result.primitives], rejected_axes=result.rejected_axes,
            structure_positions=[s.position_m for s in result.structures],
            reference_position=(-origin).tolist()))
    # Input-compatibility diagnosis only, NOT a new fair-comparison population.
    center_range_check = []
    from mtare_topo.semantics.observed_primitive_candidates import observed_primitive_candidates
    for radius in (10., 50.):
        points, ids = scene_returns(np.zeros(3), radius)
        proposals = observed_primitive_candidates(points, ids, np.zeros((5, 3)), np.zeros(5),
            fit_config=PrimitiveRelationBaselineConfig(), exit_config=RangeExitBaselineConfig())
        center_range_check.append(dict(radius_m=radius, returns=len(points),
                                       candidate_headings=[p.heading_deg for p in proposals]))
    from mtare_topo.semantics.full_sensor_local_fit import full_sensor_local_fit
    from mtare_topo.semantics.observed_axis_structure import structure_from_observed_primitives
    full_local = []
    for x in (-4., 0., 4.):
        points, ids = scene_returns(np.array([x, 0., 0.]), 50.)
        v = np.zeros((5, 2, 16, 720), np.float32)
        v[:, 0].reshape(5, -1)[4, ids % 11520] = np.linalg.norm(points, axis=1)/50.
        v[:, 1].reshape(5, -1)[4, ids % 11520] = 1
        primitives = full_sensor_local_fit(v, np.zeros((5, 3)), np.zeros(5),
            fit_config=PrimitiveRelationBaselineConfig(), exit_config=RangeExitBaselineConfig())
        result = structure_from_observed_primitives(primitives, max_residual_m=.01,
            min_crossing_sine=.1, endpoint_tolerance_m=1e-8, maximum_candidates=32)
        fit_geometry = []
        for p in primitives:
            if p.fit.axis_controls_m is None: continue
            c = np.asarray(p.fit.axis_controls_m)
            d = c[-1]-c[0]; d /= np.linalg.norm(d)
            middle_residual = np.linalg.norm(c[1]-c[0]-np.dot(c[1]-c[0], d)*d)
            reference = np.array([-x, 0., 0.])
            distances = []
            for a, b in zip(c[:-1], c[1:]):
                delta = b-a
                t = np.clip(np.dot(reference-a, delta)/np.dot(delta, delta), 0., 1.)
                distances.append(np.linalg.norm(reference-(a+t*delta)))
            fit_geometry.append(dict(heading=p.heading_deg, middle_chord_residual_m=float(middle_residual),
                                      reference_to_finite_polyline_m=float(min(distances))))
        full_local.append(dict(sensor_x=x, candidate_count=len(primitives),
            fit_reasons=[p.fit.reason for p in primitives], rejected_axes=result.rejected_axes,
            positions=[s.position_m for s in result.structures], fit_geometry=fit_geometry))
    print(json.dumps(dict(full_sensor_local_fit=full_local,
                          scope='synthetic_single_frame_in_five_frame_layout_not_research',
                          numerical_policy='exact_geometry_software_values_not_calibrated', cases=rows,
                          center_input_compatibility_only=center_range_check), indent=2))


if __name__ == '__main__': main()
