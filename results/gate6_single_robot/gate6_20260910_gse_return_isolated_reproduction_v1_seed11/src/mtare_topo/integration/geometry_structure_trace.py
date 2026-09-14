"""Transport actual finite-axis composition with ray provenance to graph users.

This is a candidate observation, never a confirmed opening or a traversal edge.
No teacher IDs are accepted. One record contains all proposals, including
rejected primitives, so a consumer cannot silently turn missing evidence into
background or claim a fitted endpoint was a physical portal.
"""
import numpy as np

from mtare_topo.semantics.observed_axis_structure import ObservedAxisStructure


def geometry_structure_record(observation, *, source_frame_keys,
                              source_ray_indices, sensor_to_world,
                              coordinate_frame, timestamp):
    if not isinstance(observation, ObservedAxisStructure):
        raise ValueError('actual observed-axis structure output required')
    if (len(source_frame_keys) != 5 or len(set(source_frame_keys)) != 5
            or any(not isinstance(k, str) or not k for k in source_frame_keys)):
        raise ValueError('five distinct causal source frame keys required')
    rays = np.asarray(source_ray_indices)
    if (rays.ndim != 1 or rays.dtype.kind not in 'iu' or np.any(rays < 0)
            or np.any(rays >= 57600) or len(np.unique(rays)) != len(rays)):
        raise ValueError('original unique five-frame ray indices required')
    matrix = np.asarray(sensor_to_world, dtype=float)
    if (matrix.shape != (4, 4) or not np.isfinite(matrix).all()
            or not np.allclose(matrix[3], [0, 0, 0, 1])
            or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(matrix[:3, :3]), 1.)):
        raise ValueError('proper rigid sensor-to-world transform required')
    if not isinstance(coordinate_frame, str) or not coordinate_frame or not np.isfinite(timestamp):
        raise ValueError('explicit coordinate frame and finite timestamp required')
    available = set(map(int, rays))
    primitives = []
    for index, primitive in enumerate(observation.primitives):
        fit = primitive.fit
        selected = tuple(fit.selected_ray_indices)
        if not set(selected).issubset(available):
            raise ValueError('primitive cites rays absent from this observation')
        controls = fit.axis_controls_m
        world = None
        if controls is not None:
            controls = np.asarray(controls, float)
            if controls.shape != (3, 3) or not np.isfinite(controls).all() or not selected:
                raise ValueError('fitted primitive requires finite controls and source support')
            world = (controls @ matrix[:3, :3].T + matrix[:3, 3]).tolist()
        primitives.append(dict(index=index, fit_status=fit.reason,
            axis_controls_world_m=world, source_rays=[dict(
                frame_key=source_frame_keys[r // 11520], ray_index=r % 11520
            ) for r in selected], residual=fit.residual))
    structures = []
    for index, structure in enumerate(observation.structures):
        members = tuple(structure.primitive_indices)
        if (not members or any(i < 0 or i >= len(primitives) for i in members)
                or len(structure.directions) != len(structure.direction_primitive_indices)
                or any(i not in members for i in structure.direction_primitive_indices)):
            raise ValueError('composition must reference its actual primitives')
        center = np.asarray(structure.position_m, float)
        directions = np.asarray(structure.directions, float)
        if (center.shape != (3,) or directions.ndim != 2 or directions.shape[1] != 3
                or not np.isfinite(center).all() or not np.isfinite(directions).all()
                or not np.allclose(np.linalg.norm(directions, axis=1), 1.)):
            raise ValueError('finite composition position and unit directions required')
        structures.append(dict(index=index, state='proposed_structure',
            position_world_m=(matrix[:3, :3] @ center + matrix[:3, 3]).tolist(),
            directions_world=(directions @ matrix[:3, :3].T).tolist(),
            direction_primitive_indices=list(structure.direction_primitive_indices),
            primitive_indices=list(members), connectivity_verified=False))
    return dict(schema_version='geometry_structure_trace_v1', timestamp=float(timestamp),
        coordinate_frame=coordinate_frame, source_frame_keys=list(source_frame_keys),
        primitives=primitives, structures=structures,
        rejected_axes=[list(r) for r in observation.rejected_axes],
        physical_openings_confirmed=False, traversed_edges=[],
        evidence_scope='observed_surface_fit_and_finite_axis_composition_only')
