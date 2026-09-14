"""Conditional window-reference negatives; not complete physical apertures.

All source-axis sphere intersections are retained, regardless of surface or
visibility support. They form a superset of the current opening producer's
reference centers. This never labels root reachability or persistent nodes.
"""
import numpy as np
from mtare_topo.data.primitive_relation_materialization import load_p1a_realized_construction
from mtare_topo.data.primitive_relation_dataset import _current_sensor_transform
from .gse_roi_crossings_v1 import roi_crossings
from .gse_reference_exclusion_binding_v1 import bound_reference_exclusion
from .gse_reference_exclusion_v1 import exclusion_from_ray_grid


def all_window_reference_positions(primitives, *, center_world_m, yaw_deg):
    positions=[]
    for primitive in primitives:
        roots=roi_crossings(primitive.centerline_xyz_m,center_m=center_world_m)
        if roots.ambiguous_segment_indices:
            raise ValueError('ambiguous ROI roots prevent complete reference exclusion')
        positions.extend(roots.positions_m.tolist())
    if len(positions)>4096:
        raise OverflowError('reference inventory capacity exceeded; never truncate')
    return _current_sensor_transform(np.asarray(positions,dtype=np.float64).reshape(-1,3),
                                     center_world_m,yaw_deg)


def bound_opening_reference_exclusion(bundle,grid,query_xyz_m,*,expected_binding,matching_radius_m):
    # Reuse exact original motion/projection/grid/source validation. Anchor
    # exclusion labels are intentionally discarded, never used for openings.
    verified=bound_reference_exclusion(bundle,grid,query_xyz_m,
        expected_binding=expected_binding,matching_radius_m=matching_radius_m)
    _,primitives=load_p1a_realized_construction(bundle['construction_teacher_only'])
    sensor=bundle['sensor_teacher_only']
    positions=all_window_reference_positions(primitives,
        center_world_m=sensor['sensor_xyz_m'][-1],yaw_deg=float(sensor['yaw_deg'][-1]))
    result=exclusion_from_ray_grid(query_xyz_m=query_xyz_m,grid=grid,
        all_anchor_xyz_m=positions,inventory_complete=True,matching_radius_m=matching_radius_m)
    result.update(binding=verified['binding'],observation_identity_binding_verified=True,
        inventory_completeness_supplied_not_verified=False,reference_count=len(positions),
        inventory_definition='all fixed10m sphere crossings of all frozen source axes; not full physical aperture truth',
        matching_radius_m=float(matching_radius_m),reference_kind='window_opening',
        physical_traversability=False)
    return result
