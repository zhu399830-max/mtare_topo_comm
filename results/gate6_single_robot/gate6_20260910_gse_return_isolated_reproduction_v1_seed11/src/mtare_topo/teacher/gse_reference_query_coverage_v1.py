"""Query-wise reference scoring coverage, not full physical annotation.

A repeated prediction near a confirmed reference is scoreable only if no
unconfirmed reference can also match it. Hidden references remain in inventory.
"""
import numpy as np
from .gse_reference_exclusion_v1 import reference_exclusion


def reference_query_coverage(*, query_xyz_m, observed_state, all_reference_xyz_m,
                             confirmed_reference_indices, matching_radius_m,
                             score_center_m, score_radius_m):
    q=np.asarray(query_xyz_m,dtype=float); refs=np.asarray(all_reference_xyz_m,dtype=float).reshape(-1,3)
    exclusion=reference_exclusion(query_xyz_m=q,observed_state=observed_state,
        all_anchor_xyz_m=refs,inventory_complete=True,matching_radius_m=matching_radius_m)
    indices=list(confirmed_reference_indices)
    if (len(indices)!=len(set(indices)) or any(type(i) is not int or not 0<=i<len(refs) for i in indices)):
        raise ValueError('unique confirmed reference indices required')
    center=np.asarray(score_center_m,dtype=float)
    if center.shape!=(3,) or not np.isfinite(center).all() or not 0<float(score_radius_m)<=10:
        raise ValueError('declared finite score sphere required')
    hidden=[i for i in range(len(refs)) if i not in set(indices)]
    possible=np.linalg.norm(q[:,None,:]-refs[None,hidden,:],axis=2)<=matching_radius_m
    conflict=possible.any(axis=1)
    inside=np.linalg.norm(q-center,axis=1)<=score_radius_m
    observed=np.asarray(observed_state)!=0
    scoreable=inside & observed & ~conflict
    return dict(query_scoreable_mask=scoreable.tolist(),
        possible_unconfirmed_reference_mask=conflict.tolist(),
        observed_inside_score_region_mask=(inside & observed).tolist(),
        reference_negative_mask=exclusion['reference_negative_mask'],
        reference_inventory_complete_supplied_not_verified=True,
        physical_annotation_complete=False, full_f1=None, scientific_gate_pass=False)


def bound_anchor_query_coverage(bundle, grid, query_xyz_m, *, produced_targets,
                                manifest_row, matching_radius_m):
    return _bound_query_coverage(bundle, grid, query_xyz_m, produced_targets=produced_targets,
        manifest_row=manifest_row, matching_radius_m=matching_radius_m, kind='anchors')


def bound_opening_query_coverage(bundle, grid, query_xyz_m, *, produced_targets,
                                 manifest_row, matching_radius_m):
    return _bound_query_coverage(bundle, grid, query_xyz_m, produced_targets=produced_targets,
        manifest_row=manifest_row, matching_radius_m=matching_radius_m, kind='openings')


def _bound_query_coverage(bundle, grid, query_xyz_m, *, produced_targets,
                          manifest_row, matching_radius_m, kind,
                          coverage_function=reference_query_coverage):
    from mtare_topo.data.gse_structure_review_v1 import canonical_sha
    from mtare_topo.data.primitive_relation_dataset import _current_sensor_transform
    from .gse_reference_exclusion_binding_v1 import bound_reference_exclusion
    from .gse_construction_paths_v3 import construction_incident_paths
    if set(manifest_row)!={'source_binding','target_record_sha256'}:
        raise ValueError('independent frozen target manifest required')
    record=produced_targets['record']
    if (produced_targets['source_binding']!=manifest_row['source_binding']
            or produced_targets['target_record_sha256']!=manifest_row['target_record_sha256']
            or canonical_sha(record)!=manifest_row['target_record_sha256']
            or record['source_frame_indices']!=manifest_row['source_binding']['source']['frame_rows']):
        raise ValueError('frozen positive record binding mismatch')
    evidence=bound_reference_exclusion(bundle,grid,query_xyz_m,
        expected_binding=manifest_row['source_binding'],matching_radius_m=matching_radius_m)
    s=bundle['sensor_teacher_only']
    if kind == 'anchors':
        groups=[g for g in construction_incident_paths(bundle['construction_teacher_only'])
                if len(g['paths'])==1 or len(g['paths'])>=3]
        refs=_current_sensor_transform(np.asarray([g['anchor_world_m'] for g in groups],dtype=float).reshape(-1,3),
            s['sensor_xyz_m'][-1],float(s['yaw_deg'][-1]))
    else:
        from mtare_topo.data.primitive_relation_materialization import load_p1a_realized_construction
        from .gse_opening_reference_exclusion_v1 import all_window_reference_positions
        _, primitives=load_p1a_realized_construction(bundle['construction_teacher_only'])
        refs=all_window_reference_positions(primitives,
            center_world_m=s['sensor_xyz_m'][-1],yaw_deg=float(s['yaw_deg'][-1]))
    confirmed=[]
    for anchor in record[kind]:
        matches=np.flatnonzero(np.all(refs==anchor['position_m'],axis=1))
        if len(matches)!=1:
            raise ValueError('positive position must uniquely match original reference')
        confirmed.append(int(matches[0]))
    region=record['score_region']
    # The verified grid emits uint8 states, serialized to a list for evidence.
    # Restore that dtype here: np.asarray([]) otherwise becomes float64 when
    # thresholding legitimately selects zero predictions. Do not weaken the
    # public reference_exclusion validator or turn unknown states into negatives.
    observed_states=np.asarray(evidence['query_observed_states'],dtype=np.uint8)
    result=coverage_function(query_xyz_m=query_xyz_m,observed_state=observed_states,
        all_reference_xyz_m=refs,confirmed_reference_indices=confirmed,matching_radius_m=matching_radius_m,
        score_center_m=region['center_m'],score_radius_m=region['radius_m'])
    result.update(reference_inventory_complete_supplied_not_verified=False,
        source_binding=manifest_row['source_binding'],target_record_sha256=manifest_row['target_record_sha256'],
        grid_content_sha256=grid.content_sha256,query_xyz_m=np.asarray(query_xyz_m).tolist(),
        reference_count=len(refs),confirmed_reference_count=len(confirmed),reference_kind=kind)
    return result
