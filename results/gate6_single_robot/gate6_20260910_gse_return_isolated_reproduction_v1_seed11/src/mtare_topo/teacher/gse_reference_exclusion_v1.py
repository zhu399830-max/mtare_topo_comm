"""Prospective local anchor negatives under an explicit reference universe.

Requires an exhaustive construction-anchor inventory, including unsupported
references, AND observation evidence at the queried location. It does not
certify that the inventory captures every real geometric event. No whole-region
completeness, opening traversability, or training permission is produced.
"""
import numpy as np


def exclusion_from_ray_grid(*, query_xyz_m, grid, all_anchor_xyz_m,
                            inventory_complete, matching_radius_m):
    """Query sealed observed cells; cell boundaries/outside stay UNKNOWN.

    Grid content is verified here. Its provenance still has to be bound to the
    exact observation by the data reader; no dataset is read by this function.
    """
    from mtare_topo.representation.gse_surface_ray_evidence_v1 import (
        query_patch_gaps, _point_cells, _numerical_bound)
    # Reuse the existing typed-grid state, frame-bit and content-hash checks.
    query_patch_gaps(grid,np.empty((0,3),dtype=np.float64),
        np.empty((0,8),dtype=np.int64),np.empty((0,8),dtype=bool))
    q=np.asarray(query_xyz_m,dtype=np.float64)
    if q.ndim!=2 or q.shape[1:]!=(3,) or len(q)>64 or not np.isfinite(q).all():
        raise ValueError('bounded finite query positions required')
    states=np.zeros(len(q),dtype=np.uint8)
    tolerance=max(grid.numerical_bound_m,_numerical_bound(q))
    for i,point in enumerate(q):
        if np.any(np.abs(point)>=10.):continue
        cells=_point_cells(point,tolerance)
        if len(cells)==1:states[i]=grid.state.reshape(-1)[cells[0]]
    result=reference_exclusion(query_xyz_m=q,observed_state=states,
        all_anchor_xyz_m=all_anchor_xyz_m,inventory_complete=inventory_complete,
        matching_radius_m=matching_radius_m)
    result['observed_states_supplied_not_verified']=False
    result['query_observed_states']=states.tolist()
    result['grid_content_sha256']=grid.content_sha256
    result['grid_source_geometry_sha256']=grid.source_geometry_sha256
    result['observation_identity_binding_verified']=False
    return result


def reference_exclusion(*, query_xyz_m, observed_state, all_anchor_xyz_m,
                        inventory_complete, matching_radius_m):
    q=np.asarray(query_xyz_m,dtype=np.float64)
    a=np.asarray(all_anchor_xyz_m,dtype=np.float64)
    state=np.asarray(observed_state)
    if (q.ndim!=2 or q.shape[1:]!=(3,) or len(q)>64
            or a.ndim!=2 or a.shape[1:]!=(3,) or len(a)>4096
            or not np.isfinite(q).all() or not np.isfinite(a).all()
            or state.shape!=(len(q),) or state.dtype.kind not in 'iu'
            or not np.isin(state,[0,1,2]).all() or type(inventory_complete) is not bool
            or not np.isfinite(matching_radius_m) or matching_radius_m<=0):
        raise ValueError('bounded finite queries, exhaustive reference inventory and explicit observation states required')
    # State0 is UNKNOWN; FREE1 and observed surface2 only establish local
    # measurement support. Neither by itself is evidence of structural absence.
    nearest=np.min(np.linalg.norm(q[:,None,:]-a[None,:,:],axis=-1),axis=1) if len(a) else np.full(len(q),np.inf)
    negative=(state!=0)&(nearest>matching_radius_m)&inventory_complete
    reasons=[]
    for i in range(len(q)):
        if state[i]==0:reason='UNOBSERVED_LOCATION'
        elif not inventory_complete:reason='REFERENCE_INVENTORY_NOT_COMPLETE'
        elif nearest[i]<=matching_radius_m:reason='POSSIBLE_REFERENCE_MATCH_KEEP_UNKNOWN'
        else:reason='OBSERVED_AND_EXCLUDED_FROM_COMPLETE_REFERENCE_UNIVERSE'
        reasons.append(reason)
    return dict(reference_negative_mask=negative.tolist(),reason=reasons,
        negative_definition='cannot match any anchor in the supplied exhaustive reference universe',
        inventory_completeness_supplied_not_verified=True,
        observed_states_supplied_not_verified=True,whole_region_complete=False,
        training_eligible=False,scientific_gate_pass=False)
