"""Non-learning decomposition of coarse sectors at the existing local boundary.

Components are adjacent observed ray directions, NOT connected free space,
traversable branches, semantic instances, or learned geometry performance.
Every original supporting ray is retained as a component or explicit residual.
"""
import numpy as np
from mtare_topo.semantics.observed_direction_proposals import range_proposals as coarse_proposals
from mtare_topo.semantics.primitive_relation_nonlearning import ELEVATION_DEG


def ray_components(mask):
    """Four-neighbor image adjacency, azimuth wraps; elevation never wraps."""
    mask=np.asarray(mask)
    if mask.ndim!=2 or mask.dtype!=np.bool_:raise ValueError('boolean ray grid required')
    rows,cols=mask.shape;seen=np.zeros_like(mask);groups=[]
    for first in np.flatnonzero(mask):
        r,c=divmod(int(first),cols)
        if seen[r,c]:continue
        stack=[(r,c)];seen[r,c]=True;group=[]
        while stack:
            r,c=stack.pop();group.append(r*cols+c)
            for nr,nc in ((r-1,c),(r+1,c),(r,(c-1)%cols),(r,(c+1)%cols)):
                if 0<=nr<rows and mask[nr,nc] and not seen[nr,nc]:seen[nr,nc]=True;stack.append((nr,nc))
        groups.append(sorted(group))
    return groups


def decompose(ranges,valid,pose,key,parents):
    ranges=np.asarray(ranges);valid=np.asarray(valid,bool);pose=np.asarray(pose)
    if ranges.shape!=(16,720) or valid.shape!=ranges.shape or pose.shape!=(4,4):raise ValueError('original input shape')
    el=np.deg2rad(ELEVATION_DEG);az=np.arange(720)*np.pi/360
    unit=np.stack(np.broadcast_arrays(np.cos(el)[:,None]*np.cos(az),np.cos(el)[:,None]*np.sin(az),np.sin(el)[:,None]+np.zeros((16,720))),axis=-1).reshape(-1,3)
    children=[];audit=[]
    for parent_index,parent in enumerate(parents):
        ids=[]
        for ref in parent['source_refs']:
            source,index=ref.rsplit('/ray:',1)
            if source!=key or not 0<=int(index)<11520:raise ValueError('foreign source ray')
            ids.append(int(index))
        if len(set(ids))!=len(ids):raise ValueError('duplicate source ray')
        source=np.zeros(11520,bool);source[ids]=True;source=source.reshape(16,720)
        active=source&valid&np.isfinite(ranges)&(ranges>10.)
        components=ray_components(active)
        remaining=np.flatnonzero(source&~active).tolist()
        record=dict(parent_candidate=parent_index,parent_proposal=parent,component_ray_indices=components,
                    residual_ray_indices=remaining,component_count=len(components),semantic_partition_qualified=False)
        assert set(sum(components,[]))|set(remaining)==set(ids)
        assert not set(sum(components,[]))&set(remaining)
        audit.append(record)
        if not components:
            children.append(dict(parent,component_kind='UNRESOLVED_PARENT_NO_BOUNDARY_COMPONENT',parent_candidate=parent_index))
            continue
        for component_index,indices in enumerate(components):
            direction=unit[indices].mean(axis=0);norm=np.linalg.norm(direction)
            # Opposed directions can cancel. Do not invent a valid target.
            if norm==0:raise ValueError('component direction cancels; no heading repair')
            world_direction=pose[:3,:3]@(direction/norm)
            children.append(dict(axis_start_xyz_m=pose[:3,3].tolist(),axis_target_xyz_m=(pose[:3,3]+4*world_direction).tolist(),
                source_refs=[key+'/ray:'+str(i) for i in indices],current_direction_supported=True,
                geometry_source_kind='observed_ray_boundary_component',creates_edge=False,physical_opening=False,traversability='unknown',
                component_kind='BOUNDARY_REACHING_RAY_COMPONENT',parent_candidate=parent_index,component_index=component_index,
                support_policy='four_neighbor_observed_ray_grid_existing10m_domain',learned=False))
    return children,audit


def range_proposals(ranges,valid,pose,key):
    parents=coarse_proposals(ranges,valid,pose,key)
    children,audit=decompose(ranges,valid,pose,key,parents)
    # Keep the full residual evidence in output, without converting it to a
    # direction or silently erasing the parent. Runner copies this separately.
    for child in children:child['parent_decomposition_evidence']=audit[child['parent_candidate']]
    return children
