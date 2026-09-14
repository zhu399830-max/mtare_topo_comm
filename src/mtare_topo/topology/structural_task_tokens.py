"""Observation-only task token binding. Retrieval is not a verified merge."""
import numpy as np


def bind_task_tokens(proposals, source_frame_keys, point_patch_index, tokens):
    tokens=np.asarray(tokens);mapping=np.asarray(point_patch_index)
    if len(source_frame_keys)!=5 or len(set(source_frame_keys))!=5 or mapping.shape!=(57600,):
        raise ValueError('exact five-frame original ray layout required')
    if tokens.ndim!=2 or tokens.shape[1]!=128 or not np.isfinite(tokens).all():
        raise ValueError('finite128D patch tokens required')
    if np.any(mapping < -1) or np.any(mapping>=len(tokens)):raise ValueError('patch index out of range')
    result=[]
    for i,p in enumerate(proposals):
        ids=[]
        for ref in p['source_refs']:
            key,ray=ref.rsplit('/ray:',1)
            if key not in source_frame_keys or not 0<=int(ray)<11520:raise ValueError('foreign source ray')
            ids.append(source_frame_keys.index(key)*11520+int(ray))
        ids=np.unique(ids);patch_ids,weights=np.unique(mapping[ids][mapping[ids]>=0],return_counts=True)
        vector=None
        if len(patch_ids):
            pooled=np.average(tokens[patch_ids].astype(float),axis=0,weights=weights)
            norm=np.linalg.norm(pooled)
            if norm>0:vector=(pooled/norm).tolist()
        result.append(dict(candidate=i,descriptor=vector,support_patch_ids=patch_ids.tolist(),
            support_return_counts=weights.tolist(),candidate_source_rays=len(ids),
            unsupported_source_rays=int((mapping[ids]<0).sum()),
            status='TOKEN_SUPPORTED' if vector is not None else 'UNKNOWN_NO_NONZERO_PATCH_TOKEN',
            axis_start_xyz_m=p['axis_start_xyz_m'],axis_target_xyz_m=p['axis_target_xyz_m'],
            correspondence_verified=False,creates_edge=False))
    return result


def retrieval_candidates(previous,current):
    """Save all cosine scores and exact best ties; no calibrated probability."""
    rows=[]
    for c in current:
        scores=[dict(previous=p['candidate'],cosine=float(np.clip(np.dot(c['descriptor'],p['descriptor']),-1,1)))
                for p in previous if c['descriptor'] is not None and p['descriptor'] is not None]
        best=max((p['cosine'] for p in scores),default=None)
        rows.append(dict(current=c['candidate'],scores=scores,
            best_previous=[p['previous'] for p in scores if p['cosine']==best],
            status='UNVERIFIED_RETRIEVAL_ONLY' if scores else 'UNKNOWN',merge_committed=False))
    return rows


def bind_directional_context(proposals, centers_sensor_m, tokens, sensor_to_map):
    """Parameter-free directional context, NOT channel membership/support proof.

    Pool all observed patches in the forward half-space of a proposal, with
    positive cosine weight. Side surfaces may contribute even when the open
    ray has no near return. Zero/behind patches contribute zero. No GT,
    predicted probability, range enlargement, hard angular threshold or fit.
    """
    c=np.asarray(centers_sensor_m,float);t=np.asarray(tokens,float);pose=np.asarray(sensor_to_map,float)
    if c.shape!=(len(t),3) or t.ndim!=2 or t.shape[1]!=128 or pose.shape!=(4,4):raise ValueError('shape mismatch')
    if not all(np.isfinite(x).all() for x in (c,t,pose)):raise ValueError('nonfinite context')
    if not np.allclose(pose[:3,:3].T@pose[:3,:3],np.eye(3),atol=1e-6) or not np.isclose(np.linalg.det(pose[:3,:3]),1):raise ValueError('proper rotation required')
    n=np.linalg.norm(c,axis=1);unit=np.divide(c,n[:,None],out=np.zeros_like(c),where=n[:,None]>0)
    rows=[]
    for i,p in enumerate(proposals):
        d=pose[:3,:3].T@(np.asarray(p['axis_target_xyz_m'])-np.asarray(p['axis_start_xyz_m']))
        if d.shape!=(3,) or not np.isfinite(d).all() or np.linalg.norm(d)==0:raise ValueError('finite nonzero direction required')
        w=np.maximum(unit@(d/np.linalg.norm(d)),0);ids=np.flatnonzero(w>0);vector=None
        if len(ids):
            v=np.average(t[ids],axis=0,weights=w[ids]);vn=np.linalg.norm(v)
            if vn>0:vector=(v/vn).tolist()
        rows.append(dict(candidate=i,descriptor=vector,context_patch_ids=ids.tolist(),context_weights=w[ids].tolist(),
            status='DIRECTIONAL_CONTEXT_ONLY' if vector is not None else 'UNKNOWN_NO_CONTEXT',
            membership_certified=False,correspondence_verified=False,creates_edge=False,
            axis_start_xyz_m=p['axis_start_xyz_m'],axis_target_xyz_m=p['axis_target_xyz_m']))
    return rows
