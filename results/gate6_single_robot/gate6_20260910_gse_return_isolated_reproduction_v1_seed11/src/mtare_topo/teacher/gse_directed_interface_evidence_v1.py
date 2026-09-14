"""Interpret existing cap intersections, without imposing crossing order.

Inward directions must come from actual source axes, not connector segments.
Returns evidence, not structural labels or a complete scoring region.
"""
import numpy as np


def directed_evidence(records, interfaces, *, directions, first_return, valid, return_sources):
    d=np.asarray(directions);r=np.asarray(first_return);v=np.asarray(valid)
    if (d.ndim!=2 or d.shape[1:]!=(3,) or not np.isfinite(d).all()
            or r.shape!=(len(d),) or r.dtype!=np.float32 or v.shape!=r.shape or v.dtype!=np.bool_
            or len(return_sources)!=len(d) or np.any(~np.isfinite(r[v])) or np.any(r[v]<0)
            or np.any(np.linalg.norm(d,axis=1)==0)):
        raise ValueError('original finite directions,float32 returns and explicit provenance required')
    lookup={}
    for item in interfaces:
        key=item['interface_id_teacher_only'];n=np.asarray(item['inward_direction'],dtype=np.float64)
        if key in lookup or n.shape!=(3,) or not np.isfinite(n).all() or abs(np.linalg.norm(n)-1)>1e-12:
            raise ValueError('unique source interface and unit actual-axis direction required')
        lookup[key]=item
    output={k:dict(entering_ray_indices=[],leaving_ray_indices=[],unknown_ray_indices=[]) for k in lookup}
    candidates={};seen=set()
    for record in records:
        i=record['ray_index'];key=record['interface_id_teacher_only'];t=record['t']
        if type(i) is not int or not 0<=i<len(d) or key not in lookup or not np.isfinite(t):
            raise ValueError('intersection binding mismatch')
        # Shared triangles on the same cap must not multiply a witness.
        fingerprint=(i,key,float(t))
        if fingerprint in seen:continue
        seen.add(fingerprint)
        item=lookup[key];target=output[key]
        limit=float(r[i])-abs(float(np.spacing(r[i]))) if v[i] else 0.
        if not record['inside_roi'] or not v[i] or not 0<t<limit:
            target['unknown_ray_indices'].append(i);continue
        dot=float(d[i]@np.asarray(item['inward_direction']))
        guard=64*np.finfo(np.float64).eps*np.linalg.norm(d[i])
        if abs(dot)<=guard:
            target['unknown_ray_indices'].append(i)
        elif dot<0:
            target['leaving_ray_indices'].append(i)
        elif return_sources[i]!=[item['source_key_teacher_only']]:
            target['unknown_ray_indices'].append(i)
        else:
            # One source may own multiple incident arcs; direction may still
            # fail to distinguish them. Resolve only within the same node.
            candidates.setdefault((i,item['node_id_teacher_only']),set()).add(key)
    for (i,_),keys in candidates.items():
        for key in keys:
            field='entering_ray_indices' if len(keys)==1 else 'unknown_ray_indices'
            output[key][field].append(i)
    for target in output.values():
        for field in target:target[field]=sorted(set(target[field]))
        # Repeated intersections of one curved source can have conflicting
        # directions. Keep those rays unknown, never choose the useful sign.
        ambiguous=set(target['unknown_ray_indices'])|(set(target['entering_ray_indices'])&set(target['leaving_ray_indices']))
        target['entering_ray_indices']=sorted(set(target['entering_ray_indices'])-ambiguous)
        target['leaving_ray_indices']=sorted(set(target['leaving_ray_indices'])-ambiguous)
        target['unknown_ray_indices']=sorted(ambiguous)
    return dict(interfaces=output,qualified_labels=0,complete_region=False,
        limitation='Finite inward source-interface evidence only; no physical connectivity or structural label proof.')
