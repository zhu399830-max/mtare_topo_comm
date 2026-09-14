"""Deterministic loss-side interior section nomination, never a training label.

Use saved original-return source arc intervals, not crop-boundary crossings,
model scores, or a nearest-axis assignment. Disconnected observed intervals
stay separate. A later source-bound geometric/observation check must qualify
each nominated section; this module cannot certify channel observability.
"""
import numpy as np


def nominate_interior_sections(records):
    """One midpoint slab per connected observed source-arc component.

    Input is the original saved Nx7 residual table. Multi-source returns are
    withheld intact. Unique intervals, not ray density, determine the central
    slab. No tolerance bridges gaps, no threshold selects nearest residuals.
    All components are returned including zero-span unresolved components.
    """
    r=np.asarray(records)
    if r.ndim!=2 or r.shape[1]!=7 or r.dtype.kind not in 'if' or not np.isfinite(r).all():
        raise ValueError('finite original Nx7 records required')
    if np.any(r[:,[0,1,4]]!=np.floor(r[:,[0,1,4]])) or np.any(r[:,[0,1]]<0) or np.any(r[:,4]<1):
        raise ValueError('invalid ray/source/triangle-count fields')
    if np.any(r[:,2:4]<0) or np.any(r[:,5]<0) or np.any(r[:,5]>r[:,6]):
        raise ValueError('invalid source residual/arc interval')
    keys=[(int(row[0]),int(row[1])) for row in r]
    if len(set(keys))!=len(keys):raise ValueError('duplicate ray/source record')
    rays,counts=np.unique(r[:,0],return_counts=True)
    ambiguous=set(rays[counts>1]);single=r[np.array([row[0] not in ambiguous for row in r],dtype=bool)]
    result=[]
    for source in sorted(set(single[:,1])):
        values=single[single[:,1]==source]
        intervals=sorted(set((float(row[5]),float(row[6])) for row in values))
        components=[]
        for interval in intervals:
            if not components or interval[0]>max(x[1] for x in components[-1]):components.append([interval])
            else:components[-1].append(interval)
        for index,component in enumerate(components):
            low,high=component[len(component)//2]
            members=set(component)
            ids=sorted(int(row[0]) for row in values if (float(row[5]),float(row[6])) in members)
            result.append(dict(source_index=int(source),component_index=index,
                observed_arc_intervals_m=[list(v) for v in component],
                selected_slab_m=[low,high],reference_arc_m=(low+high)/2 if high>low else None,
                source_return_indices=ids,status='CANDIDATE_REQUIRES_SUPPORT' if high>low else 'UNRESOLVED_ZERO_SPAN',
                training_qualified=False,structural_membership=None))
    return dict(candidates=result,ambiguous_return_indices=sorted(map(int,ambiguous)),
                labels_generated=0,selection_uses_model_scores=False,
                limitation='Source arc component is not a channel instance, semantic event, or traversability edge.')


def section_frame_from_original_rings(vertices_xyz_m, vertex_arc_m, selected_slab_m, angular_segments=64):
    """Read midpoint center and tangent from original mesh rings; no axis fit.

    Reference geometry only. Reject a slab that skips original rings; section
    intersections, origin transform and observed support remain caller tasks.
    """
    v=np.asarray(vertices_xyz_m);a=np.asarray(vertex_arc_m)
    if v.ndim!=2 or v.shape[1]!=3 or a.shape!=(len(v),) or not np.isfinite(v).all() or not np.isfinite(a).all():
        raise ValueError('finite original mesh rings required')
    if angular_segments!=64 or (len(v)-2)%64 or len(v)<130:raise ValueError('original 64-vertex rings plus two cap centers required')
    rings=v[:-2].reshape(-1,64,3);arcs=a[:-2].reshape(-1,64)
    if not np.all(arcs==arcs[:,:1]) or np.any(np.diff(arcs[:,0])<=0):raise ValueError('ring arc layout')
    low,high=map(float,selected_slab_m)
    matches=np.flatnonzero((arcs[:-1,0]==low)&(arcs[1:,0]==high))
    if len(matches)!=1:raise ValueError('section slab must be consecutive original rings')
    i=int(matches[0]);first=rings[i].mean(0);second=rings[i+1].mean(0);delta=second-first
    length=np.linalg.norm(delta)
    if not np.isfinite(length) or length==0:raise ValueError('degenerate ring tangent')
    return dict(center_m=(first+second)/2,normal=delta/length,reference_arc_m=(low+high)/2,
                training_qualified=False)
