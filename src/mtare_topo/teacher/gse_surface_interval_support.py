"""Join saved nearest-face arc hypotheses to local reference intervals.

This is conditional support, not an independently qualified point label.
Source ambiguity, boundary support, and absent local axes remain explicit.
Residual magnitudes do not select or discard hypotheses.
"""
import numpy as np


def join_surface_interval_support(records, intervals_by_source):
    rows=np.asarray(records,dtype=np.float64)
    if rows.ndim!=2 or rows.shape[1]!=7 or not np.isfinite(rows).all():
        raise ValueError('finite seven-column residual records required')
    if np.any(rows[:,:2]<0) or not np.equal(rows[:,:2],np.floor(rows[:,:2])).all():
        raise ValueError('integer ray and source indices required')
    if np.any(rows[:,5]>rows[:,6]):raise ValueError('ordered arc bounds required')
    # State: 0 source crossing unresolved; 1 no interval overlap;
    # 2 overlap but boundary/multiple; 3 strictly one interval (conditional).
    state=np.zeros(len(rows),dtype=np.uint8)
    chosen=np.full(len(rows),-1,dtype=np.int32)
    overlaps=np.zeros(len(rows),dtype=np.int32)
    for source in np.unique(rows[:,1]).astype(int):
        if source not in intervals_by_source:raise ValueError('missing source interval inventory')
        interval=intervals_by_source[source]
        if interval['status']!='REFERENCE_INTERVALS_ONLY':continue
        idx=np.flatnonzero(rows[:,1]==source);low=rows[idx,5];high=rows[idx,6]
        state[idx]=1
        for j,(lo,hi) in enumerate(interval['intervals_m']):
            overlap=(high>=lo)&(low<=hi)
            overlaps[idx]+=overlap
            strict=(lo<low)&(high<hi)
            chosen[idx[strict]]=j
        state[idx[overlaps[idx]>0]]=2
        strict=(overlaps[idx]==1)&(chosen[idx]>=0)
        state[idx[strict]]=3
        chosen[idx[~strict]]=-1
    rays,inverse,counts=np.unique(rows[:,0].astype(np.int64),return_inverse=True,return_counts=True)
    singleton=(counts[inverse]==1)
    return dict(row_state=state,conditional_interval_index=chosen,overlap_count=overlaps,
        ray_indices=rays,source_hypotheses_per_ray=counts,
        single_source_interior_ray_indices=rows[(state==3)&singleton,0].astype(np.int64),
        membership=None,point_mask_qualified=False,
        meaning='Nearest-face hypotheses only; strict interval containment is necessary, not a structural label')


def describe_endpoint_support(records, intervals_by_source, source_lengths_m):
    """Add boundary descriptions without upgrading the original support states.

    Closed source ends inside the window differ from sphere intersections.
    Exact original cumulative arc lengths are required; no tolerance widening.
    A source endpoint can be an artificial construction seam, not a terminal.
    """
    result=join_surface_interval_support(records,intervals_by_source)
    rows=np.asarray(records,dtype=np.float64)
    contact=np.zeros(len(rows),dtype=np.uint8)  # bit1 start, bit2 end
    cap=np.zeros(len(rows),dtype=bool)
    endpoint_interval=np.full(len(rows),-1,dtype=np.int32)
    crop_contact=np.zeros(len(rows),dtype=bool)
    for source in np.unique(rows[:,1]).astype(int):
        if source not in source_lengths_m:raise ValueError('missing original source arc length')
        length=float(source_lengths_m[source])
        if not np.isfinite(length) or length<=0:raise ValueError('positive finite source arc length required')
        idx=np.flatnonzero(rows[:,1]==source);low=rows[idx,5];high=rows[idx,6]
        if np.any(low<0) or np.any(high>length):raise ValueError('arc outside original source')
        contact[idx]=(low==0).astype(np.uint8)+2*(high==length).astype(np.uint8)
        cap[idx]=(low==high)&(contact[idx]>0)
        interval=intervals_by_source[source]
        if interval['status']!='REFERENCE_INTERVALS_ONLY':continue
        for j,(lo,hi) in enumerate(interval['intervals_m']):
            # A clipped interval end remains uncertain. Being at a source end
            # does not excuse simultaneously crossing the ROI at the other end.
            crop=((lo>0)&(low<=lo)&(high>=lo))|((hi<length)&(low<=hi)&(high>=hi))
            crop_contact[idx]|=crop
            closed_low=(low>lo)|((lo==0)&(low==0))
            closed_high=(high<hi)|((hi==length)&(high==length))
            qualified=(contact[idx]>0)&closed_low&closed_high&(result['overlap_count'][idx]==1)
            endpoint_interval[idx[qualified]]=j
    return dict(**result,source_endpoint_contact_bits=contact,zero_width_source_cap_hypothesis=cap,
        source_endpoint_interval_hypothesis=endpoint_interval,roi_crop_boundary_contact=crop_contact,
        physical_terminal=None,structural_membership=None,
        endpoint_meaning='Source geometry endpoint only, never a terminal label or a confirmed point instance')
