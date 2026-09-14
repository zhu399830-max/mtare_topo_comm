"""All local reference-axis intervals, not surface masks or traversable cells.

Reuses existing ROI crossing semantics. A source can leave and re-enter the
same window: its global ID cannot pick a local interval. Return-to-axis arc
evidence must be independently established; this module never invents it by
nearest-center assignment or by copying a ray-crossing witness.
"""
import numpy as np
from .gse_roi_crossings_v1 import roi_crossings


def local_axis_intervals(points_m, *, center_m):
    points=np.asarray(points_m,dtype=np.float64)
    center=np.asarray(center_m,dtype=np.float64)
    roots=roi_crossings(points,center_m=center)
    result=dict(status='UNKNOWN',intervals_m=[],membership=None,physical_traversability=False)
    if roots.ambiguous_segment_indices:
        return dict(result,reason='AMBIGUOUS_SOURCE_ROI_CROSSING')
    lengths=np.linalg.norm(np.diff(points,axis=0),axis=-1)
    arcs=np.concatenate(([0.],np.cumsum(lengths)))
    cuts=sorted(set([0.,float(arcs[-1]),*map(float,roots.source_arc_m)]))
    intervals=[]
    for lo,hi in zip(cuts,cuts[1:]):
        if hi<=lo:continue
        middle=(lo+hi)/2
        segment=min(int(np.searchsorted(arcs,middle,side='right')-1),len(lengths)-1)
        if lengths[segment]<=0:return dict(result,reason='DEGENERATE_SOURCE_SEGMENT')
        point=points[segment]+(middle-arcs[segment])/lengths[segment]*(points[segment+1]-points[segment])
        if np.linalg.norm(point-center)<roots.radius_m:intervals.append([lo,hi])
    return dict(result,status='REFERENCE_INTERVALS_ONLY',intervals_m=intervals,
                reason='ALL_SOURCE_ROI_COMPONENTS_RETAINED')


def return_interval_candidates(intervals, *, independently_bound_arc_range_m=None):
    """No caller-provided arc evidence => unknown, even with one local interval.

    Arc evidence refers to the actual returned surface point, not a point where
    its ray crossed the ROI. Caller must bind this semantic provenance outside
    the numerical helper. Output stays a necessary condition, never a label.
    """
    result=dict(status='UNKNOWN',interval_indices=[],membership=None,point_mask_qualified=False)
    if intervals['status']!='REFERENCE_INTERVALS_ONLY':return result
    if independently_bound_arc_range_m is None:
        return dict(result,reason='MISSING_RETURN_ARC_EVIDENCE')
    bounds=np.asarray(independently_bound_arc_range_m,float)
    if bounds.shape!=(2,) or not np.isfinite(bounds).all() or bounds[0]>bounds[1]:
        raise ValueError('finite ordered independently bound return arc range required')
    indices=[i for i,(lo,hi) in enumerate(intervals['intervals_m'])
             if bounds[1]>=lo and bounds[0]<=hi]
    # Only strictly interior support can certify a single numerical candidate.
    strict=[i for i in indices if intervals['intervals_m'][i][0]<bounds[0]
            and bounds[1]<intervals['intervals_m'][i][1]]
    return dict(result,interval_indices=indices,
        status='UNIQUE_REFERENCE_INTERVAL_ONLY' if len(indices)==len(strict)==1 else 'UNKNOWN',
        reason='INDEPENDENT_ARC_RANGE_FILTER_NOT_STRUCTURAL_MEMBERSHIP')
