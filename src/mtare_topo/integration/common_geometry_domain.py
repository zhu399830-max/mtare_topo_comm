"""Shared geometric domain transport; clipping never certifies an opening."""
from copy import deepcopy
import numpy as np
from mtare_topo.evaluation.local_axis_diagnostic import clip_axis_to_ball


def common_domain_record(record,radius_m=10.):
    out=deepcopy(record);out['primitives']=[];out['structures']=[]
    center=np.asarray(record['sensor_to_local_odometry'])[:3,3]
    rejected=[]
    for primitive in record['primitives']:
        axis=primitive['axis_controls_world_m']
        if axis is None:
            rejected.append([primitive['index'],'no_axis']);continue
        segments=clip_axis_to_ball(axis,center,radius_m)
        if not segments:
            rejected.append([primitive['index'],'outside_common_domain']);continue
        # Keep a bend only when the two clipped pieces really meet.
        if len(segments)==2 and np.allclose(segments[0][-1],segments[1][0],rtol=0,atol=1e-10):
            pieces=[np.stack([segments[0][0],segments[0][1],segments[1][1]])]
        else:
            pieces=[np.stack([a,(a+b)/2,b]) for a,b in segments]
        for piece in pieces:
            p=deepcopy(primitive);p['parent_primitive_index']=p['index'];p['index']=len(out['primitives'])
            p['axis_controls_world_m']=piece.tolist()
            p['common_domain_clipped']=True
            out['primitives'].append(p)
    out['domain_rejections']=rejected;out['common_domain_radius_m']=radius_m
    out['physical_openings_confirmed']=False
    return out
