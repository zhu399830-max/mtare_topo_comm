"""Geometric comparability diagnostics, not truth labels or graph corrections."""
import numpy as np


def clip_axis_to_ball(controls,center,radius=10.):
    axis=np.asarray(controls,float);center=np.asarray(center,float)
    if axis.shape!=(3,3) or center.shape!=(3,) or not np.isfinite(axis).all() or not np.isfinite(center).all() or not np.isfinite(radius) or radius<=0:
        raise ValueError('finite three-control axis and positive ball required')
    segments=[]
    for start,end in zip(axis[:-1],axis[1:]):
        delta=end-start;aa=float(delta@delta)
        if aa==0:continue
        offset=start-center;bb=2*float(offset@delta);cc=float(offset@offset)-radius*radius
        disc=bb*bb-4*aa*cc
        if disc<0:continue
        lo=max(0.,(-bb-np.sqrt(disc))/(2*aa));hi=min(1.,(-bb+np.sqrt(disc))/(2*aa))
        if hi>lo:segments.append(np.stack([start+lo*delta,start+hi*delta]))
    return segments


def sampled_hausdorff(first,second,spacing=.25):
    if not first or not second or spacing<=0:raise ValueError('nonempty clipped axes required')
    def points(segments):
        return np.concatenate([np.linspace(a,b,max(2,int(np.ceil(np.linalg.norm(b-a)/spacing))+1)) for a,b in segments])
    a=points(first);b=points(second);distance=np.linalg.norm(a[:,None]-b[None],axis=-1)
    return float(max(distance.min(axis=0).max(),distance.min(axis=1).max()))
