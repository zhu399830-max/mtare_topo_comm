"""Same causal returns and raw-return attributes for voxel and SPG partitions.

No targets, construction identities or checkpoint enter these functions.
The float32 boundary is shared by both methods and precedes the 10m ROI.
"""
import math
import numpy as np
from mtare_topo.data.cano_sensor_smoke import lidar_local_directions
from .gse_surface_patches_v1 import extract_surface_patches
from .gse_partition_attributes import summarize_partition
from .gse_spg_extractor import extract_spg


def registered_returns(ranges_m, valid_mask, translation_m, yaw_deg):
    r=np.asarray(ranges_m);v=np.asarray(valid_mask)
    t=np.asarray(translation_m);y=np.asarray(yaw_deg)
    if r.shape!=(5,16,720) or v.shape!=r.shape or t.shape!=(5,3) or y.shape!=(5,):
        raise ValueError('exact five-frame sensor shapes required')
    if not all(np.isfinite(a).all() for a in (r,v,t,y)):
        raise ValueError('nonfinite student input')
    if np.any((r<0)|(r>50)) or np.any((v!=0)&(v!=1)):
        raise ValueError('invalid range or return mask')
    if np.any(t[-1]!=0) or y[-1]!=0:
        raise ValueError('current relative pose must be identity')
    if not np.any(v):
        return np.empty((0,3),np.float32),np.empty(0,np.int16),np.empty(0,np.int64)
    # Preserve the legacy nonlearning backprojection convention, then use ONE
    # float32 input for both methods. Do not mix historical cached R1 scores.
    values=np.stack((r.astype(np.float32)/50.,v),axis=1)
    # Same arithmetic/order as legacy _register_points without importing its
    # unrelated scipy.optimize/Fortran baseline dependencies.
    directions=np.asarray(lidar_local_directions(),dtype=np.float64)
    parts=[];slots=[]
    for frame in range(5):
        mask=values[frame,1].astype(bool)
        distance=values[frame,0].astype(np.float64)*50.
        local=directions[mask]*distance[mask,None]
        angle=math.radians(float(y[frame]));c=math.cos(angle);s=math.sin(angle)
        rotated=local.copy()
        rotated[:,0]=c*local[:,0]-s*local[:,1]
        rotated[:,1]=s*local[:,0]+c*local[:,1]
        parts.append(rotated+t[frame]);slots.append(np.full(len(local),frame,np.int16))
    xyz=np.concatenate(parts);frames=np.concatenate(slots)
    xyz=xyz.astype(np.float32)
    indices=np.flatnonzero(v.ravel())
    inside=np.linalg.norm(xyz.astype(np.float64),axis=1)<=10.
    return xyz[inside],frames[inside],indices[inside]


def paired_partitions(points,frames,*,geof_backend,partition_backend):
    if points.dtype!=np.float32 or points.ndim!=2 or points.shape[1]!=3:
        raise ValueError('shared float32 coordinates required')
    if not np.isfinite(points).all() or np.any(np.linalg.norm(points.astype(float),axis=1)>10.):
        raise ValueError('both methods require the same finite 10m ROI')
    r1=extract_surface_patches(points,np.ones(len(points),bool),frames)
    a1=summarize_partition(points,frames,r1.point_patch_index)
    r2=extract_spg(points,frames,geof_backend=geof_backend,partition_backend=partition_backend)
    a2=None
    if r2['original_point_to_component'] is not None:
        a2=summarize_partition(points,frames,r2['original_point_to_component'])
    for attributes in (a1,a2):
        if attributes is not None:
            if attributes.point_count.sum()!=len(points) or attributes.frame_support.sum()!=len(points):
                raise ValueError('original-return mapping or frame support lost')
    return a1,a2,r2
