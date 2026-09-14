"""Rigid five-scan registration with original return indices preserved.

Direction proposals remain the historical horizontal-sector estimator: a
rotated sector is projected onto current XY for grouping. This does not turn
that estimator into a layered 3D opening detector. Surface fitting retains XYZ.
"""
import numpy as np
from mtare_topo.data.cano_sensor_smoke import lidar_local_directions


def validate_rotations(rotations):
    r=np.asarray(rotations,float)
    if (r.shape!=(5,3,3) or not np.isfinite(r).all()
            or not np.allclose(np.swapaxes(r,1,2)@r,np.eye(3),atol=1e-6)
            or not np.allclose(np.linalg.det(r),1.,atol=1e-6)
            or not np.allclose(r[-1],np.eye(3),atol=1e-6)):
        raise ValueError('five proper rotations into current scan frame required')
    return r


def register_returns(values,translations,rotations):
    r=validate_rotations(rotations)
    t=np.asarray(translations,float);v=np.asarray(values)
    if t.shape!=(5,3) or not np.isfinite(t).all() or not np.allclose(t[-1],0):
        raise ValueError('five finite current-frame translations required')
    if (v.shape!=(5,2,16,720) or not np.isfinite(v).all()
            or np.any(v[:,0]<0) or np.any(v[:,0]>1)
            or np.any((v[:,1]!=0)&(v[:,1]!=1))
            or np.any((v[:,1]==1)&(v[:,0]<=0))):
        raise ValueError('normalized original ranges and binary validity required')
    directions=np.asarray(lidar_local_directions(),float)
    points=[]
    for i in range(5):
        mask=v[i,1].astype(bool)
        local=directions[mask]*(v[i,0].astype(np.float64)[mask,None]*50.)
        points.append(local@r[i].T+t[i])
    return np.concatenate(points),np.flatnonzero(v[:,1].reshape(-1))


def rotate_sector_mask(mask,rotation):
    columns=np.flatnonzero(mask)
    angle=np.radians(columns*.5)
    directions=np.stack((np.cos(angle),np.sin(angle),np.zeros(len(angle))),axis=1)
    rotated=directions@rotation.T
    if np.any(np.linalg.norm(rotated[:,:2],axis=1)<1e-8):
        raise ValueError('vertical direction cannot be represented by horizontal proposal grouping')
    target=np.rint((np.degrees(np.arctan2(rotated[:,1],rotated[:,0]))%360)*2).astype(int)%720
    result=np.zeros(720,bool);result[target]=True
    return result
