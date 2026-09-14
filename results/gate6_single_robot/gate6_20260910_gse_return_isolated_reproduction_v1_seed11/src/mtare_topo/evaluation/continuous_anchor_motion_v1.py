"""Known relative-motion diagnostic, not registration or node association."""
import numpy as np


def rotation(yaw_deg):
    a=np.deg2rad(yaw_deg);c,s=np.cos(a),np.sin(a)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]],dtype=float)


def compose_current_frames(translations,yaws,frame_rows):
    """T_first_current from previous sensor expressed in each current frame.

    Every new window must contain the preceding current frame at slot -2.
    Only past/current motion is consumed. No future smoothing is performed.
    """
    t=np.asarray(translations,dtype=float);y=np.asarray(yaws,dtype=float);f=np.asarray(frame_rows)
    n=len(t)
    if (not 1<=n<=64 or t.shape!=(n,5,3) or y.shape!=(n,5) or f.shape!=(n,5)
            or f.dtype.kind not in 'iu' or np.any(f<0) or not np.isfinite(t).all() or not np.isfinite(y).all()
            or np.any(t[:,-1]!=0) or np.any(y[:,-1]!=0) or np.any(np.diff(f,axis=1)!=1)
            or not np.array_equal(f[1:,:-1],f[:-1,1:])):
        raise ValueError('finite exact adjacent fiveframe motion required')
    rotations=[np.eye(3)];positions=[np.zeros(3)]
    for i in range(1,n):
        r=rotations[-1]@rotation(y[i,-2]).T
        p=positions[-1]-r@t[i,-2]
        # The shared four historical poses must agree under this transform.
        # Tolerance is numerical float32 storage consistency, not acceptance of
        # localization/safety/model error; a mismatch must stop the diagnostic.
        old=positions[-1]+t[i-1,1:]@rotations[-1].T
        new=p+t[i,:-1]@r.T
        if not np.allclose(old,new,atol=2e-5,rtol=0):raise ValueError('overlapping motion poses disagree')
        for k in range(4):
            if not np.allclose(rotations[-1]@rotation(y[i-1,k+1]),r@rotation(y[i,k]),atol=2e-6,rtol=0):
                raise ValueError('overlapping motion orientations disagree')
        rotations.append(r);positions.append(p)
    return np.array(rotations),np.array(positions)


def transform_candidates(local_positions,rotations,positions):
    x=np.asarray(local_positions,dtype=float)
    if (x.shape!=(len(positions),32,3) or not np.isfinite(x).all()
            or rotations.shape!=(len(positions),3,3) or positions.shape!=(len(positions),3)):
        raise ValueError('exact finite32 query positions per frame required')
    return np.einsum('nij,nqj->nqi',rotations,x)+positions[:,None,:]


def set_motion_summary(points,selected):
    """Symmetric nearest-set distances, NOT identity matching or accuracy."""
    values=[]
    for i in range(1,len(points)):
        a=points[i-1][selected[i-1]];b=points[i][selected[i]]
        if not len(a) or not len(b):
            values.append(None);continue
        d=np.linalg.norm(a[:,None]-b[None,:],axis=-1)
        values.append(float((d.min(axis=0).mean()+d.min(axis=1).mean())/2))
    return values
