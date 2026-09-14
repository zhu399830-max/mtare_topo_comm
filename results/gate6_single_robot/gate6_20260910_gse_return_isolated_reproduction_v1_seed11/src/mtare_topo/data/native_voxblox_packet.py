"""Student-only text protocol; caller supplies authorized, authenticated inputs."""
import numpy as np


def encode_observation(range_valid, translation, yaw):
    v = np.asarray(range_valid)
    t, y = np.asarray(translation), np.asarray(yaw)
    if (v.shape != (5,2,16,720) or t.shape != (5,3) or y.shape != (5,)
            or not all(np.isfinite(a).all() for a in (v,t,y))
            or np.any((v[:,0]<0)|(v[:,0]>1))
            or np.any((v[:,1]!=0)&(v[:,1]!=1))
            or np.any((v[:,1]==1)&(v[:,0]<=0))
            or np.any(t[-1]!=0) or y[-1]!=0):
        raise ValueError('exact finite five-frame normalized sensor tensors required')
    lines=['GSE_RANGE_V1 5']
    for f in range(5):
        lines.append(' '.join(format(float(x),'.17g') for x in (*t[f], y[f])))
        # Same float32 normalization inverse used by existing sensor baseline.
        ranges=v[f,0].astype(np.float32).ravel()*np.float32(50)
        lines.extend(f'{float(r):.17g} {int(valid)}' for r,valid in zip(ranges,v[f,1].ravel()))
    packet=('\n'.join(lines)+'\n').encode('ascii')
    if len(packet)>4*1024**2: raise ValueError('bounded input packet exceeded')
    return packet
