"""Eight nearest candidate message neighbors, never physical graph edges."""
import numpy as np
import torch
from scipy.spatial import cKDTree
from .gse_candidate_readout import CandidateGeometryBatch


def candidate_geometry(positions, unary, known):
    n=len(positions)
    if (positions.shape!=(n,3) or unary.shape!=(n,18) or known.shape!=(n,18)
            or known.dtype!=torch.bool or not 1<=n<=4592
            or positions.dtype not in (torch.float32,torch.float64) or unary.dtype!=positions.dtype
            or any(t.device.type!='cpu' for t in (positions,unary,known))
            or not torch.isfinite(positions).all() or not torch.isfinite(unary).all()):
        raise ValueError('finite CPU candidate geometry with known masks required')
    xyz=positions.detach().numpy().astype(np.float64);tree=cKDTree(xyz)
    index=np.full((n,8),-1,dtype=np.int64);valid=np.zeros((n,8),dtype=bool)
    relation=np.zeros((n,8,9));relation[:,:,8]=1.
    for i in range(n):
        count=min(8,n-1)
        if not count:continue
        distances,_=tree.query(xyz[i],k=min(n,count+1))
        radius=float(np.max(distances));eps=64*np.finfo(np.float64).eps*max(1.,radius)
        pool=[j for j in tree.query_ball_point(xyz[i],radius+eps) if j!=i]
        pool.sort(key=lambda j:(float(np.linalg.norm(xyz[j]-xyz[i])),j))
        chosen=pool[:count];index[i,:count]=chosen;valid[i,:count]=True
        delta=xyz[chosen]-xyz[i];relation[i,:count,:3]=delta/10.
        relation[i,:count,3]=np.linalg.norm(delta,axis=1)/10.
        for k,j in enumerate(chosen):
            normals=bool(known[i,3:6].all() and known[j,3:6].all() and unary[i,6]>0 and unary[j,6]>0)
            relation[i,k,5]=float(normals)
            if normals:relation[i,k,4]=float(torch.abs(torch.dot(unary[i,3:6],unary[j,3:6])))
    return CandidateGeometryBatch(unary.detach()[None],torch.ones(1,n,dtype=torch.bool),
        torch.from_numpy(index)[None],torch.from_numpy(valid)[None],
        torch.tensor(relation,dtype=positions.dtype)[None],known[None])
