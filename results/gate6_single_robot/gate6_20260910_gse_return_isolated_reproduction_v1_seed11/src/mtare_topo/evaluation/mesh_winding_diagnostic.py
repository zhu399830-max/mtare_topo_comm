"""Independent oriented solid-angle diagnostic; not a safety classifier."""
import numpy as np


def oriented_winding(triangles, points):
    triangles=np.asarray(triangles,dtype=np.float64)
    points=np.asarray(points,dtype=np.float64)
    if triangles.ndim!=3 or triangles.shape[1:]!=(3,3) or not len(triangles):
        raise ValueError('nonempty Tx3x3 triangles required')
    if points.ndim!=2 or points.shape[1:]!=(3,) or not np.isfinite(triangles).all() or not np.isfinite(points).all():
        raise ValueError('finite Nx3 points required')
    result=[]
    for point in points:
        a,b,c=np.moveaxis(triangles-point,1,0)
        na=np.linalg.norm(a,axis=1);nb=np.linalg.norm(b,axis=1);nc=np.linalg.norm(c,axis=1)
        numerator=np.einsum('ij,ij->i',a,np.cross(b,c))
        denominator=(na*nb*nc+np.einsum('ij,ij->i',a,b)*nc+
                     np.einsum('ij,ij->i',b,c)*na+np.einsum('ij,ij->i',c,a)*nb)
        if np.any((numerator==0)&(denominator<=0)):
            raise ValueError('query on triangle/branch boundary is not qualified')
        result.append(np.sum(2*np.arctan2(numerator,denominator),dtype=np.float64)/(4*np.pi))
    return np.asarray(result)
