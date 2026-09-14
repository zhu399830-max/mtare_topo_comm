"""Fixed 0.5m 3D response lattice; geometry only, never teacher-driven queries.

Forty cells per axis cover [-10,10]. Cells intersecting the closed radius10
ball are retained; boundary cell centers may be outside the ball. Predictions
are cell-center plus at most0.25m per-axis offset, projected to the ball.
Unknown cells are NOT assumed free. This is not a detector or a training run.
"""
from dataclasses import dataclass
import numpy as np

STEP=.5
SIZE=40
RADIUS=10.


@dataclass(frozen=True)
class CenterResponseLattice:
    centers_m: np.ndarray
    intersects_domain: np.ndarray


def lattice():
    axis=-RADIUS+(np.arange(SIZE,dtype=np.float64)+.5)*STEP
    centers=np.stack(np.meshgrid(axis,axis,axis,indexing='ij'),axis=-1).reshape(-1,3)
    nearest=np.maximum(np.abs(centers)-STEP/2.,0.)
    keep=np.sum(nearest*nearest,axis=1)<=RADIUS**2
    centers.flags.writeable=False;keep.flags.writeable=False
    return CenterResponseLattice(centers,keep)


def cell_indices(positions_m):
    p=np.asarray(positions_m)
    if (p.ndim!=2 or p.shape[1]!=3 or p.dtype.kind!='f' or not np.isfinite(p).all()
            or np.any(np.linalg.norm(p.astype(float),axis=1)>RADIUS+1e-10)):
        raise ValueError('finite Nx3 positions inside the fixed10m ball required')
    # Includes +10 boundary without moving the actual point or adding cells.
    ijk=np.floor((p.astype(float)+RADIUS)/STEP).astype(np.int64).clip(0,SIZE-1)
    return (ijk[:,0]*SIZE+ijk[:,1])*SIZE+ijk[:,2]


def decode_positions(indices,offsets_m):
    ids=np.asarray(indices);offset=np.asarray(offsets_m)
    if (ids.ndim!=1 or ids.dtype.kind not in 'iu' or offset.shape!=(len(ids),3)
            or offset.dtype.kind!='f' or not np.isfinite(offset).all()
            or np.any(np.abs(offset)>STEP/2.+1e-10)
            or np.any(ids<0) or np.any(ids>=SIZE**3)):
        raise ValueError('aligned real lattice indices and bounded cell offsets required')
    fixed=lattice()
    if not fixed.intersects_domain[ids].all():raise ValueError('non-domain cell cannot predict a structure')
    positions=fixed.centers_m[ids]+offset.astype(float)
    scale=np.maximum(1.,np.linalg.norm(positions,axis=1,keepdims=True)/RADIUS)
    return positions/scale
