"""Bounded point-to-source-surface evidence, without semantic assignment.

All triangles within an explicitly supplied distance are retained, including
ties and overlapping sheets. Caller must justify the distance from numeric
provenance before using real returns. No closest-axis fallback is permitted.
"""
import numpy as np
from scipy.spatial import cKDTree


def triangle_distances(point, triangles):
    """Euclidean point-to-closed-triangle distances, including degenerate faces."""
    t = np.asarray(triangles,dtype=np.float64)
    p = np.asarray(point,dtype=np.float64)
    a,b,c = t[:,0],t[:,1],t[:,2]
    ab,ac = b-a,c-a
    normal = np.cross(ab,ac)
    n2 = np.einsum('ij,ij->i',normal,normal)
    safe = n2 > 0
    projection = p-np.divide(np.einsum('ij,ij->i',p-a,normal),n2,
                            out=np.zeros(len(t)),where=safe)[:,None]*normal
    ap = projection-a
    # Signed barycentric coordinates from cross products avoid ill-conditioned
    # cancellation of the Gram determinant for long thin sampled triangles.
    v = np.divide(np.einsum('ij,ij->i',np.cross(ap,ac),normal),n2,
                  out=np.zeros(len(t)),where=safe)
    w = np.divide(np.einsum('ij,ij->i',np.cross(ab,ap),normal),n2,
                  out=np.zeros(len(t)),where=safe)
    inside = safe & (v>=0) & (w>=0) & (v+w<=1)
    distances = np.where(inside,np.linalg.norm(p-projection,axis=1),np.inf)
    for x,y in ((a,b),(b,c),(c,a)):
        edge = y-x; e2 = np.einsum('ij,ij->i',edge,edge)
        u = np.clip(np.divide(np.einsum('ij,ij->i',p-x,edge),e2,
                             out=np.zeros(len(t)),where=e2>0),0,1)
        distances = np.minimum(distances,np.linalg.norm(p-(x+u[:,None]*edge),axis=1))
    return distances


class SourceSurfaceMatcher:
    def __init__(self, vertices, faces, *, max_candidates_per_return=200000):
        self.triangles = np.asarray(vertices,dtype=np.float64)[np.asarray(faces)]
        if not np.isfinite(self.triangles).all() or not len(self.triangles):
            raise ValueError('finite nonempty source surface required')
        centers = self.triangles.mean(axis=1)
        self.radius = np.linalg.norm(self.triangles-centers[:,None],axis=2).max()
        self.tree = cKDTree(centers)
        self.max_candidates = max_candidates_per_return

    def nearest_residual(self, point):
        """Exact minimum over the supplied source mesh, not an accepted label.

        A nearest-centroid triangle gives an upper bound; any improving face
        has its centroid within that bound plus the maximum face radius.
        This measures residuals without inventing an acceptance tolerance.
        """
        p = np.asarray(point,dtype=np.float64)
        if p.shape!=(3,) or not np.isfinite(p).all():
            raise ValueError('finite point required')
        _, seed = self.tree.query(p)
        upper = triangle_distances(p,self.triangles[[seed]])[0]
        radius = np.nextafter(upper+self.radius,np.inf)
        if self.tree.query_ball_point(p,radius,return_length=True)>self.max_candidates:
            raise MemoryError('source surface candidate budget exceeded')
        ids = np.asarray(sorted(self.tree.query_ball_point(p,radius)),dtype=np.int64)
        # Include the seed explicitly against roundoff at the broad-phase edge.
        ids = np.union1d(ids,[seed])
        distances = triangle_distances(p,self.triangles[ids])
        minimum = float(distances.min())
        return dict(minimum_distance_m=minimum,
                    minimum_triangle_indices=ids[distances==minimum],
                    membership=None,point_label_qualified=False)

    def match(self, point, *, maximum_distance_m):
        p = np.asarray(point,dtype=np.float64)
        if p.shape!=(3,) or not np.isfinite(p).all() or not np.isfinite(maximum_distance_m) or maximum_distance_m<0:
            raise ValueError('explicit finite point and nonnegative distance required')
        # Outward rounding is only a broad-phase inclusion guard. Final test
        # below retains the caller's original distance without widening it.
        radius = np.nextafter(self.radius+maximum_distance_m,np.inf)
        count = self.tree.query_ball_point(p,radius,return_length=True)
        if count>self.max_candidates:
            raise MemoryError('source surface candidate budget exceeded')
        ids = np.asarray(sorted(self.tree.query_ball_point(p,radius)),dtype=np.int64)
        distances = triangle_distances(p,self.triangles[ids])
        accepted = distances<=maximum_distance_m
        return dict(triangle_indices=ids[accepted],distances_m=distances[accepted],
                    membership=None,point_label_qualified=False)
