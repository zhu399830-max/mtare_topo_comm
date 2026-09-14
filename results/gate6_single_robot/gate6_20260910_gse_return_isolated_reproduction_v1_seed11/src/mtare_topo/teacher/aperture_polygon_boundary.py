"""Whole-spherical-cell bounds for finite constant polygonal straight tubes.

Reject curved/tapered operands instead of changing their geometry. This is a
geometry-only numerical adapter; observed support is a separate input.
"""
from dataclasses import dataclass
import numpy as np
from .csg_mesh_provenance import mesh_swept_superellipse


@dataclass(frozen=True)
class FinitePolygonPrism:
    normals: np.ndarray
    offsets: np.ndarray
    numerical_padding_m: float


def straight_primitive_prism(primitive):
    points=primitive.centerline_xyz_m
    if len(points)!=2 or primitive.endpoint_half_axes_m[0]!=primitive.endpoint_half_axes_m[1] or primitive.endpoint_shape_exponent[0]!=primitive.endpoint_shape_exponent[1]:
        raise ValueError('only original two-point constant-section straight operands supported')
    mesh=mesh_swept_superellipse(primitive,axial_spacing_m=.05,angular_segments=64)
    rings=mesh.vertices_xyz_m[:-2].reshape(-1,64,3);first,last=rings[0],rings[-1]
    axis=points[1]-points[0];length=np.linalg.norm(axis);axis/=length
    edges=np.roll(first,-1,axis=0)-first
    normals=np.cross(edges,axis);normals/=np.linalg.norm(normals,axis=1)[:,None]
    midpoint=points.mean(axis=0)
    offsets=np.einsum('ij,ij->i',normals,first)
    flip=normals@midpoint>offsets;normals[flip]*=-1;offsets[flip]*=-1
    padding=256*np.finfo(float).eps*max(1.,length,float(np.max(np.abs(mesh.vertices_xyz_m))))
    # Each corresponding side strip must stay on its declared plane.
    residual=np.einsum('ijk,jk->ij',rings,normals)-offsets
    if np.max(np.abs(residual))>padding:raise ValueError('mesh side strips are not the declared constant prism')
    normals=np.concatenate([normals,[-axis,axis]])
    offsets=np.r_[offsets,-axis@points[0],axis@points[1]]
    return FinitePolygonPrism(normals,offsets,padding)


def classify_polygon_union(vertices,faces,prisms,*,center_m=(0.,0.,0.),radius_m=10.):
    vertices=np.asarray(vertices,dtype=float);faces=np.asarray(faces,dtype=int);center=np.asarray(center_m,dtype=float)
    if center.shape!=(3,) or not np.isfinite(center).all() or not np.isfinite(radius_m) or radius_m<=0:
        raise ValueError('finite sphere required')
    if not np.allclose(np.linalg.norm(vertices,axis=1),1.,rtol=0,atol=1e-12):raise ValueError('unit sphere required')
    tri=vertices[faces];direction=tri.sum(axis=1);direction/=np.linalg.norm(direction,axis=1)[:,None]
    # A convex spherical covering cap containing all three vertices contains
    # their normalized positive combinations, i.e. the spherical triangle.
    if np.any(np.einsum('ijk,ik->ij',tri,direction)<=0):raise ValueError('nonconvex cell covering cap')
    chord=np.linalg.norm(tri-direction[:,None],axis=2).max(axis=1)+256*np.finfo(float).eps
    cell_center=center+radius_m*direction;bound=radius_m*chord
    free=np.zeros(len(faces),dtype=bool);blocked=np.ones(len(faces),dtype=bool)
    for p in prisms:
        if not np.allclose(np.linalg.norm(p.normals,axis=1),1.,rtol=0,atol=1e-12):raise ValueError('unit halfspace normals required')
        values=cell_center@p.normals.T-p.offsets
        error=p.numerical_padding_m+256*np.finfo(float).eps*(radius_m+float(np.max(np.abs(center)))+1.)
        free|=np.all(values+bound[:,None]<-error,axis=1)
        blocked&=np.any(values-bound[:,None]>error,axis=1)
    if np.any(free&blocked):raise ValueError('contradictory whole-cell bounds')
    states=np.full(len(faces),-1,dtype=np.int8);states[free]=1;states[blocked]=0
    return states
