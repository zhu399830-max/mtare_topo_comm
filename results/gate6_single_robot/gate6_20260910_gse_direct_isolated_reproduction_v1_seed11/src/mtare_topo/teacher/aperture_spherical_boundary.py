"""Conservative whole-cell classification for declared radial circular tubes.

Only an analytic synthetic adapter: off-center/curved/real meshes and LiDAR
visibility are NOT inferred here. Each cap is an open free spherical region.
"""
import numpy as np


def octahedral_sphere(level=4):
    if type(level) is not int or not 0<=level<=6:raise ValueError('bounded subdivision level required')
    vertices=[(1.,0.,0.),(-1.,0.,0.),(0.,1.,0.),(0.,-1.,0.),(0.,0.,1.),(0.,0.,-1.)]
    faces=[(x,y,z) for x in (0,1) for y in (2,3) for z in (4,5)]
    for _ in range(level):
        mids={};new=[]
        def midpoint(a,b):
            key=tuple(sorted((a,b)))
            if key not in mids:
                p=np.asarray(vertices[a])+vertices[b];p/=np.linalg.norm(p)
                mids[key]=len(vertices);vertices.append(tuple(p))
            return mids[key]
        for a,b,c in faces:
            ab=midpoint(a,b);bc=midpoint(b,c);ca=midpoint(c,a)
            new.extend(((a,ab,ca),(ab,b,bc),(ca,bc,c),(ab,bc,ca)))
        faces=new
    vertices=np.asarray(vertices);faces=np.asarray(faces,dtype=int)
    edge_faces={};adj=[set() for _ in faces]
    for i,face in enumerate(faces):
        for a,b in zip(face,np.roll(face,-1)):
            edge_faces.setdefault(tuple(sorted((int(a),int(b)))),[]).append(i)
    for members in edge_faces.values():
        if len(members)!=2:raise ValueError('boundary must be closed two-face adjacency')
        a,b=members;adj[a].add(b);adj[b].add(a)
    return vertices,faces,tuple(tuple(sorted(a)) for a in adj)


def classify_radial_caps(vertices,faces,axes,angular_radii):
    vertices=np.asarray(vertices,dtype=float);faces=np.asarray(faces,dtype=int)
    axes=np.asarray(axes,dtype=float).reshape(-1,3);radii=np.asarray(angular_radii,dtype=float)
    if radii.shape!=(len(axes),) or not np.isfinite(axes).all() or not np.isfinite(radii).all() or np.any(radii<=0) or np.any(radii>=np.pi/2):
        raise ValueError('finite open hemispherical caps required')
    if np.any(np.linalg.norm(axes,axis=1)==0):raise ValueError('nonzero cap axes required')
    if not np.allclose(np.linalg.norm(vertices,axis=1),1.,rtol=0,atol=1e-12):raise ValueError('unit sphere vertices required')
    axes=axes/np.linalg.norm(axes,axis=1)[:,None]
    triangles=vertices[faces];center=triangles.sum(axis=1);center/=np.linalg.norm(center,axis=1)[:,None]
    cover=np.arccos(np.clip(np.einsum('ijk,ik->ij',triangles,center),-1,1)).max(axis=1)
    if np.any(cover>=np.pi/2):raise ValueError('whole-cell covering cap must be convex')
    distance=np.arccos(np.clip(center@axes.T,-1,1))
    # Conservative angular padding solely for finite-precision calculations;
    # boundary cells remain UNKNOWN instead of selecting a favorable side.
    padding=64*np.finfo(np.float64).eps
    whole_free=np.any(distance+cover[:,None]+padding<radii,axis=1)
    whole_blocked=np.all(distance-cover[:,None]-padding>radii,axis=1)
    states=np.full(len(faces),-1,dtype=np.int8);states[whole_free]=1;states[whole_blocked]=0
    return states
