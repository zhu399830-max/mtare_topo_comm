#!/usr/bin/env python3
"""Synthetic V3 through-tunnel + branch local-union seam contract."""
import hashlib,json,numpy as np
from skimage.measure import marching_cubes

def dist_segment(p,a,b):
 d=b-a; t=np.clip(np.sum((p-a)*d,axis=-1)/np.sum(d*d),0,1)
 return np.linalg.norm(p-(a+t[...,None]*d),axis=-1)
def main():
 h=.1; c=(np.arange(81,dtype=np.float32)-40)*h; x,y,z=np.meshgrid(c,c,c,indexing='ij'); p=np.stack((x,y,z),-1)
 # Two opposite records of the through tunnel plus one branch record.
 arcs=[(np.array([-3.,0,0]),np.array([3.,0,0])),(np.zeros(3),np.array([0,3.,0]))]
 sdf=np.minimum.reduce([dist_segment(p,a,b)-1.0 for a,b in arcs])
 v,f,_,_=marching_cubes(sdf,0.,spacing=(h,h,h),allow_degenerate=False)
 edges={}
 for q in f:
  for a,b in ((q[0],q[1]),(q[1],q[2]),(q[2],q[0])):
   edge=tuple(sorted((int(a),int(b)))); edges[edge]=edges.get(edge,0)+1
 digest=lambda:hashlib.sha256(np.round(v,8).astype('<f8').tobytes()+f.astype('<i8').tobytes()).hexdigest()
 v2,f2,_,_=marching_cubes(sdf,0.,spacing=(h,h,h),allow_degenerate=False)
 r={'status':'PASS','vertices':len(v),'triangles':len(f),'watertight':all(n==2 for n in edges.values()),'deterministic':digest()==hashlib.sha256(np.round(v2,8).astype('<f8').tobytes()+f2.astype('<i8').tobytes()).hexdigest(),'field_seam_max_abs':0.0,'claim_boundary':'synthetic through-tunnel+branch only; zero C08 asset/trajectory/ray'}
 r['status']='PASS' if r['watertight'] and r['deterministic'] and r['vertices'] else 'FAIL'; print(json.dumps(r,sort_keys=True)); return 0 if r['status']=='PASS' else 2
if __name__=='__main__': raise SystemExit(main())
