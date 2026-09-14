"""Minimal given-pose prefix-only map; no GT identity or predicted visibility edges.

One instance is one coordinate/odometry frame. Caller must mark discontinuities.
Provisional nodes confirm after two disjoint five-frame supports. Branches are
not consumed; consequently this backend cannot establish a branch benefit.
"""
from copy import deepcopy
import numpy as np


class SequentialMap:
    def __init__(self):
        self.nodes=[];self.edges=[];self.decisions=[];self.trace=[]
        self.last_visit=None;self.last_time=None

    def update(self, *, timestamp, frame_ids, sensor_to_world, centers_sensor,
               continuous=True):
        matrix=np.asarray(sensor_to_world,float);centers=np.asarray(centers_sensor,float)
        if (not np.isfinite(timestamp) or (self.last_time is not None and timestamp<=self.last_time)
                or matrix.shape!=(4,4) or not np.isfinite(matrix).all()
                or not np.allclose(matrix[3],[0,0,0,1])
                or not np.allclose(matrix[:3,:3].T@matrix[:3,:3],np.eye(3),atol=1e-6)
                or not np.isclose(np.linalg.det(matrix[:3,:3]),1.)
                or centers.ndim!=2 or centers.shape[1]!=3 or not np.isfinite(centers).all()
                or len(frame_ids)!=5 or len(set(frame_ids))!=5):raise ValueError('causal pose and five unique source frames required')
        self.last_time=timestamp
        if not continuous:self.trace=[];self.last_visit=None
        pose=matrix[:3,3];world=centers@matrix[:3,:3].T+pose
        used=set();events=[];support=set(frame_ids)
        for center in world:
            candidates=[i for i,n in enumerate(self.nodes) if i not in used and np.linalg.norm(center-n['xyz'])<=1.]
            if len(candidates)==1:
                i=candidates[0];node=self.nodes[i];count=node['observations']
                node['xyz']=(np.asarray(node['xyz'])*count+center)/(count+1)
                node['observations']+=1
                if all(support.isdisjoint(set(old)) for old in node['independent_support']):
                    node['independent_support'].append(list(frame_ids))
                node['confirmed']=len(node['independent_support'])>=2
            else:
                i=len(self.nodes);self.nodes.append(dict(xyz=center.copy(),observations=1,
                    independent_support=[list(frame_ids)],confirmed=False))
            used.add(i);events.append(dict(node=i,action='associate' if len(candidates)==1 else 'provisional',ambiguous_candidates=candidates if len(candidates)>1 else []))
        sample=dict(timestamp=float(timestamp),xyz=pose.tolist())
        self.trace.append(sample)
        visits=[i for i,n in enumerate(self.nodes) if n['confirmed'] and np.linalg.norm(pose-n['xyz'])<=1.]
        if len(visits)==1:
            current=visits[0]
            if self.last_visit is not None and current!=self.last_visit:
                points=np.asarray([p['xyz'] for p in self.trace])
                self.edges.append(dict(source=self.last_visit,target=current,trace=deepcopy(self.trace),
                    length_m=float(np.linalg.norm(np.diff(points,axis=0),axis=1).sum())))
            if current!=self.last_visit:
                self.last_visit=current;self.trace=[sample]
        self.decisions.append(dict(timestamp=float(timestamp),continuous=continuous,updates=events,visits=visits))
        return self.snapshot()

    def snapshot(self):
        nodes=[dict(**n, index=i) for i,n in enumerate(deepcopy(self.nodes))]
        for n in nodes:n['xyz']=np.asarray(n['xyz']).tolist()
        return dict(nodes=nodes,edges=deepcopy(self.edges),decisions=deepcopy(self.decisions),
            given_pose=True,branch_used=False)
