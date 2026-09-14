"""Return through recorded graph traces, not straight node-to-node chords."""
from copy import deepcopy
import math
import numpy as np
from mtare_topo.planning.topological_frontier import _shortest_paths,_path


def recorded_return_route(*,nodes,edges,pending,current_node,target_node,target_order):
    by_id={n['id']:n for n in nodes}
    if len(by_id)!=len(nodes) or current_node not in by_id or target_node not in by_id:
        raise ValueError('unique recorded node identities required')
    def validate_trace(trace):
        if not trace:raise ValueError('missing traversal trace')
        previous=None
        for s in trace:
            p=np.asarray(s['xyz_m'],float)
            if p.shape!=(3,) or not np.isfinite(p).all() or not math.isfinite(s['order']):raise ValueError('invalid trace sample')
            if previous is not None and s['order']<=previous:raise ValueError('original trace is not causal')
            previous=s['order']
    def at(sample,node):
        return np.allclose(sample['xyz_m'],by_id[node]['xyz_m'],atol=1e-8,rtol=0)
    adjacency={i:[] for i in by_id};links={}
    for edge in edges:
        a,b=edge['source'],edge['target'];trace=edge['trace']
        if a not in by_id or b not in by_id or a==b:raise ValueError('invalid recorded edge')
        validate_trace(trace)
        if not at(trace[0],a) or not at(trace[-1],b):raise ValueError('edge trace does not join its nodes')
        length=sum(float(np.linalg.norm(np.asarray(y['xyz_m'])-x['xyz_m'])) for x,y in zip(trace,trace[1:]))
        if length<=0 or not math.isclose(length,edge['length_m'],rel_tol=1e-8,abs_tol=1e-8):raise ValueError('edge length disagrees with actual trace')
        if (a,b) in links or (b,a) in links:raise ValueError('ambiguous parallel trace edges')
        links[a,b]=trace;links[b,a]=list(reversed(trace))
        adjacency[a].append((b,length));adjacency[b].append((a,length))
    validate_trace(pending)
    if not at(pending[0],current_node):raise ValueError('pending trace not attached to current node')
    distance,predecessor=_shortest_paths(adjacency,current_node)
    if target_node not in distance:raise ValueError('no recorded route to stored place')
    node_path=_path(predecessor,current_node,target_node)
    samples=deepcopy(list(reversed(pending)))
    def append(trace):
        if not np.allclose(samples[-1]['xyz_m'],trace[0]['xyz_m'],atol=1e-8,rtol=0):
            raise ValueError('unrecorded gap in return route')
        samples.extend(deepcopy(trace[1:]))
    for a,b in zip(node_path,node_path[1:]):append(links[a,b])
    # The observation can occur between metric anchors. Stop at its ORIGINAL
    # sample if the return route already crosses it; otherwise append the
    # recorded partial segment from its associated anchor.
    hits=[i for i,s in enumerate(samples) if s['order']==target_order]
    if hits:samples=samples[:hits[0]+1]
    else:
        tails=[]
        for trace in [e['trace'] for e in edges if e['source']==target_node]+([pending] if current_node==target_node else []):
            for i,s in enumerate(trace):
                if s['order']==target_order:tails.append(trace[:i+1])
        if len(tails)!=1:raise ValueError('stored observation has no unique recorded attachment')
        append(tails[0])
    return dict(kind='recorded_trace_return',node_path=list(node_path),samples=samples,
        target_observation_order=target_order,
        length_m=sum(float(np.linalg.norm(np.asarray(y['xyz_m'])-x['xyz_m'])) for x,y in zip(samples,samples[1:])),
        new_edge_inferred=False,local_planner_revalidation_required=True)
