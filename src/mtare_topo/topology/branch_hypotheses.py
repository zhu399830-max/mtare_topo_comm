"""Deterministic conservative signed-relation grouping, not physical mapping."""
from dataclasses import dataclass
from itertools import groupby
import math

@dataclass(frozen=True,slots=True)
class BranchHypotheses:
    groups: tuple[tuple[int,...],...]
    unresolved_ray_ids: tuple[int,...]
    repulsive_pairs: tuple[tuple[int,int],...]
    blocked_positive_pairs: tuple[tuple[int,int],...]
    physical_connection_verified: bool = False

def group_signed_relations(ray_ids,relations):
    ids=set(ray_ids)
    if len(ids)!=len(ray_ids) or any(type(i) is not int or i<0 for i in ray_ids):raise ValueError('unique stable ray identities')
    edges=[];seen=set();parent={i:i for i in ids};mutex=set();blocked=[]
    def root(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    for a,b,signed in relations:
        if a not in ids or b not in ids or a==b:raise ValueError('foreign/self pair')
        pair=tuple(sorted((a,b)))
        if pair in seen:raise ValueError('duplicate unordered pair')
        seen.add(pair)
        if signed is None:continue
        if not math.isfinite(signed) or not -1<=signed<=1:raise ValueError('bounded signed evidence')
        if signed!=0:edges.append((abs(signed),signed<0,pair))
    # At an exact confidence tie, consider positive chains jointly. If their
    # union conflicts with repulsion, defer the entire tied component rather
    # than choosing a winner by array ordering.
    for _,batch_iter in groupby(sorted(edges,key=lambda e:-e[0]),key=lambda e:e[0]):
        batch=list(batch_iter)
        for _,negative,(a,b) in batch:
            if negative and root(a)!=root(b):mutex.add((a,b))
        adjacency={};positive=[]
        for _,negative,(a,b) in batch:
            if not negative:
                ra,rb=root(a),root(b);positive.append((a,b))
                adjacency.setdefault(ra,set()).add(rb);adjacency.setdefault(rb,set()).add(ra)
        visited=set()
        for start in sorted(adjacency):
            if start in visited:continue
            todo=[start];component=set()
            while todo:
                x=todo.pop()
                if x in component:continue
                component.add(x);todo.extend(adjacency[x]-component)
            visited|=component
            if any(root(a) in component and root(b) in component for a,b in mutex):
                blocked.extend((a,b) for a,b in positive if root(a) in component)
            else:
                representative=min(component)
                for x in component:parent[x]=representative
    groups={}
    for i in sorted(ids):groups.setdefault(root(i),[]).append(i)
    accepted=tuple(sorted(tuple(v) for v in groups.values() if len(v)>1))
    unresolved=tuple(sorted(v[0] for v in groups.values() if len(v)==1))
    return BranchHypotheses(accepted,unresolved,tuple(sorted(mutex)),tuple(sorted(set(blocked))))
