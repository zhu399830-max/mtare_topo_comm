"""Confidence-ordered merging under all retained repulsion constraints.

No calibrated probability claim; thresholds belong to the caller. Exact
positive-score ties are simultaneous. A conflicting tied merge is deferred,
never rolled back into or allowed to destroy an earlier consistent group.
"""
from dataclasses import dataclass
from itertools import groupby
import math
from .branch_hypotheses import BranchHypotheses

@dataclass(frozen=True)
class ConflictGroupingResult:
    hypotheses: BranchHypotheses
    decisions: tuple

def group_local_conflicts(ray_ids, relations):
    ids=set(ray_ids)
    if len(ids)!=len(ray_ids) or any(type(i) is not int or i<0 for i in ray_ids):raise ValueError('unique stable ray IDs')
    parent={i:i for i in ids};size={i:1 for i in ids};forbidden={i:set() for i in ids}
    negatives=set();positive=[];seen=set();blocked=[];decisions=[]
    def root(a):
        while parent[a]!=a:parent[a]=parent[parent[a]];a=parent[a]
        return a
    def merge(a,b):
        a,b=root(a),root(b)
        if a==b:return
        if b in forbidden[a]:raise AssertionError('attempt to merge forbidden components')
        if (size[a],-a)<(size[b],-b):a,b=b,a
        parent[b]=a;size[a]+=size[b]
        for n in tuple(forbidden[b]):
            forbidden[n].discard(b);forbidden[n].add(a)
        forbidden[a].update(forbidden[b]);del forbidden[b]
    for a,b,score in relations:
        if a not in ids or b not in ids or a==b:raise ValueError('foreign or self pair')
        pair=tuple(sorted((a,b)))
        if pair in seen:raise ValueError('duplicate pair')
        seen.add(pair)
        if score is None:continue
        if not math.isfinite(score) or not -1<=score<=1:raise ValueError('bounded finite score')
        if score<0:
            negatives.add(pair);forbidden[a].add(b);forbidden[b].add(a)
        elif score>0:positive.append((float(score),pair))
    # Confidence only orders attraction; every threshold-qualified repulsion
    # remains a constraint, including one encountered later in input order.
    for confidence,items in groupby(sorted(positive,key=lambda x:(-x[0],x[1])),key=lambda x:x[0]):
        edges=[pair for _,pair in items];adj={};root_edges=[]
        for a,b in edges:
            ra,rb=root(a),root(b)
            if ra==rb:decisions.append((a,b,confidence,'ALREADY_CONSISTENT'));continue
            if rb in forbidden[ra]:
                blocked.append((a,b));decisions.append((a,b,confidence,'REJECT_REPULSION'));continue
            adj.setdefault(ra,set()).add(rb);adj.setdefault(rb,set()).add(ra);root_edges.append((a,b,ra,rb))
        component_of={};components=[]
        for start in sorted(adj):
            if start in component_of:continue
            index=len(components);todo=[start];component=set()
            while todo:
                x=todo.pop()
                if x in component:continue
                component.add(x);todo.extend(adj[x]-component)
            for x in component:component_of[x]=index
            components.append(component)
        conflict=[any(forbidden[x]&component for x in component) for component in components]
        for a,b,ra,rb in root_edges:
            if conflict[component_of[ra]]:
                blocked.append((a,b));decisions.append((a,b,confidence,'DEFER_EQUAL_CONFIDENCE_CONFLICT'))
            else:
                merge(a,b);decisions.append((a,b,confidence,'ACCEPT_CONSISTENT'))
    groups={}
    for i in sorted(ids):groups.setdefault(root(i),[]).append(i)
    assert all(root(a)!=root(b) for a,b in negatives)
    h=BranchHypotheses(tuple(sorted(tuple(v) for v in groups.values() if len(v)>1)),tuple(sorted(v[0] for v in groups.values() if len(v)==1)),tuple(sorted(negatives)),tuple(sorted(blocked)))
    return ConflictGroupingResult(h,tuple(decisions))
