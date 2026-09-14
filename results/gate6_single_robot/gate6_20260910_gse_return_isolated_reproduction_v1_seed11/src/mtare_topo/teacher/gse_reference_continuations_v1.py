"""Teacher-only source continuation, not physical connectivity or membership.

Suppress only degree-two construction splits with exactly coincident source
endpoints. A displaced anchor connector is not evidence of free space. Keep
every endpoint incidence, junction boundary, unresolved join and pure cycle.
Different returned components are NOT negative membership labels: missing
observation, geometry overlap and reference quality still need checking.
"""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ReferenceSource:
    source_id: str
    node_ids: tuple[str, str]
    endpoint_xyz_m: tuple[tuple[float, float, float], tuple[float, float, float]]


@dataclass(frozen=True)
class ReferenceContinuation:
    source_ids: tuple[str, ...]
    # node, source, endpoint side; two incidences at one node remain distinct.
    structural_boundaries: tuple[tuple[str, str, int], ...]
    unresolved_degree_two_nodes: tuple[str, ...]
    closed_reference_cycle: bool


def reference_continuations(sources):
    sources=tuple(sources)
    if not sources or len(sources)>4096:
        raise ValueError('one to4096 complete reference sources required')
    by_id={};incidence={}
    for source in sources:
        if (type(source) is not ReferenceSource or type(source.source_id) is not str
                or not source.source_id or source.source_id in by_id
                or len(source.node_ids)!=2 or len(source.endpoint_xyz_m)!=2):
            raise ValueError('unique typed two-ended sources required')
        by_id[source.source_id]=source
        for side,(node,xyz) in enumerate(zip(source.node_ids,source.endpoint_xyz_m)):
            if (type(node) is not str or not node or len(xyz)!=3
                    or any(type(v) not in (float,int) or not math.isfinite(v) for v in xyz)):
                raise ValueError('finite source endpoints and explicit node identities required')
            incidence.setdefault(node,[]).append((source.source_id,side,tuple(xyz)))
    parent={key:key for key in by_id}
    def find(key):
        while parent[key]!=key:
            parent[key]=parent[parent[key]];key=parent[key]
        return key
    unresolved=set()
    for node,ends in incidence.items():
        if len(ends)!=2:
            continue
        (a,_,pa),(b,_,pb)=ends
        if pa!=pb:
            unresolved.add(node)
            continue  # no snapping, angle tolerance or invented connector
        ra,rb=find(a),find(b)
        parent[max(ra,rb)]=min(ra,rb)
    components={}
    for source in by_id:
        components.setdefault(find(source),set()).add(source)
    result=[]
    for members in components.values():
        boundaries=[];missing=set()
        for node,ends in incidence.items():
            for source,side,_ in ends:
                if source not in members:
                    continue
                if len(ends)!=2:
                    boundaries.append((node,source,side))
                elif node in unresolved:
                    missing.add(node)
        result.append(ReferenceContinuation(tuple(sorted(members)),tuple(sorted(boundaries)),
            tuple(sorted(missing)),not boundaries and not missing))
    return tuple(sorted(result,key=lambda x:x.source_ids))
