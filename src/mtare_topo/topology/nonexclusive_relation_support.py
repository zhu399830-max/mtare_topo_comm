"""Non-transitive anchored relation evidence, NOT qualified branch detection.

Anchors must be observation-derived by the caller. High pair scores retain their
original same-group meaning; a shared neighbour never merges two anchors.
"""
from dataclasses import dataclass
import math
import struct

LOW=struct.unpack('f',struct.pack('f',.1))[0]
HIGH=struct.unpack('f',struct.pack('f',.9))[0]

@dataclass(frozen=True)
class RelationSupport:
    ray_ids: tuple
    anchors: tuple
    support_by_anchor: tuple
    repulsion_by_anchor: tuple
    shared_rays: tuple
    unsupported_rays: tuple
    branch_identity_verified: bool = False
    physical_connection_verified: bool = False

def anchored_relation_support(ray_ids, anchors, relations):
    rays=tuple(sorted(ray_ids));centers=tuple(sorted(anchors))
    if len(set(rays))!=len(rays) or len(set(centers))!=len(centers):raise ValueError('duplicate identities')
    if not set(centers)<=set(rays):raise ValueError('foreign anchor')
    support={a:{} for a in centers};repulsion={a:{} for a in centers};seen=set();ray_set=set(rays)
    for a,b,p in relations:
        if a not in ray_set or b not in ray_set or a==b:raise ValueError('foreign/self relation')
        pair=tuple(sorted((a,b)))
        if pair in seen:raise ValueError('duplicate relation')
        seen.add(pair)
        if p is None:continue
        if not math.isfinite(p) or not 0<=p<=1:raise ValueError('invalid probability')
        # Preserve the original float32 .9/.1 threshold boundary exactly.
        for anchor,other in [(a,b),(b,a)]:
            if anchor not in support:continue
            if p>=HIGH:support[anchor][other]=float(p)
            elif p<=LOW:repulsion[anchor][other]=float(p)
    counts={r:0 for r in rays}
    for values in support.values():
        for r in values:counts[r]+=1
    return RelationSupport(rays,centers,
        tuple((a,tuple(sorted(support[a].items()))) for a in centers),
        tuple((a,tuple(sorted(repulsion[a].items()))) for a in centers),
        tuple(r for r in rays if counts[r]>1),tuple(r for r in rays if counts[r]==0))
