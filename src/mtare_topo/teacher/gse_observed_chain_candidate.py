"""Candidate supervision contract, NOT integrated into any teacher or training.

Checks one explicitly supplied, source-bound ray chain. A positive result is
only a conditional geometric witness, never robot reachability, an aperture,
complete annotation, or independent verification of its input intervals.
No graph search through unobserved construction is performed.
"""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class SourceSpan:
    source: str
    t0: float
    t1: float
    # Actual source incidence at the end closer to the target structure.
    exit_node: str


@dataclass(frozen=True)
class ObservedJoin:
    node: str
    # Complete construction incidence, not just the visible subset.
    incident_sources: tuple[str, ...]
    # Same ray parameter at an independently bound transition event.
    t: float
    from_source: str
    to_source: str
    evidence_id: str


@dataclass(frozen=True)
class RayChain:
    frame_slot: int
    ray_index: int
    origin_xyz: tuple[float, float, float]
    direction_xyz: tuple[float, float, float]
    first_return_m: float
    observed_start_m: float
    observed_end_m: float
    start_source: str
    target_node: str
    target_incident_sources: tuple[str, ...]
    target_evidence_id: str
    spans: tuple[SourceSpan, ...]
    joins: tuple[ObservedJoin, ...]
    conflicting_nodes: tuple[str, ...] = ()


def qualify_candidate(chain: RayChain):
    """Return reasoned unknown or a candidate; no conversion to training bool."""
    def unknown(reason):
        return {'status': 'UNKNOWN', 'reason': reason, 'training_qualified': False}

    if type(chain.frame_slot) is not int or chain.frame_slot not in range(5):
        raise ValueError('five causal slots only')
    if type(chain.ray_index) is not int or chain.ray_index not in range(11520):
        raise ValueError('original within-frame ray identity required')
    numbers = (*chain.origin_xyz, *chain.direction_xyz, chain.first_return_m,
               chain.observed_start_m, chain.observed_end_m)
    if len(chain.origin_xyz) != 3 or len(chain.direction_xyz) != 3 or not all(math.isfinite(v) for v in numbers):
        raise ValueError('finite3D ray required')
    if not any(chain.direction_xyz) or chain.first_return_m <= 0:
        raise ValueError('nondegenerate observed return required')
    # All t values use the original parameterization; direction need not be unit.
    if not 0 <= chain.observed_start_m < chain.observed_end_m <= chain.first_return_m:
        return unknown('NOT_WITHIN_OBSERVED_PRE_RETURN_INTERVAL')
    # The sphere is convex: endpoints inside imply the straight witnessed
    # interval is inside. Coordinates must be current-sensor3D, not world XY.
    for t in (chain.observed_start_m,chain.observed_end_m):
        point=[o+t*d for o,d in zip(chain.origin_xyz,chain.direction_xyz)]
        if sum(x*x for x in point)>100.:
            return unknown('OUTSIDE_ORIGINAL_10M_DOMAIN')
    if not chain.spans or len(chain.joins) != len(chain.spans)-1:
        return unknown('MISSING_CHAIN_SPANS_OR_TRANSITIONS')
    if chain.conflicting_nodes:
        return unknown('CONFLICTING_STRUCTURE_EVIDENCE')
    sources=[s.source for s in chain.spans]
    if not all(sources) or len(set(sources)) != len(sources):
        return unknown('REPEATED_OR_EMPTY_SOURCE')
    if sources[0] != chain.start_source:
        return unknown('START_SOURCE_MISMATCH')
    for span in chain.spans:
        if not all(math.isfinite(t) for t in (span.t0,span.t1)) or not 0 <= span.t0 < span.t1 <= chain.first_return_m:
            return unknown('UNOBSERVED_OR_INVALID_SPAN')
    if not chain.spans[0].t0 <= chain.observed_start_m < chain.spans[0].t1:
        return unknown('START_NOT_COVERED')
    previous_t=chain.observed_start_m
    for left,right,join in zip(chain.spans,chain.spans[1:],chain.joins):
        if len(join.incident_sources)!=2 or len(set(join.incident_sources))!=2:
            return unknown('INTERMEDIATE_NOT_DEGREE_TWO')
        if set(join.incident_sources)!={left.source,right.source}:
            return unknown('INCIDENCE_MISMATCH')
        if left.exit_node!=join.node or not join.node or join.node==chain.target_node:
            return unknown('TRANSITION_NODE_MISMATCH')
        if join.from_source!=left.source or join.to_source!=right.source:
            return unknown('DIRECTED_TRANSITION_MISMATCH')
        if not join.evidence_id:
            return unknown('NO_OBSERVED_TRANSITION')
        if not math.isfinite(join.t) or not previous_t <= join.t <= chain.observed_end_m:
            return unknown('NONCAUSAL_OR_REVERSED_TRANSITION')
        # No epsilon or XY snapping: the SAME ray parameter must lie in both
        # bound 3D source spans. Disjoint layers/gaps cannot be bridged.
        if not max(left.t0,right.t0) <= join.t <= min(left.t1,right.t1):
            return unknown('GAP_OR_UNBOUND_TRANSITION')
        previous_t=join.t
    last=chain.spans[-1]
    if not last.t0 <= chain.observed_end_m <= last.t1:
        return unknown('TARGET_INTERVAL_NOT_COVERED')
    if last.exit_node != chain.target_node or last.source not in chain.target_incident_sources:
        return unknown('TARGET_INCIDENCE_MISMATCH')
    if not chain.target_evidence_id:
        return unknown('NO_OBSERVED_TARGET_WITNESS')
    return {'status':'CONDITIONAL_RAY_CHAIN_CANDIDATE','target_node_teacher_only':chain.target_node,
            'sources_teacher_only':sources,'evidence_ids':[j.evidence_id for j in chain.joins]+[chain.target_evidence_id],
            'training_qualified':False,'physical_reachability_claim':False,
            'complete_background':False}
