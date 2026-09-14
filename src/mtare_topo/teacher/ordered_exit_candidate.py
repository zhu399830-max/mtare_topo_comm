"""Fast candidate from an oriented crossing stream, NOT a mesh certificate.

Assumes complete intersections and binary operand interiors. Stream consistency
cannot prove either assumption: deployment must qualify the upstream geometry
and compare to an independent reference. No spatial event grouping is used.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ExitCandidate:
    status: str
    distance_m: float | None = None
    operands: tuple[int, ...] = ()
    reason: str = ''


def ordered_exit_candidate(distances, operands, outward_dots, initial_inside, *, maximum_m=50.):
    ts=np.asarray(distances,float);ids=np.asarray(operands);dots=np.asarray(outward_dots,float)
    initial=np.asarray(initial_inside)
    if ts.ndim!=1 or ids.shape!=ts.shape or dots.shape!=ts.shape:
        raise ValueError('aligned one-dimensional crossing arrays required')
    if initial.ndim!=1 or not len(initial) or not np.isin(initial,[0,1]).all():
        raise ValueError('binary nonempty initial occupancy required')
    if not np.isfinite(maximum_m) or maximum_m<=0:
        raise ValueError('positive finite range required')
    if len(ids) and (ids.dtype.kind not in 'iu' or (ids<0).any() or (ids>=len(initial)).any()):
        raise ValueError('invalid operand index')
    def uncertain(reason):return ExitCandidate('needs_reference',reason=reason)
    if not len(ts):return uncertain('missing_intersections')
    if not np.isfinite(ts).all() or not np.isfinite(dots).all() or (ts<=0).any():
        return uncertain('invalid_crossing')
    if (dots==0).any():return uncertain('tangent_crossing')
    occupancy=initial.astype(bool).copy()
    if not occupancy.any():return uncertain('origin_not_inside')
    first=None
    for distance in np.unique(ts):
        group=np.flatnonzero(ts==distance);sources=ids[group]
        if len(np.unique(sources))!=len(sources):
            return uncertain('duplicate_or_multishell_crossing')
        exits=dots[group]>0
        if exits.any() and not exits.all():return uncertain('coincident_entry_exit')
        if not np.array_equal(occupancy[sources],exits):
            return uncertain('nonalternating_or_missing_crossing')
        before=occupancy.any();occupancy[sources]=~exits
        if before and not occupancy.any() and first is None:
            first=(float(distance),tuple(sorted(int(v) for v in sources)))
    if occupancy.any():return uncertain('unclosed_crossing_stream')
    if first is None:return uncertain('no_supported_exit')
    if first[0]>maximum_m:return ExitCandidate('out_of_range')
    return ExitCandidate('candidate',first[0],first[1])
