"""Versioned exact event reduction with persistent-occupant witnesses.

This consumes an already validated event stream; it does not certify native
intersection completeness. No geometric tolerance or analytic teacher input.
"""
from fractions import Fraction


def reduce_covered_events(events, inside, maximum=Fraction(50)):
    active = {i for i, x in enumerate(inside) if x}
    first = None
    witnesses = []
    if not active or not events:
        return {'status': 'needs_reference', 'reason': 'missing_origin_or_events'}
    if any(not isinstance(e[0], Fraction) or e[0] <= 0 or e[1] < 0
           or e[1] >= len(inside) or e[3] not in (-1, 1) for e in events):
        return {'status': 'needs_reference', 'reason': 'invalid_event'}
    unique = set(events)
    for root in sorted({e[0] for e in unique}):
        group = [e for e in unique if e[0] == root]
        ids = [e[1] for e in group]
        if len(set(ids)) != len(ids):
            return {'status': 'needs_reference', 'reason': 'multiple_planes_same_operand_root'}
        leaving = {e[1] for e in group if e[3] == 1}
        entering = {e[1] for e in group if e[3] == -1}
        if not leaving <= active or entering & active:
            return {'status': 'needs_reference', 'reason': 'nonalternating_stream'}
        if leaving and entering:
            # An occupant NOT participating in this event covers a full local
            # interval across the root. A merely entering/exiting operand is
            # not such a witness (uncovered touching remains ambiguous).
            persistent = active - set(ids)
            if not persistent:
                return {'status': 'needs_reference', 'reason': 'coincident_entry_exit'}
            witnesses.append((root, tuple(sorted(persistent))))
        was = bool(active)
        active.difference_update(leaving)
        active.update(entering)
        if was and not active and first is None:
            first = (root, tuple(sorted(leaving)))
    # Do not return early: later inconsistent/missing events still invalidate
    # the stream, including the claimed continuous-cover witness.
    if active or first is None:
        return {'status': 'needs_reference', 'reason': 'unclosed_stream'}
    if first[0] > maximum:
        return {'status': 'out_of_range'}
    return {'status': 'candidate', 'exact_t': first[0], 'operands': first[1],
            'unique_events': len(unique), 'covered_mixed_events': witnesses}
