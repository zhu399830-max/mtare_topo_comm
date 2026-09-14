"""Necessary two-sided observation evidence for terminal correspondence.

Consumes source-bound witnesses, not network inputs. This check is separate
from source-axis component matching and never supplies negative membership.
"""
import numpy as np


def terminal_relation_witness(*, source_id, node_id, cap_rays, opening_rays,
                              source_ids, origin_distances, interface_hits,
                              interfaces, first_return, frame_rows):
    distances = np.asarray(origin_distances, dtype=float)
    ranges = np.asarray(first_return, dtype=float).reshape(-1)
    if (len(source_ids) != len(set(source_ids)) or source_id not in source_ids
            or distances.shape != (5, len(source_ids)) or np.isnan(distances).any()
            or np.isneginf(distances).any() or ranges.shape != (57600,)
            or len(frame_rows) != 5 or any(b <= a for a, b in zip(frame_rows, frame_rows[1:]))):
        raise ValueError('bound original five-frame population required')
    sets = []
    for rays in (cap_rays, opening_rays):
        if any(type(i) is not int or not 0 <= i < 57600 for i in rays):
            raise ValueError('invalid original ray index')
        sets.append(set(rays))
    cap, opening = sets
    if any(not np.isfinite(ranges[i]) or ranges[i] < 0 for i in cap | opening):
        raise ValueError('witness needs finite first return')
    lookup = {i['interface_id_teacher_only']: i for i in interfaces}
    if len(lookup) != len(interfaces):
        raise ValueError('duplicate interface identity')
    blocked = set()
    for hit in interface_hits:
        ray = hit['ray_index']
        if ray not in cap | opening:
            continue
        interface = lookup[hit['interface_id_teacher_only']]
        t = hit['t']
        if not np.isfinite(t) or t < 0:
            raise ValueError('invalid interface parameter')
        if (hit['inside_roi'] and t <= ranges[ray]
                and interface['node_id_teacher_only'] != node_id):
            blocked.add(ray)
    # Do not cherry-pick a clean ray while hiding contradictory node evidence.
    if blocked:
        return dict(status='UNKNOWN_INTERVENING_NODE', common_frame_slots=[],
                    blocked_ray_indices=sorted(blocked), membership=None)
    index = source_ids.index(source_id)
    common = {i // 11520 for i in cap} & {i // 11520 for i in opening}
    supported = [f for f in sorted(common)
                 if distances[f, index] < 0 and np.sum(distances[f] < 0) == 1]
    return dict(status='TWO_SIDED_SOURCE_WITNESS_ONLY' if supported else 'UNKNOWN_NO_COMMON_SOURCE_ORIGIN',
                common_frame_slots=supported, blocked_ray_indices=[], membership=None)
