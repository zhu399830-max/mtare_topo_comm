"""Direction-compatible branch witnesses, not negative membership labels.

Virtual source interfaces may coincide or overlap. Their crossing parameters
must be before the first return, but no relative crossing order is imposed.
The caller must bind the arrays to the original scan and construction. A
witness alone does not certify an anchor, ownership, or physical separation.
"""
from .gse_directed_interface_evidence_v1 import directed_evidence


def branch_transition_witness(records, interfaces, *, leaving_interface,
                              entering_interface, directions, first_return,
                              valid, return_sources):
    lookup = {row['interface_id_teacher_only']: row for row in interfaces}
    if (len(lookup) != len(interfaces) or leaving_interface not in lookup
            or entering_interface not in lookup or leaving_interface == entering_interface):
        raise ValueError('two distinct, uniquely bound source interfaces required')
    interpreted = directed_evidence(records, interfaces, directions=directions,
        first_return=first_return, valid=valid, return_sources=return_sources)
    left = lookup[leaving_interface]
    right = lookup[entering_interface]
    same_node = left['node_id_teacher_only'] == right['node_id_teacher_only']
    outgoing = set(interpreted['interfaces'][leaving_interface]['leaving_ray_indices'])
    incoming = set(interpreted['interfaces'][entering_interface]['entering_ray_indices'])
    rays = sorted(outgoing & incoming) if same_node else []
    return dict(
        status=('LOCAL_DIRECTION_WITNESS_ONLY' if rays else
                'UNKNOWN_NO_COMMON_DIRECTED_RAY' if same_node else 'UNKNOWN_DIFFERENT_REFERENCE_NODES'),
        ray_indices=rays, leaving_interface=leaving_interface,
        entering_interface=entering_interface, membership=None,
        physical_separation_certified=False,
    )
