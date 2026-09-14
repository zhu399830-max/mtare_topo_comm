"""Prospective supplementary sampling; not supervision or model proposals.

No filesystem access. Select the traversal before seeing poses or scans.
Keep all failures; never substitute a more visible structure or direction.
"""
import math

from .gse_structural_inventory_v1 import inventory_parent
from .gse_surface_selection_v1 import VARIANTS, stable_rank


def nominate_parent(report, split, variant_groups):
    """One hash-fixed incident traversal per entity, all eligible entities.

    Original 16-edge observations remain the independent background stratum;
    this function neither replaces them nor labels them as negatives.
    """
    inventory = inventory_parent(report, split, variant_groups)
    intervals = {r['traversal_id']: r for r in report['intervals']}
    nominations = []
    for entity in inventory['entities']:
        candidates = entity['causal_traversals']
        traversal = min(candidates, key=lambda t: (
            stable_rank('structure_supplement_direction_v1', entity['parent_id'],
                        entity['node_id_teacher_only'], t), t)) if candidates else None
        nomination = dict(entity, traversal_id=traversal,
                          status='POSE_BINDING_REQUIRED' if traversal else 'NO_CAUSAL_HISTORY')
        # Copy only source metadata, never construction coordinates into observations.
        nomination['variant_records'] = ([] if traversal is None else
            sorted(intervals[traversal]['variants'], key=lambda r: VARIANTS.index(r['variant'])))
        nominations.append(nomination)
    return nominations


def choose_position(nomination, *, decision_xyz_by_variant, anchor_xyz_by_variant):
    """Choose one shared sequence minimizing worst-variant 3D distance.

    Ties use source sequence identity. Distances beyond the fixed 10m scope
    are retained and flagged, not resampled. Proximity is NOT visibility.
    Positions must be supplied by a separately hash-bound source reader.
    """
    if nomination['status'] != 'POSE_BINDING_REQUIRED':
        raise ValueError('a causal nomination is required')
    if (set(decision_xyz_by_variant) != set(VARIANTS) or
            set(anchor_xyz_by_variant) != set(VARIANTS)):
        raise ValueError('all three variants required')
    records = {r['variant']: r for r in nomination['variant_records']}
    if set(records) != set(VARIANTS):
        raise ValueError('paired source records required')
    ids = records[VARIANTS[0]]['source_sequence_ids']
    if not ids or any(r['source_sequence_ids'] != ids for r in records.values()):
        raise ValueError('shared nonempty sequence identities required')
    def xyz(value):
        if (len(value) != 3 or any(isinstance(x, bool) or not math.isfinite(x) for x in value)):
            raise ValueError('finite three-dimensional position required')
        return value
    distances = {}
    for v in VARIANTS:
        positions = decision_xyz_by_variant[v]
        if len(positions) != len(ids):
            raise ValueError('pose rows must correspond to every nominated decision')
        anchor = xyz(anchor_xyz_by_variant[v])
        distances[v] = [math.dist(xyz(p), anchor) for p in positions]
    index = min(range(len(ids)), key=lambda i: (max(distances[v][i] for v in VARIANTS), ids[i]))
    sources = []
    for v in VARIANTS:
        r = records[v]
        sources.append(dict(task=r['task'], variant=v, source_sequence_id=ids[index],
            sequence_row=r['sequence_rows'][index], frame_rows=list(r['frame_rows'][index]),
            decision_route_arc_m=r['decision_arc_m'][index]))
    return dict(parent_id=nomination['parent_id'], split=nomination['split'],
        node_id_teacher_only=nomination['node_id_teacher_only'], kind=nomination['kind'],
        traversal_id=nomination['traversal_id'], decision_index=index, sources=sources,
        distances_m={v: distances[v][index] for v in VARIANTS},
        all_variants_within_10m=all(distances[v][index] <= 10.0 for v in VARIANTS),
        observable_label=False, continuous_route_evidence=False)


def merge_source_requests(background, selections):
    """Deduplicate exports, not entities; preserve all nomination memberships.

    A window shared by two structures is one input, not two independent scans.
    Background must be the sealed original rows passed by the source reader.
    """
    unique = {}
    def add(source, role):
        key = (source['task'], source['source_sequence_id'])
        binding = {k: source[k] for k in ('task', 'source_sequence_id', 'sequence_row', 'frame_rows')}
        if key not in unique:
            unique[key] = dict(source=binding, sampling_roles=[])
        elif unique[key]['source'] != binding:
            raise ValueError('conflicting source row/history for one observation')
        if role not in unique[key]['sampling_roles']:
            unique[key]['sampling_roles'].append(role)
    for row in background:
        add(row, 'original_unfiltered_background')
    for selection in selections:
        role = selection['parent_id'] + ':' + selection['node_id_teacher_only']
        for source in selection['sources']:
            add(source, role)
    output = [unique[key] for key in sorted(unique)]
    for row in output:
        row['sampling_roles'].sort()
    return output
