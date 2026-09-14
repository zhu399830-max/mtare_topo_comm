"""Join partial references with conservative, ray-witnessed correspondence.

This is construction-conditioned supervision, not a physical route certificate.
Absence of a witnessed correspondence never supplies a negative label.
"""
from copy import deepcopy
import numpy as np

from mtare_topo.data.primitive_relation_dataset import _current_sensor_transform
from .gse_reference_anchor_targets_v1 import produce_junction_reference_targets
from .gse_reference_opening_targets_v1 import produce_opening_reference_targets
from .gse_reference_terminal_targets_v1 import produce_terminal_reference_targets


def produce_joint_reference_targets(bundle, raw_interfaces):
    return _produce_joint_reference_targets(bundle,raw_interfaces,produce_junction_reference_targets)


def _produce_joint_reference_targets(bundle, raw_interfaces, anchor_producer, *, require_all_rays=True, opening_producer=None):
    # The anchor entry point binds and checks every raw intersection, including
    # those belonging to nodes that do not become positive anchor targets.
    anchors = anchor_producer(bundle, raw_interfaces)
    openings = (opening_producer or produce_opening_reference_targets)(bundle)
    record = deepcopy(anchors['record'])
    if (record['source_frame_indices'] != openings['record']['source_frame_indices']
            or record['coordinate_frame'] != openings['record']['coordinate_frame']):
        raise ValueError('partial target observation mismatch')
    record['openings'] = deepcopy(openings['record']['openings'])
    record['membership'] = [[None] * len(record['anchors']) for _ in record['openings']]
    sensor = bundle['sensor_teacher_only']
    origin, yaw = sensor['sensor_xyz_m'][-1], float(sensor['yaw_deg'][-1])
    interfaces = {r['interface_id_teacher_only']: r for r in raw_interfaces['interfaces_teacher_only']}
    by_ray = {}
    for hit in raw_interfaces['raw_interface_intersections']:
        if hit['inside_roi']:
            by_ray.setdefault(hit['ray_index'], []).append(hit)
    relation_evidence = []
    for oi, (opening, evidence) in enumerate(zip(record['openings'], openings['teacher_provenance'], strict=True)):
        candidates = {}
        blocked_rays = []
        nodes_by_ray = {}
        for ray in evidence['crossing_ray_indices']:
            hits = []
            for hit in by_ray.get(ray, []):
                point = _current_sensor_transform(hit['intersection_world_m'], origin, yaw)
                # The ray exits this opening in its outward direction. Its
                # source-interface hit must be on the interior side first.
                if np.dot(np.asarray(opening['position_m']) - point, opening['direction']) > 0:
                    hits.append(hit)
            nodes = {interfaces[h['interface_id_teacher_only']]['node_id_teacher_only'] for h in hits}
            nodes_by_ray[ray] = nodes
            if len(nodes) != 1:
                blocked_rays.append(ray)
                continue
            witnessed = False
            for ai, anchor in enumerate(anchors['teacher_provenance']):
                if anchor['node_id_teacher_only'] not in nodes:
                    continue
                for interface, witnesses in zip(anchor['interface_ids'], anchor.get('entering_witness_ray_indices',anchor['witness_ray_indices']), strict=True):
                    if ray in witnesses and any(
                        h['interface_id_teacher_only'] == interface
                        and h['source_key_teacher_only'] == evidence['primitive_id_teacher_only'] for h in hits
                    ):
                        candidates.setdefault(ai, set()).add(ray)
                        witnessed = True
            if not witnessed:
                blocked_rays.append(ray)
        # Conflicting or intervening-node evidence cannot be hidden by picking
        # a convenient ray. Unknown pairs remain unknown, not false.
        conflicting_rays = []
        if len(candidates) == 1:
            candidate_node = anchors['teacher_provenance'][next(iter(candidates))]['node_id_teacher_only']
            conflicting_rays = [ray for ray,nodes in nodes_by_ray.items() if nodes - {candidate_node}]
        veto = blocked_rays if require_all_rays else conflicting_rays
        if len(candidates) == 1 and not veto:
            ai = next(iter(candidates))
            record['membership'][oi][ai] = True
            relation_evidence.append(dict(opening_index=oi, anchor_index=ai,
                ray_indices=sorted(candidates[ai]), status='PARTIAL_REFERENCE_CORRESPONDENCE'))
        else:
            relation_evidence.append(dict(opening_index=oi, anchor_index=None,
                candidate_anchor_indices=sorted(candidates), blocked_ray_indices=blocked_rays,
                status='UNKNOWN_NO_UNIQUE_DIRECT_WITNESS'))
        if not require_all_rays:
            relation_evidence[-1]['missing_direct_witness_ray_indices'] = blocked_rays
            relation_evidence[-1]['conflicting_node_ray_indices'] = conflicting_rays
    terminals = produce_terminal_reference_targets(bundle)
    terminal_provenance = []
    terminal_unknown = []
    if terminals is not None:
        if terminals['record']['source_frame_indices'] != record['source_frame_indices']:
            raise ValueError('terminal observation mismatch')
        extra = terminals['record']['anchors']
        if len(record['anchors']) + len(extra) > 32:
            raise OverflowError('joint anchor capacity exceeded; no target truncation')
        record['anchors'].extend(deepcopy(extra))
        if len({tuple(a['position_m']) for a in record['anchors']}) != len(record['anchors']):
            raise ValueError('coincident anchor references; no silent merging')
        for row in record['membership']:
            row.extend([None] * len(extra))
        terminal_provenance = terminals['teacher_provenance']
        terminal_unknown = terminals['unknown_candidates']
    return dict(record=record,
        teacher_provenance=dict(anchors=anchors['teacher_provenance'],
            terminals=terminal_provenance, terminal_anchor_start=len(anchors['record']['anchors']),
            openings=openings['teacher_provenance'], relations=relation_evidence),
        unknown_candidates=dict(anchors=anchors['unknown_candidates'], openings=openings['unknown_candidates'],
            terminals=terminal_unknown),
        supervision_status='PARTIAL_REFERENCE_GEOMETRY_AND_CORRESPONDENCE',
        full_training_gate_eligible=False, human_reviewed=False, scientific_gate_pass=False,
        missing_tasks=['complete_scoring_regions', 'reference_target_quality', 'terminal_opening_correspondence'])
