"""Partial joint references with separate entering and interior witnesses."""
import numpy as np
from copy import deepcopy
from mtare_topo.data.gse_structure_review_v1 import canonical_sha
from mtare_topo.data.primitive_relation_dataset import _current_sensor_transform
from .gse_reference_anchor_targets_v2 import produce_junction_reference_targets
from .gse_joint_reference_targets_v1 import _produce_joint_reference_targets


def produce_joint_reference_targets(bundle,raw_interfaces):
    return _produce_joint_v2(bundle,raw_interfaces,require_all_rays=True)


def _produce_joint_v2(bundle,raw_interfaces,*,require_all_rays,opening_producer=None,anchor_producer=None):
    result=_produce_joint_reference_targets(bundle,raw_interfaces,anchor_producer or produce_junction_reference_targets,
                                            require_all_rays=require_all_rays,opening_producer=opening_producer)
    _add_interior_correspondence(result,bundle,raw_interfaces)
    result['producer_version']='joint_partial_reference_v2'
    result['source_binding']=dict(
        source={k:deepcopy(bundle['source'][k]) for k in ('task','source_sequence_id','frame_rows')},
        construction_sha256=canonical_sha(bundle['construction_teacher_only']))
    if result['record']['source_frame_indices']!=result['source_binding']['source']['frame_rows']:
        raise ValueError('joint positive targets do not match producer source frames')
    result['target_record_sha256']=canonical_sha(result['record'])
    return result


def _add_interior_correspondence(result,bundle,raw_interfaces,*,witness_field='interior_witness_ray_indices'):
    record=result['record']; provenance=result['teacher_provenance']
    interfaces={r['interface_id_teacher_only']:r for r in raw_interfaces['interfaces_teacher_only']}
    hits={}
    for h in raw_interfaces['raw_interface_intersections']:
        if h['inside_roi']: hits.setdefault(h['ray_index'],[]).append(h)
    sensor=bundle['sensor_teacher_only']
    origin,yaw=sensor['sensor_xyz_m'][-1],float(sensor['yaw_deg'][-1])
    for oi,(opening,evidence) in enumerate(zip(record['openings'],provenance['openings'],strict=True)):
        candidates={}
        rejected=[]
        for ai,anchor in enumerate(provenance['anchors']):
            for key,rays in zip(anchor['interface_ids'],anchor.get(witness_field,
                    [[] for _ in anchor['interface_ids']]),strict=True):
                if interfaces[key]['endpoint_key_teacher_only'][0] != evidence['primitive_id_teacher_only']:
                    continue
                for ray in sorted(set(rays)&set(evidence['crossing_ray_indices'])):
                    other_nodes=set()
                    for hit in hits.get(ray,[]):
                        point=_current_sensor_transform(hit['intersection_world_m'],origin,yaw)
                        if np.dot(np.asarray(opening['position_m'])-point,opening['direction']) > 0:
                            other_nodes.add(interfaces[hit['interface_id_teacher_only']]['node_id_teacher_only'])
                    if other_nodes-{anchor['node_id_teacher_only']}:
                        rejected.append(ray)
                    else:
                        candidates.setdefault(ai,set()).add(ray)
        existing={i for i,v in enumerate(record['membership'][oi]) if v is True}
        supported=existing|set(candidates)
        # Missing witnesses are not contradictory evidence; however a witnessed
        # alternative node or competing interface cannot be ignored to pick a
        # convenient ray. Do not invent negative membership for other anchors.
        if len(supported)==1 and candidates and not rejected:
            ai=next(iter(supported))
            record['membership'][oi][ai]=True
            provenance['relations'][oi]=dict(opening_index=oi,anchor_index=ai,
                status=('PARTIAL_INTERIOR_REFERENCE_CORRESPONDENCE' if witness_field=='interior_witness_ray_indices'
                        else 'PARTIAL_SURFACE_ENTRY_REFERENCE_CORRESPONDENCE'),
                ray_indices=sorted(candidates.get(ai,[])),
                previous_entering_evidence=provenance['relations'][oi])
        elif len(supported)>1 or rejected:
            record['membership'][oi]=[None]*len(record['anchors'])
            provenance['relations'][oi]=dict(opening_index=oi,anchor_index=None,
                status='UNKNOWN_COMPETING_INTERIOR_CORRESPONDENCE',
                candidate_anchor_indices=sorted(supported),blocked_ray_indices=sorted(set(rejected)))
