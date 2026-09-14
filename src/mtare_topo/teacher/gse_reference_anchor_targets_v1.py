"""Partial junction reference-position supervision from bound inward evidence.

Not complete semantic annotation: openings, unknown background and physical
reachability remain unsupervised. Real export requires its own frozen scope.
"""
import numpy as np
from mtare_topo.data.primitive_relation_dataset import _current_sensor_transform
from .gse_construction_paths_v3 import construction_incident_paths
from .gse_directed_interface_binding_v1 import bind_axes,interpret_bound_result


def produce_junction_reference_targets(bundle,raw_result):
    return _produce_junction_reference_targets(bundle,raw_result)


def _produce_junction_reference_targets(bundle,raw_result,interior=None,additional=None,*,cap_source_settings=None):
    # Recompute the interpreted evidence with original source/motion/XYZ checks;
    # never trust a supplied "junction_supported=True" assertion.
    directed=interpret_bound_result(bundle,raw_result)
    cap_audit=None
    if cap_source_settings is not None:
        from mtare_topo.evaluation.gse_bound_cap_precision import audit_bound_cap_precision
        cap_audit=audit_bound_cap_precision(bundle,raw_result,**cap_source_settings)
        for key,evidence in directed['interfaces'].items():
            qualified=cap_audit['interfaces'][key]['stable_entering_ray_indices']
            if not set(qualified)<=set(evidence['entering_ray_indices']):
                raise ValueError('cap qualification cannot invent entering witnesses')
            evidence['entering_ray_indices']=list(qualified)
    groups=construction_incident_paths(bundle['construction_teacher_only'])
    axes=bind_axes(groups,raw_result['interfaces_teacher_only'])
    by_node={}
    for axis in axes:
        key=axis['interface_id_teacher_only']
        entering=directed['interfaces'][key]['entering_ray_indices']
        rays=sorted(set(entering) | set(interior['interface_ray_indices'][key] if interior is not None else []))
        if additional is not None:
            rays=sorted(set(rays) | set(additional['entry_ray_indices'][key]) | set(additional['departure_ray_indices'][key]))
        if rays:
            by_node.setdefault(axis['node_id_teacher_only'],[]).append((axis,rays))
    sensor=bundle['sensor_teacher_only'];origin=sensor['sensor_xyz_m'][-1];yaw=float(sensor['yaw_deg'][-1])
    anchors=[];provenance=[];unknown=[]
    for group in groups:
        node=group['node_id_teacher_only']
        if len(group['paths'])<3 or np.linalg.norm(np.asarray(group['anchor_world_m'])-origin)>=10.:
            continue
        support=by_node.get(node,[])
        # An extra hidden branch must not erase the visible three-way event.
        # Coincident same-direction references do not supply distinct layout.
        directions={tuple(a['inward_direction']) for a,_ in support}
        if len(directions)<3:
            unknown.append(dict(node_id_teacher_only=node,reason='FEWER_THAN_THREE_DISTINCT_OBSERVED_INWARD_DIRECTIONS'))
            continue
        position=_current_sensor_transform(np.asarray(group['anchor_world_m']),origin,yaw).tolist()
        anchors.append(dict(position_m=position,evidence='Partial reference position: at least three distinct bound finite inward source-interface directions; not complete local annotation.'))
        provenance.append(dict(node_id_teacher_only=node,interface_ids=[a['interface_id_teacher_only'] for a,_ in support],
            witness_ray_indices=[list(rays) for _,rays in support],position_source='construction_reference_not_observed_point_estimator'))
        if cap_audit is not None:
            provenance[-1]['cap_precision_policy']='source_float64_float32_interval_v1'
            provenance[-1]['entering_witness_ray_indices']=[directed['interfaces'][a['interface_id_teacher_only']]['entering_ray_indices'] for a,_ in support]
        if interior is not None:
            anchors[-1]['evidence']='Partial reference position: three distinct bound branch directions from entering or source-interior observations; not complete local annotation.'
            provenance[-1]['entering_witness_ray_indices']=[directed['interfaces'][a['interface_id_teacher_only']]['entering_ray_indices'] for a,_ in support]
            provenance[-1]['interior_witness_ray_indices']=[interior['interface_ray_indices'][a['interface_id_teacher_only']] for a,_ in support]
        if additional is not None:
            anchors[-1]['evidence']='Partial reference: three observed branch directions from cap, interior or bound lateral overlap entry/departure; not complete annotation.'
            provenance[-1]['surface_entry_witness_ray_indices']=[additional['entry_ray_indices'][a['interface_id_teacher_only']] for a,_ in support]
            provenance[-1]['surface_departure_witness_ray_indices']=[additional['departure_ray_indices'][a['interface_id_teacher_only']] for a,_ in support]
    if len(anchors)>32:raise OverflowError('anchor capacity exceeded; no target truncation')
    record=dict(schema='gse_surface_observed_targets_v1',coordinate_frame='current_sensor_m',
        source_frame_indices=list(bundle['source']['frame_rows']),anchors=anchors,openings=[],membership=[],
        score_region=dict(center_m=[0.,0.,0.],radius_m=10.,anchors_complete=False,openings_complete=False,
            evidence='No complete-background certificate supplied; unmatched predictions remain unknown.'))
    result=dict(record=record,teacher_provenance=provenance,unknown_candidates=unknown,
        supervision_status='PARTIAL_REFERENCE_POSITION_ONLY',full_training_gate_eligible=False,
        missing_tasks=['independent_openings','opening_membership','complete_scoring_regions'],
        human_reviewed=False,scientific_gate_pass=False)
    if cap_audit is not None:
        result['cap_precision_audit']=cap_audit
    return result
