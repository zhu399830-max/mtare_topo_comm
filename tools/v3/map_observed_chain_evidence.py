"""Read-only field mapping and minimal missing-source scope; no label export."""
from _bootstrap import PROJECT_ROOT as ROOT
import hashlib,json,gzip
from ai_junction_pilot import sha,write

CARD='configs/v3/gate3/data_cards/gse_new_fit_construction_alignment_v1.json'
SPEC='configs/v3/gate3/gse_new_fit_construction_alignment_v1.json'
OUT='docs/figures/gse_supervision_acquisition_pilot_v1/observed_chain_field_mapping.json'

def main():
    card=json.loads((ROOT/CARD).read_text());spec=json.loads((ROOT/SPEC).read_text())
    rows=[];supplement=None
    for i in (9,11):
        e=card['scope']['entries'][i];b=card['scope']['reference_reveal'][i]
        raw=(ROOT/b['path']).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=b['sha256']:raise ValueError('reference drift')
        full=json.loads(gzip.decompress(raw));r=full['raw_interfaces'];p=full['produced_targets']['teacher_provenance']
        if r['source']['task']!=e['task'] or r['source']['frame_rows']!=e['frame_rows']:raise ValueError('identity drift')
        interfaces=r['interfaces_teacher_only'];sources={x['endpoint_key_teacher_only'][0] for x in interfaces}
        nodes={x['node_id_teacher_only'] for x in interfaces}
        row=dict(case=i,reference_path=b['path'],reference_sha256=b['sha256'],
            saved_interface_nodes=sorted(nodes),saved_interface_sources=sorted(sources),
            mapping=dict(frame_ray_identity='AVAILABLE in original witness indices and source frame rows',
                ray_origin_direction_first_return='AVAILABLE or reconstructible from bound original student/sensor, not guessed',
                full_degree2_incidence='AVAILABLE in saved construction, teacher-only',
                original_target_reference='AVAILABLE; not independent observability certificate',
                continuous_source_spans='NOT_STORED as complete bound source intervals',
                target_support='Source witness identities available; per-witness lateral t is not stored in anchor lists',
                source_ids_are_not_intervals=True),new_labels=0)
        if i==11:
            ids=p['openings'][0]['crossing_ray_indices']
            if p['openings'][0]['primitive_id_teacher_only']!='primitive:edge_0009':raise ValueError('opening source changed')
            if len(ids)!=len(set(ids)) or any(type(k)is not int or not 0<=k<57600 for k in ids):raise ValueError('ray identity')
            row['missing_intermediate_node']='node_0011'
            row['intermediate_interface_available']='node_0011' in nodes
            row['start_source_interface_available']='primitive:edge_0009' in sources
            binding=card['scope']['construction_alignment'][i]
            required=[e['student_path'],binding['sensor'],binding['construction'],b['path']]
            for path in required:
                if sha(ROOT/path)!=spec['input_sha256'][path] if path in spec['input_sha256'] else sha(ROOT/path)!=e['student_sha256']:
                    raise ValueError('input drift '+path)
            supplement=dict(status='PROPOSED_NOT_EXECUTED',operation='bounded_missing_geometry_evidence_only',
                case=i,task=e['task'],parent=e['parent'],split='fit',frames=e['frame_rows'],
                frame_ray_indices=ids,ray_count=len(ids),raw_ray_slots=57600,
                per_frame_counts=[sum(k//11520==j for k in ids) for j in range(5)],
                source_ids=['primitive:edge_0009','primitive:edge_0013'],intermediate_node='node_0011',target_node='node_0013',
                input_sha256={path:(spec['input_sha256'][path] if path in spec['input_sha256'] else e['student_sha256']) for path in required},
                geometry_settings=full['produced_targets']['geometry_evidence_settings'],archive_sha256=full['archive_sha256'],
                selection='All original opening0 crossing rays, independent of predicted score or geometry success; no ray removal',
                required_output=['All closed-source intersections per original ray including no-hit cases','Original t,3D position,source/triangle and frame','Origin containment for interval parity','Local clipping and first-return bounds','Ambiguities retained, no inferred labels'],
                no_teacher_target_generation=True,no_training=True,no_radius_or_resolution_change=True,
                limitation='Only missing source evidence: not a new training card or execution approval; exact source/archive/sidecar binding and preflight required before computation')
        rows.append(row)
    result=dict(status='SAVED_FIELD_MAPPING_COMPLETE_NOT_CHAIN_QUALIFICATION',cases=rows,minimal_supplement=supplement,
        tool_sha256=sha(ROOT/'tools/v3/map_observed_chain_evidence.py'),new_labels=0,raycasts=0,training_steps=0)
    write(ROOT/OUT,result)
    print(json.dumps(dict(output=OUT,cases=2,missing_node=rows[1]['missing_intermediate_node'],
        missing_node_interface=not rows[1]['intermediate_interface_available'],rays=supplement['ray_count'],
        per_frame=supplement['per_frame_counts'],computed=False)))

if __name__=='__main__':main()
