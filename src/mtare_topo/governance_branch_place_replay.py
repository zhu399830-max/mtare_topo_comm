from pathlib import Path

SCHEMA='v3_branch_place_replay_card_v1'
SLUG='gse_branch_place_replay_v1'
PARENT='results/gate6_single_robot/gate6_20260910_gse_live_geometry_execution_v3_seed11'
TRACE=PARENT+'/artifacts/geometry/live_geometry.jsonl'
POLICY=dict(direction_tolerance_deg=15,target_match_m=2,place_radius_m=4,support_observations=3)

def scope(root):
    entries=dict((r.split('  ',1)[1],r.split('  ',1)[0]) for r in (root/PARENT/'artifacts/evidence_sha256.txt').read_text().splitlines())
    return dict(trace=TRACE,sha256=entries[TRACE],worlds=['tunnel'],episodes=1,rows=292,
        geometry_observations=288,split='previously_used_development_episode',
        timing='all saved causal observations, no resampling; actual timestamps preserved',
        policy=POLICY,teacher_reads=0,new_control=False,new_training=False)

def validate_card(card):
    from .governance import ValidationReport
    a=card.get('approval',{});errors=[]
    if card.get('schema_version')!=SCHEMA or card.get('scope')!=scope(Path(__file__).resolve().parents[2]):errors.append('sealed episode scope drift')
    if a.get('status')!='APPROVED' or a.get('authorized_gates')!=[6] or a.get('authorized_operations')!=['audit']:errors.append('exact audit authority required')
    return ValidationReport(not errors,tuple(errors),())


def validate_mechanism_card(card):
    """Saved-input intervention has no training/validation/test population."""
    import hashlib
    import json
    from .governance import ValidationReport
    root=Path(__file__).resolve().parents[2]
    parent='results/gate6_single_robot/gate6_20260910_gse_learned_constrained_execution_v1_seed11'
    s=card.get('scope',{});a=card.get('approval',{});errors=[]
    seal={p:h for h,p in (line.split('  ',1) for line in (root/parent/'artifacts/evidence_sha256.txt').read_text().splitlines())}
    expected={p:seal[p] for p in [parent+'/artifacts/sensors.bag',parent+'/artifacts/geometry/live_geometry.jsonl']}
    if s.get('source_sha256')!=expected:errors.append('exact sealed source binding required')
    if (s.get('worlds')!=['tunnel'] or s.get('episodes')!=1 or s.get('raw_scans')!=293
            or s.get('effective_observations')!=289 or len(set(s.get('source_frame_keys',[])))!=293):
        errors.append('fixed development population required')
    if s.get('teacher')!='none' or s.get('training') is not False or s.get('protected_worlds_read') is not False:
        errors.append('no labels, training or protected worlds')
    if s.get('interventions')!=['original retained axes','reflect retained axes about sensor, unchanged scan','rotate retained axes 90 degrees about sensor Z, unchanged scan']:
        errors.append('fixed interventions required')
    if (a.get('status')!='APPROVED' or a.get('authorized_gates')!=[6] or a.get('authorized_operations')!=['audit']
            or a.get('scope_sha256')!=hashlib.sha256(json.dumps(s,sort_keys=True).encode()).hexdigest()):
        errors.append('scope-bound audit authority required')
    return ValidationReport(not errors,tuple(errors),())


def validate_local_pair_card(card):
    import hashlib
    import json
    from .governance import ValidationReport
    s=card.get('scope',{});a=card.get('approval',{});errors=[]
    entries=s.get('entries',[])
    if (len(entries)!=132 or len({e['parent'] for e in entries})!=44 or s.get('reference_pairs')!=77
            or s.get('radius_m')!=10 or s.get('training') is not False or s.get('scan_reads') is not False):
        errors.append('fixed44-parent/77-pair coordinate-only scope required')
    for e in entries:
        if e['parent'].rsplit('_',1)[-1] not in {'C01','C02','C03','C04','C05','C06','C07'}:
            errors.append('protected parent')
        if not e['coordinate_prefix'].endswith('/sensor_xyz_m'):errors.append('coordinate-only source required')
        if set(e['coordinate_plan']['selected_rows'])!={f for o in e['observations'] for f in o['frame_rows']}:
            errors.append('source rows mismatch')
    if (a.get('status')!='APPROVED' or a.get('authorized_gates')!=[3] or a.get('authorized_operations')!=['audit']
            or a.get('scope_sha256')!=hashlib.sha256(json.dumps(s,sort_keys=True).encode()).hexdigest()):
        errors.append('scope-bound audit authority required')
    return ValidationReport(not errors,tuple(errors),())


def validate_full_local_pair_card(card):
    import hashlib
    import json
    from .governance import ValidationReport
    s=card.get('scope',{});a=card.get('approval',{});errors=[];entries=s.get('entries',[])
    fields=['sensor_xyz_m','traversal_index','local_frame_index','frame_row','source_global_sequence_index']
    if (len(entries)!=132 or len({e['parent'] for e in entries})!=44 or s.get('reference_pairs')!=77
            or s.get('radius_m')!=10 or s.get('training') is not False or s.get('scan_reads') is not False or s.get('fields')!=fields):
        errors.append('fixed coordinate/index-only scope required')
    for e in entries:
        if e['parent'].rsplit('_',1)[-1] not in {'C01','C02','C03','C04','C05','C06','C07'}:errors.append('protected parent')
        if set(e['arrays'])!=set(fields):errors.append('unexpected payload')
        for f,plan in e['arrays'].items():
            if not plan['prefix'].endswith('/'+f):errors.append('field prefix mismatch')
    if (a.get('status')!='APPROVED' or a.get('authorized_gates')!=[3] or a.get('authorized_operations')!=['audit']
            or a.get('scope_sha256')!=hashlib.sha256(json.dumps(s,sort_keys=True).encode()).hexdigest()):errors.append('scope-bound authority required')
    return ValidationReport(not errors,tuple(errors),())


def validate_local_pair_pilot_card(card):
    import hashlib
    import json
    from .governance import ValidationReport
    s=card.get('scope',{});a=card.get('approval',{});errors=[];rows=s.get('observations',[])
    if s.get('strata')!=159 or not rows or len(rows)>159 or s.get('training') is not False or s.get('labels_generated')!=0:
        errors.append('fixed pilot input-only population required')
    if s.get('student_fields')!=['ranges_m','valid_mask','relative_translation_current_sensor_m','relative_yaw_current_sensor_deg']:
        errors.append('student leakage boundary changed')
    if len({(r['task'],r['source_global_sequence_index']) for r in rows})!=len(rows):errors.append('duplicate source observation')
    if any(r['parent'].rsplit('_',1)[-1] not in {'C01','C02','C03','C04','C05','C06','C07'} for r in rows):errors.append('protected parent')
    if (a.get('status')!='APPROVED' or a.get('authorized_gates')!=[3] or a.get('authorized_operations')!=['data_export']
            or a.get('scope_sha256')!=hashlib.sha256(json.dumps(s,sort_keys=True).encode()).hexdigest()):errors.append('scope-bound export authority required')
    return ValidationReport(not errors,tuple(errors),())


def validate_saved_positive_origins_card(card):
    import hashlib,json
    from .governance import ValidationReport
    s=card.get('scope',{});a=card.get('approval',{});errors=[]
    entries=s.get('entries',[]);rows=[r for e in entries for r in e['selected']]
    if len(entries)!=210 or len(rows)!=839 or len({(r['source']['task'],r['source']['source_sequence_id']) for r in rows})!=839:errors.append('all839 unique positives required')
    if s.get('split_counts')!={'fit':725,'calibration':57,'development':57}:errors.append('split drift')
    if len({r['source']['parent_id'] for r in rows})!=70 or any(r['source']['parent_id'].rsplit('_',1)[-1] not in {'C01','C02','C03','C04','C05','C06','C07'} for r in rows):errors.append('parent scope drift')
    if any(e['prefix'].rsplit('/',1)[-1]!='sensor_xyz_m' for e in entries) or any(r['original_membership'] is not True for r in rows):errors.append('coordinate-only old positive scope required')
    if any(s.get(k) is not False for k in ('ray_reads','labels_changed','training')) or s.get('field_spacing_m')!=.025:errors.append('fixed prerequisite-only scope required')
    if s.get('limits')!=dict(address_space_bytes=3*1024**3,wall_seconds=1800,output_bytes=1024**3):errors.append('resource drift')
    if a.get('status')!='APPROVED' or a.get('authorized_gates')!=[3] or a.get('authorized_operations')!=['data_export'] or a.get('scope_sha256')!=hashlib.sha256(json.dumps(s,sort_keys=True).encode()).hexdigest():errors.append('bound approval required')
    return ValidationReport(not errors,tuple(errors),())


def validate_continuation_probe_card(card):
    import hashlib
    import json
    from .governance import ValidationReport
    s=card.get('scope',{});a=card.get('approval',{});errors=[]
    expected={('S07_flat_loop_rich_C03__c1_mixed',96960),('S07_flat_loop_rich_C03__ellipse',96957),
        ('S08_3d_loop_rich_C01__c1_mixed',114960),('S08_3d_loop_rich_C01__ellipse',114960),('S08_3d_loop_rich_C01__rounded_rectangle',114960)}
    rows=s.get('rows',[])
    if len(rows)!=5 or {(r['source']['task'],r['source']['source_global_sequence_index']) for r in rows}!=expected:errors.append('fixed five observations required')
    if any(r['source']['split']!='fit' or len(r['source']['frame_rows'])!=5 for r in rows):errors.append('fit-only five-frame scope required')
    if any(s.get(k) is not False for k in ('labels_changed','ray_recast','training','independent_evaluation')):errors.append('prerequisite-only boundary required')
    if s.get('field_spacing_m')!=.025 or s.get('resources')!=dict(address_space_bytes=3*1024**3,wall_seconds=600,output_bytes=1024**3):errors.append('fixed geometry/resources required')
    if a.get('status')!='APPROVED' or a.get('authorized_gates')!=[3] or a.get('authorized_operations')!=['data_export'] or a.get('scope_sha256')!=hashlib.sha256(json.dumps(s,sort_keys=True).encode()).hexdigest():errors.append('scope-bound approval required')
    return ValidationReport(not errors,tuple(errors),())


def validate_local_pair_support_card(card):
    import hashlib
    import json
    from .governance import ValidationReport
    s=card.get('scope',{});a=card.get('approval',{});errors=[];rows=s.get('observations',[])
    if len(rows)!=159 or len({(r['task'],r['source_global_sequence_index']) for r in rows})!=159:
        errors.append('exact159 distinct fixed inputs required')
    expected={'fit':(144,32),'calibration':(9,3),'development':(6,1)}
    for split,(n,p) in expected.items():
        chosen=[r for r in rows if r['split']==split]
        if len(chosen)!=n or len({r['parent'] for r in chosen})!=p:errors.append('population drift '+split)
    if any(r['parent'].rsplit('_',1)[-1] not in {'C01','C02','C03','C04','C05','C06','C07'} for r in rows):errors.append('protected parent')
    if s.get('geometry_settings')!=dict(axial_spacing_m=.05,angular_segments=64,field_spacing_m=.025) or s.get('qualify_cap_precision') is not True:
        errors.append('frozen V8 geometry required')
    if s.get('archive',{}).get('sha256')!='b963b3207c65b2ed69504f4d6040e9b94a05d18f9305dc35ca0ca2a3ff04efea':errors.append('archive mismatch')
    if not {'unknown_not_background','no_training','no_test','no_root_reachability_claim','automatic_partial_labels_not_human_review','one_development_parent_not_generalization'}<=set(s.get('restrictions',[])):
        errors.append('supervision boundary missing')
    if s.get('resources')!=dict(parent_address_space_bytes=1024**3,worker_address_space_bytes=3*1024**3,wall_seconds=10800,output_bytes=30*1024**3):errors.append('resource drift')
    if (a.get('status')!='APPROVED' or a.get('authorized_gates')!=[3] or a.get('authorized_operations')!=['data_export']
            or a.get('scope_sha256')!=hashlib.sha256(json.dumps(s,sort_keys=True).encode()).hexdigest()):errors.append('scope-bound support authority required')
    return ValidationReport(not errors,tuple(errors),())
