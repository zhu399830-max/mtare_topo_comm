"""Validation for the single fixed partial-reference fitting diagnostic."""
import hashlib
import json


def validate_card(card):
    from .governance import ValidationReport
    errors=[]
    try:
        s=card['scope'];a=card['approval'];b=s['binding']
        assert card['schema_version']=='gse_membership_fit_card_v1'
        assert a['status']=='APPROVED' and a['authorized_operations']==['training'] and a['authorized_gates']==[3]
        assert a['scope_sha256']==hashlib.sha256(json.dumps(s,sort_keys=True).encode()).hexdigest()
        assert s['updates']==500 and s['seed']==0 and s['variant']=='C'
        assert s['learning_rate']==.001 and s['weight_decay']==.0001 and s['accumulation']==4
        assert b['selected_windows']==16 and b['selected_unique_frames']==80
        assert b['decoded_observations_per_uncached_pass']==105 and len(b['entries'])==16
        assert len({e['source']['parent_id'] for e in b['entries']})==8
        assert all(e['source']['split']=='fit' and e['source']['parent_id'].rsplit('_',1)[1] in
                   {'C01','C02','C03','C04','C05','C06'} for e in b['entries'])
        assert s['unknown_is_background'] is False and s['teacher_in_forward'] is False
        if 'membership_gradient_to_shared' in s:
            assert s['membership_gradient_to_shared'] is False
            assert s['initial_checkpoint']=='results/gate3_semantics/gate3_20260911_gse_membership_fit_v1r1_seed0/artifacts/initial.pt'
        assert s['limits']==dict(wall_seconds=7200,host_bytes=32*1024**3,gpu_bytes=28*1024**3,output_bytes=4*1024**3)
    except (KeyError,TypeError,AssertionError,IndexError):errors.append('fixed partial-reference fitting contract drift')
    return ValidationReport(passed=not errors,errors=errors,warnings=[])


def validate_observed_regions_card(card):
    from .governance import ValidationReport
    errors=[]
    try:
        s=card['scope'];a=card['approval'];e=s['entry']
        assert card['schema_version']=='gse_observed_regions_card_v1'
        assert (s['observations'],s['parents'],s['frames'],s['raw_ray_slots'],s['valid_returns'])==(1,1,5,57600,56792)
        assert e['task']=='S04_3d_unicyclic_small_C01__ellipse' and e['split']=='fit' and e['frame_rows']==[301,302,303,304,305]
        assert s['resolution_m']==.25 and s['radius_m']==10 and s['references_only_after_component_freeze'] is True
        assert s['training'] is False and s['new_labels']==0 and s['unknown_is_background'] is False
        assert s['limits']==dict(wall_seconds=600,host_bytes=4*1024**3,output_bytes=100*1024**2)
        assert a['status']=='APPROVED' and a['authorized_operations']==['audit'] and a['authorized_gates']==[3]
        assert a['scope_sha256']==hashlib.sha256(json.dumps(s,sort_keys=True).encode()).hexdigest()
    except (KeyError,TypeError,AssertionError):errors.append('observed-region diagnostic scope drift')
    return ValidationReport(passed=not errors,errors=errors,warnings=[])


def validate_chain_geometry_card(card):
    from .governance import ValidationReport
    errors=[]
    try:
        s=card['scope'];a=card['approval'];e=s['entry'];ids=e['original_ray_indices']
        assert card['schema_version']=='gse_chain_geometry_card_v1'
        assert (s['observations'],s['parents'],s['frames'],s['raw_ray_slots'],s['selected_rays'])==(1,1,5,57600,2774)
        assert e['task']=='S04_3d_unicyclic_small_C01__ellipse' and e['parent']=='S04_3d_unicyclic_small_C01' and e['split']=='fit'
        assert e['frame_rows']==[301,302,303,304,305]
        assert len(ids)==2774 and ids==sorted(set(ids)) and all(type(i)is int and 0<=i<57600 for i in ids)
        assert [sum(i//11520==j for i in ids) for j in range(5)]==[732,623,535,470,414]
        assert s['sources']==['primitive:edge_0009','primitive:edge_0013']
        assert s['geometry_settings']==dict(axial_spacing_m=.05,angular_segments=64,field_spacing_m=.025)
        assert s['limits']==dict(worker_address_space_bytes=3*1024**3,wall_seconds=600,output_bytes=100*1024**2)
        assert s['teacher_calls']==0 and s['new_labels']==0 and s['training'] is False and s['unknown_is_background'] is False
        assert a['status']=='APPROVED' and a['authorized_operations']==['audit'] and a['authorized_gates']==[3]
        assert a['scope_sha256']==hashlib.sha256(json.dumps(s,sort_keys=True).encode()).hexdigest()
    except (KeyError,TypeError,AssertionError,ValueError):errors.append('exact two-source geometry supplement contract drift')
    return ValidationReport(passed=not errors,errors=errors,warnings=[])


def validate_lateral_recovery_card(card):
    from .governance import ValidationReport
    errors=[]
    try:
        s=card['scope'];a=card['approval']
        assert card['schema_version']=='gse_lateral_recovery_card_v1'
        assert s['observations']==3 and s['parents']==3 and s['selected_original_rays']==21440
        assert [e['ray_count'] for e in s['entries']]==[10440,8830,2170]
        for e in s['entries']:
            ids=e['original_ray_indices'];assert len(ids)==e['ray_count'] and ids==sorted(set(ids))
            assert all(type(i) is int and 0<=i<57600 for i in ids) and e['source']['split']=='fit'
        assert s['geometry_settings']==dict(axial_spacing_m=.05,angular_segments=64,field_spacing_m=.025)
        assert s['limits']==dict(worker_address_space_bytes=3*1024**3,parent_rss_bytes=1024**3,wall_seconds=1800,output_bytes=1024**3)
        assert a['status']=='APPROVED' and a['authorized_operations']==['audit'] and a['authorized_gates']==[3]
        assert a['scope_sha256']==hashlib.sha256(json.dumps(s,sort_keys=True).encode()).hexdigest()
    except (KeyError,TypeError,AssertionError):errors.append('lateral recovery contract drift')
    return ValidationReport(passed=not errors,errors=errors,warnings=[])


def validate_common_gradient_card(card):
    from .governance import ValidationReport
    errors=[]
    try:
        s=card['scope'];a=card['approval']
        assert card['schema_version']=='gse_common_gradient_card_v1'
        assert validate_common_fit_card(s['training_card']).passed
        assert s['optimizer_steps']==0 and s['observation_indices']==list(range(16))
        assert s['checkpoint'].endswith('gse_common_structure_fit_v1_seed0/artifacts/final.pt')
        assert s['limits']==dict(wall_seconds=600,host_bytes=32*1024**3,gpu_bytes=28*1024**3,output_bytes=1024**3)
        assert a['status']=='APPROVED' and a['authorized_operations']==['audit'] and a['authorized_gates']==[3]
        assert a['scope_sha256']==hashlib.sha256(json.dumps(s,sort_keys=True).encode()).hexdigest()
    except (KeyError,TypeError,AssertionError):errors.append('common final checkpoint gradient scope drift')
    return ValidationReport(passed=not errors,errors=errors,warnings=[])


def validate_gradient_card(card):
    from .governance import ValidationReport
    errors=[]
    try:
        s=card['scope'];a=card['approval']
        assert card['schema_version']=='gse_membership_gradient_card_v1'
        assert validate_card(s['training_card']).passed
        assert s['optimizer_steps']==0 and s['checkpoint_role']=='final_only'
        if 'observation_indices' in s:
            assert s['observation_indices']==[11,12,13,14,15] and s['selected_windows']==5
        assert a['status']=='APPROVED' and a['authorized_operations']==['audit'] and a['authorized_gates']==[3]
        assert a['scope_sha256']==hashlib.sha256(json.dumps(s,sort_keys=True).encode()).hexdigest()
    except (KeyError,TypeError,AssertionError):errors.append('fixed-checkpoint no-update gradient contract drift')
    return ValidationReport(passed=not errors,errors=errors,warnings=[])


def validate_common_observation_card(card):
    from .governance import ValidationReport
    errors=[]
    try:
        s=card['scope'];a=card['approval'];b=s['binding']
        assert card['schema_version']=='gse_common_observation_card_v1'
        assert a['status']=='APPROVED' and a['authorized_operations']==['data_export'] and a['authorized_gates']==[3]
        assert a['scope_sha256']==hashlib.sha256(json.dumps(s,sort_keys=True).encode()).hexdigest()
        assert b['selected_windows']==16 and b['selected_unique_frames']==80 and b['decoded_observations_per_uncached_pass']==105
        assert len(b['entries'])==16 and len({e['source']['parent_id'] for e in b['entries']})==8
        assert all(e['source']['split']=='fit' for e in b['entries'])
        assert s['training_steps']==0 and s['teacher_payload_reads']==0
        assert s['checkpoint']['sha256']=='8d5d2e0ec9779c28b068d0c7db80aeed38ebdb22dff5b781c26234d9001287fb'
        assert s['limits']==dict(wall_seconds=720,host_bytes=4*1024**3,gpu_bytes=28*1024**3,output_bytes=512*1024**2)
    except (KeyError,TypeError,AssertionError):errors.append('fixed16 common observation export drift')
    return ValidationReport(passed=not errors,errors=errors,warnings=[])


def validate_common_fit_card(card):
    from .governance import ValidationReport
    errors=[]
    try:
        s=card['scope'];a=card['approval'];t=s['task']
        assert card['schema_version']=='gse_common_structure_fit_card_v1'
        assert a['status']=='APPROVED' and a['authorized_operations']==['training'] and a['authorized_gates']==[3]
        assert a['scope_sha256']==hashlib.sha256(json.dumps(s,sort_keys=True).encode()).hexdigest()
        assert t['observations']==16 and t['independent_parents']==8 and t['split']=='fit_only'
        assert t['counts']==dict(anchors=24,window_sections=24,positive_relations=22,negative_relations=13,unknown_relations=5,complete_background_windows=0,dimension_targets=0)
        assert s['updates']==500 and s['accumulation']==4 and s['seed']==0 and s['variant']=='C'
        assert s['learning_rate']==.001 and s['weight_decay']==.0001 and s['partial_reference_only'] is True
        assert s['limits']==dict(wall_seconds=7200,host_bytes=32*1024**3,gpu_bytes=28*1024**3,output_bytes=4*1024**3)
    except (KeyError,TypeError,AssertionError):errors.append('common structural partial-fit contract drift')
    return ValidationReport(passed=not errors,errors=errors,warnings=[])
