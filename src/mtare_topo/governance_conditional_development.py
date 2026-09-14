"""Exact read-only-development input transport, not training authorization."""
import hashlib
import json


def validate_card(card):
    from .governance import ValidationReport
    errors = []
    try:
        assert card['schema_version'] == 'gse_conditional_development_inputs_card_v1'
        s, a = card['scope'], card['approval']
        assert s['parents'] == 5 and s['frames'] == 1200 and s['physical_edges'] == 80
        assert len(s['observations']) == 240 and len(s['task_prefixes']) == 15
        expected = {f'S{x}_C07' for x in ('03_flat_unicyclic_small','04_3d_unicyclic_small','05_flat_branch_medium','06_3d_branch_medium','10_3d_complex')}
        assert {e['parent'] for e in s['observations']} == expected
        assert all(e['split'] == 'development' and len(e['frame_rows']) == 5 and e['task'].startswith(e['parent']+'__') for e in s['observations'])
        assert all(v['partition'] == 'c07' for v in s['task_prefixes'].values())
        assert s['split_audit']['strict_unseen'] is False
        assert s['training_steps'] == 0 and s['labels_generated'] == 0
        assert s['limits'] == dict(wall_seconds=900, host_bytes=4*1024**3, output_bytes=1024**3)
        assert a['status'] == 'APPROVED' and a['authorized_operations'] == ['data_export'] and a['authorized_gates'] == [3]
        assert a['scope_sha256'] == hashlib.sha256(json.dumps(s, sort_keys=True).encode()).hexdigest()
    except (KeyError, TypeError, AssertionError):
        errors.append('fixed five-parent development input scope drift')
    return ValidationReport(passed=not errors, errors=errors, warnings=[])


def validate_feature_card(card):
    from .governance import ValidationReport
    errors=[]
    try:
        assert card['schema_version']=='gse_conditional_development_features_card_v1'
        s,a=card['scope'],card['approval']
        assert len(s['entries'])==240 and (s['parents'],s['frames'])==(5,1200)
        expected={f'S{x}_C07' for x in ('03_flat_unicyclic_small','04_3d_unicyclic_small','05_flat_branch_medium','06_3d_branch_medium','10_3d_complex')}
        assert {e['parent'] for e in s['entries']}==expected
        assert all(e['split']=='development' and e['task'].startswith(e['parent']+'__') for e in s['entries'])
        assert s['checkpoint']['sha256']=='8d5d2e0ec9779c28b068d0c7db80aeed38ebdb22dff5b781c26234d9001287fb'
        assert s['training_steps']==0 and s['teacher_payload_reads']==0
        assert s['valid_returns_per_observation'] is None
        assert s['limits']==dict(wall_seconds=900,host_bytes=4*1024**3,gpu_bytes=28*1024**3,output_bytes=4*1024**3)
        assert a['status']=='APPROVED' and a['authorized_operations']==['data_export'] and a['authorized_gates']==[3]
        assert a['scope_sha256']==hashlib.sha256(json.dumps(s,sort_keys=True).encode()).hexdigest()
    except (KeyError,TypeError,AssertionError):errors.append('frozen development feature scope drift')
    return ValidationReport(passed=not errors,errors=errors,warnings=[])


def validate_geometry_card(card):
    from .governance import ValidationReport
    errors=[]
    try:
        assert card['schema_version']=='gse_conditional_development_geometry_card_v1'
        s,a=card['scope'],card['approval']
        assert (len(s['entries']),s['parents'],s['frames'],s['patches'],s['roi_returns'])==(240,5,1200,202340,8498948)
        expected={f'S{x}_C07' for x in ('03_flat_unicyclic_small','04_3d_unicyclic_small','05_flat_branch_medium','06_3d_branch_medium','10_3d_complex')}
        identities=[e['identity'] for e in s['entries']]
        assert [e['case'] for e in identities]==list(range(240))
        assert {e['parent_id'] for e in identities}==expected
        assert all(e['split']=='development' and e['task'].startswith(e['parent_id']+'__') for e in identities)
        assert s['target_schema']=='construction_conditioned_geometry_targets_v1'
        assert s['observability_certified'] is False and s['connectivity_certified'] is False and s['training_steps']==0
        assert s['limits']==dict(workers=4,worker_address_bytes=4*1024**3,total_rss_bytes=32*1024**3,wall_seconds=21600,case_seconds=900,output_bytes=4*1024**3,max_candidates=200000,reference_capacity=256)
        assert s['original_surface_binding']['archive_sha256']=='4d159a53fe9755d1319298f7b5c4a237f7836db3c67e2e0c6993ea350c85ec27'
        assert s['split_audit']['strict_unseen'] is False
        assert a['status']=='APPROVED' and a['authorized_operations']==['data_export'] and a['authorized_gates']==[3]
        assert a['scope_sha256']==hashlib.sha256(json.dumps(s,sort_keys=True).encode()).hexdigest()
    except (KeyError,TypeError,AssertionError):errors.append('same-definition conditional development geometry scope drift')
    return ValidationReport(passed=not errors,errors=errors,warnings=[])
