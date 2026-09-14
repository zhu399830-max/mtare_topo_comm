"""Exact fixed-cache scope for the explicitly revised supervision task."""
import hashlib
import json


def validate_card(card):
    from .governance import ValidationReport
    errors = []
    try:
        s = card['scope']; a = card['approval']; entries = s['entries']
        assert card['schema_version'] == 'gse_conditional_geometry_card_v1'
        assert (len(entries), s['parents'], s['frames'], s['roi_returns'], s['patches']) == (12, 12, 60, 446184, 10193)
        assert len({e['parent'] for e in entries}) == 12 and [e['case'] for e in entries] == list(range(12))
        assert all(e['split'] == 'fit' and any('_C%02d__' % k in e['task'] for k in range(1, 7)) for e in entries)
        assert s['target_schema'] == 'construction_conditioned_geometry_targets_v1'
        assert s['observability_certified'] is False and s['connectivity_certified'] is False
        assert s['target_fields'] == ['axis_abs_dot', 'reference_center_height_difference_m']
        assert s['training_steps'] == 0 and s['raw_sensor_reads'] == 0
        assert s['limits'] == dict(wall_seconds=300, host_bytes=4*1024**3, output_bytes=128*1024**2)
        assert a['status'] == 'APPROVED' and a['authorized_operations'] == ['data_export'] and a['authorized_gates'] == [3]
        assert a['scope_sha256'] == hashlib.sha256(json.dumps(s, sort_keys=True).encode()).hexdigest()
    except (KeyError, TypeError, AssertionError): errors.append('conditional geometry fixed-cache scope drift')
    return ValidationReport(passed=not errors, errors=errors, warnings=[])


def validate_fit_card(card):
    from .governance import ValidationReport
    errors=[]
    try:
        assert card['schema_version']=='gse_conditional_fit_card_v1'
        s=card['scope'];a=card['approval'];p=s['population']
        assert (p['parents'],p['frames'],p['patches'],p['roi_returns'])==(12,60,10193,446184)
        assert len(p['entries'])==12 and len({e['parent'] for e in p['entries']})==12
        assert all(e['split']=='fit' and any('_C%02d__'%k in e['task'] for k in range(1,7)) for e in p['entries'])
        assert p['target_schema']=='construction_conditioned_geometry_targets_v1' and not p['observability_certified']
        assert s['variants']==list('ABC') and (s['seed'],s['updates'],s['microbatch'],s['accumulation'])==(0,2000,1,4)
        assert (s['lr'],s['weight_decay'])==(.001,.0001) and s['fit_only'] is True and s['encoder_frozen'] is True
        assert s['evaluation_steps']==[0,2000]
        assert s['loss']=='per observation: unordered component pair means, same/cross category equal means, axis L1 + height L1/10m; four observations averaged'
        assert s['limits']==dict(wall_seconds=43200,host_bytes=32*1024**3,gpu_bytes=28*1024**3,output_bytes=30*1024**3)
        assert a['status']=='APPROVED' and a['authorized_operations']==['training'] and a['authorized_gates']==[3]
        assert a['scope_sha256']==hashlib.sha256(json.dumps(s,sort_keys=True).encode()).hexdigest()
    except (KeyError,TypeError,AssertionError):errors.append('conditional fit fixed scope drift')
    return ValidationReport(passed=not errors,errors=errors,warnings=[])


def validate_correction_card(card):
    from .governance import ValidationReport
    import copy
    errors=[]
    try:
        assert card['schema_version']=='gse_conditional_axis_correction_card_v1'
        s=card['scope'];a=card['approval']
        assert s['variants']==['C'] and s['axis_objective']=='conditional_axis_soft_target_logit_v1'
        base='results/gate3_semantics/gate3_20260911_gse_new12_conditional_fit_v1_seed0/artifacts/'
        assert s['original_initial']==base+'initial.pt' and s['original_schedule']==base+'schedule.npy'
        assert s['loss']=='per observation: unordered component pair means, same/cross category equal means, axis soft-target logit loss + height L1/10m; four observations averaged'
        assert a['scope_sha256']==hashlib.sha256(json.dumps(s,sort_keys=True).encode()).hexdigest()
        normalized=copy.deepcopy(card);normalized['schema_version']='gse_conditional_fit_card_v1'
        normalized['scope']['variants']=list('ABC')
        normalized['scope']['loss']='per observation: unordered component pair means, same/cross category equal means, axis L1 + height L1/10m; four observations averaged'
        normalized['approval']['scope_sha256']=hashlib.sha256(json.dumps(normalized['scope'],sort_keys=True).encode()).hexdigest()
        assert validate_fit_card(normalized).passed
    except (KeyError,TypeError,AssertionError):errors.append('fixed conditional C correction scope drift')
    return ValidationReport(passed=not errors,errors=errors,warnings=[])


def validate_corrected_ab_card(card):
    from .governance import ValidationReport
    import copy
    errors=[]
    try:
        assert card['schema_version']=='gse_conditional_corrected_ab_card_v1'
        s=card['scope'];a=card['approval']
        assert s['variants']==['A','B']
        assert s['reused_corrected_c']=='results/gate3_semantics/gate3_20260911_gse_new12_axis_logit_correction_v1_seed0'
        assert a['scope_sha256']==hashlib.sha256(json.dumps(s,sort_keys=True).encode()).hexdigest()
        normalized=copy.deepcopy(card);normalized['schema_version']='gse_conditional_axis_correction_card_v1';normalized['scope']['variants']=['C']
        normalized['approval']['scope_sha256']=hashlib.sha256(json.dumps(normalized['scope'],sort_keys=True).encode()).hexdigest()
        assert validate_correction_card(normalized).passed
    except (KeyError,TypeError,AssertionError):errors.append('same-protocol corrected A/B scope drift')
    return ValidationReport(passed=not errors,errors=errors,warnings=[])
