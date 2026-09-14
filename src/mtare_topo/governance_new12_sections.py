import hashlib,json


def validate_interior_card(card):
    from .governance import ValidationReport
    errors=[]
    try:
        s=card['scope'];a=card['approval'];es=s['entries']
        assert card['schema_version']=='gse_new12_interior_support_card_v1'
        assert len(es)==12 and s['parents']==12 and s['frames']==60 and s['roi_returns']==446184
        assert [e['observation'] for e in es]==list(range(12)) and len({e['identity']['parent'] for e in es})==12
        assert all(e['identity']['split']=='fit' and any('_C%02d__'%k in e['identity']['task'] for k in range(1,7)) for e in es)
        assert s['labels_generated']==0 and s['training_steps']==0
        assert s['selection']=='middle unique observed arc slab per exact connected interval component; singleton only'
        assert s['limits']==dict(wall_seconds=900,host_bytes=4*1024**3,output_bytes=128*1024**2,candidates_per_observation=256)
        assert a['status']=='APPROVED' and a['authorized_operations']==['audit'] and a['authorized_gates']==[3]
        assert a['scope_sha256']==hashlib.sha256(json.dumps(s,sort_keys=True).encode()).hexdigest()
    except (KeyError,TypeError,AssertionError):errors.append('new12 interior scope drift')
    return ValidationReport(passed=not errors,errors=errors,warnings=[])


def validate_card(card):
    from .governance import ValidationReport
    errors=[]
    try:
        s=card['scope'];a=card['approval'];es=s['entries']
        assert card['schema_version']=='gse_new12_section_support_card_v1'
        assert len(es)==12 and s['parents']==12 and s['frames']==60
        assert [e['case'] for e in es]==list(range(12)) and len({e['parent'] for e in es})==12
        assert all(e['split']=='fit' and any('_C%02d__'%k in e['task'] for k in range(1,7)) for e in es)
        assert [r['case'] for r in s['references']]==list(range(12))
        assert [r['case'] for r in s['references'] if r['path'] is None]==[1,3,4,5,6,8]
        assert s['training_steps']==0 and s['new_labels']==0 and s['rerender'] is False
        assert s['limits']==dict(wall_seconds=120,host_bytes=4*1024**3,output_bytes=64*1024**2)
        assert a['status']=='APPROVED' and a['authorized_operations']==['audit'] and a['authorized_gates']==[3]
        assert a['scope_sha256']==hashlib.sha256(json.dumps(s,sort_keys=True).encode()).hexdigest()
    except (KeyError,TypeError,AssertionError):errors.append('new12 cached section scope drift')
    return ValidationReport(passed=not errors,errors=errors,warnings=[])
