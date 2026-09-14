"""Fixed new12 cached operand-target export scope."""
import hashlib,json


def validate_card(card):
    from .governance import ValidationReport
    errors=[]
    try:
        s=card['scope'];a=card['approval'];es=s['entries']
        assert card['schema_version']=='gse_new12_affinity_targets_card_v1'
        assert (len(es),s['parents'],s['frames'],s['roi_returns'],s['patches'])==(12,12,60,446184,10193)
        assert len({e['parent'] for e in es})==12 and [e['case'] for e in es]==list(range(12))
        assert all(e['split']=='fit' and any('_C%02d__'%k in e['task'] for k in range(1,7)) for e in es)
        assert s['training_steps']==0 and s['raw_sensor_reads']==0 and s['structural_membership'] is False
        assert s['source_rule']=='pinned nonempty face_sources subset active; singleton only'
        assert s['patch_rule']=='all original returns qualified singleton and same operand; otherwise unknown'
        assert s['limits']==dict(wall_seconds=300,host_bytes=4*1024**3,output_bytes=64*1024**2)
        assert a['status']=='APPROVED' and a['authorized_operations']==['data_export'] and a['authorized_gates']==[3]
        assert a['scope_sha256']==hashlib.sha256(json.dumps(s,sort_keys=True).encode()).hexdigest()
    except (KeyError,TypeError,AssertionError):errors.append('new12 cached target scope drift')
    return ValidationReport(passed=not errors,errors=errors,warnings=[])
