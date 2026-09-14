"""Exact frozen-encoder new12 export scope, no teacher payload or updates."""
import hashlib,json


def validate_card(card):
    from .governance import ValidationReport
    errors=[]
    try:
        s=card['scope'];a=card['approval'];es=s['entries']
        assert card['schema_version']=='gse_new12_features_card_v1'
        assert (len(es),s['parents'],s['frames'],s['valid_returns'])==(12,12,60,684440)
        assert len({e['parent'] for e in es})==12 and [e['case'] for e in es]==list(range(12))
        assert all(e['split']=='fit' and any('_C%02d__'%k in e['task'] for k in range(1,7)) for e in es)
        assert s['checkpoint']['sha256']=='8d5d2e0ec9779c28b068d0c7db80aeed38ebdb22dff5b781c26234d9001287fb'
        assert s['checkpoint']['epoch']==2 and s['checkpoint']['seed']==0
        assert s['training_steps']==0 and s['teacher_payload_reads']==0
        assert s['limits']==dict(wall_seconds=720,host_bytes=4*1024**3,gpu_bytes=28*1024**3,output_bytes=512*1024**2)
        assert a['status']=='APPROVED' and a['authorized_operations']==['data_export'] and a['authorized_gates']==[3]
        assert a['scope_sha256']==hashlib.sha256(json.dumps(s,sort_keys=True).encode()).hexdigest()
    except (KeyError,TypeError,AssertionError):errors.append('new12 frozen feature scope drift')
    return ValidationReport(passed=not errors,errors=errors,warnings=[])
