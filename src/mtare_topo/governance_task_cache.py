import hashlib,json
from mtare_topo.governance import ValidationReport
from mtare_topo.governance_direction_task_pilot import validate_card as validate_source

def validate_card(card):
    errors=[]
    try:
        s=card['scope'];a=card['approval']
        assert card['schema_version']=='gse_direction_task_model_cache_v1'
        assert (s['observations'],s['unique_frames'],s['independent_fragments'])==(12,24,3)
        assert s['training_steps']==s['labels_generated']==s['teacher_payload_reads']==0
        assert s['variants']==list('ABC') and len(s['entries'])==12
        expected={(r['task'],tuple(r['frame_rows'])) for r in s['source_scope']['observations']}
        assert {(r['task'],tuple(r['frame_rows'])) for r in s['entries']}==expected and len(expected)==12
        assert len({(r['task'],f) for r in s['entries'] for f in r['frame_rows']})==24
        assert all(r['split']=='fit' for r in s['source_scope']['observations'])
        assert all(len(s['models'][m]['sha256'])==64 for m in 'ABC')
        assert a['status']=='APPROVED' and a['authorized_operations']==['data_export'] and a['authorized_gates']==[3]
        assert a['scope_sha256']==hashlib.sha256(json.dumps(s,sort_keys=True).encode()).hexdigest()
    except (AssertionError,KeyError,TypeError):errors.append('task cache scope mismatch')
    return ValidationReport(not errors,tuple(errors))
