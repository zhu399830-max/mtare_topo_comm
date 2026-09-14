import hashlib,json
from mtare_topo.governance import ValidationReport
def validate_card(card):
    errors=[]
    try:
        s=card['scope'];a=card['approval']
        assert card['schema_version']=='gse_local_conflict_replay_v1'
        assert (s['observations'],s['parents'],s['rays'],s['pairs'])==(12,3,135699,1815651)
        assert s['training_steps']==s['model_forwards']==0
        assert s['thresholds']==[.1,.9] and s['unknown_filtered'] is False
        assert s['partial_ai_reference_only'] is True
        assert a['authorized_operations']==['audit'] and a['authorized_gates']==[3] and a['status']=='APPROVED'
        assert a['scope_sha256']==hashlib.sha256(json.dumps(s,sort_keys=True).encode()).hexdigest()
    except (KeyError,AssertionError,TypeError):errors.append('local conflict replay exact scope mismatch')
    return ValidationReport(not errors,tuple(errors))
