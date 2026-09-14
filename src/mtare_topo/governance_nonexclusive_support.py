import hashlib,json
from mtare_topo.governance import ValidationReport

def validate_card(card):
    errors=[]
    try:
        s=card['scope'];a=card['approval']
        assert card['schema_version']=='gse_nonexclusive_support_v1'
        assert (s['observations'],s['parents'],s['rays'],s['pairs'])==(12,3,135699,1815651)
        assert s['training_steps']==s['model_forwards']==0
        assert s['maximum_anchors']==32 and s['thresholds']==[.1,.9]
        assert s['teacher_in_inference'] is False and s['branch_identity_verified'] is False
        assert a['status']=='APPROVED' and a['authorized_operations']==['audit'] and a['authorized_gates']==[3]
        assert a['scope_sha256']==hashlib.sha256(json.dumps(s,sort_keys=True).encode()).hexdigest()
    except (KeyError,AssertionError,TypeError):errors.append('nonexclusive support bounded scope mismatch')
    return ValidationReport(not errors,tuple(errors))
