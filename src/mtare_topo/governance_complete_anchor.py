import hashlib,json
from mtare_topo.governance import ValidationReport

def validate_card(card):
    errors=[]
    try:
        s=card['scope'];a=card['approval']
        assert card['schema_version']=='gse_complete_anchor_queries_v1'
        assert (s['observations'],s['parents'],s['rays'],s['complete_pairs'])==(12,3,135699,4336032)
        assert s['training_steps']==0 and s['maximum_anchors']==32 and s['thresholds']==[.1,.9]
        assert s['teacher_in_inference'] is False
        assert a['status']=='APPROVED' and a['authorized_operations']==['audit'] and a['authorized_gates']==[3]
        assert a['scope_sha256']==hashlib.sha256(json.dumps(s,sort_keys=True).encode()).hexdigest()
    except (AssertionError,KeyError,TypeError):errors.append('complete anchor diagnostic scope mismatch')
    return ValidationReport(not errors,tuple(errors))
