"""Exact, non-blind AI review scope; never authorizes training."""
import hashlib
import json
from mtare_topo.governance import ValidationReport

def validate_card(card):
    errors=[]
    try:
        s=card['scope'];a=card['approval']
        assert card['schema_version']=='gse_ai_branch_review_v1'
        assert (s['observations'],s['unique_frames'],s['parents'])==(12,24,3)
        assert s['training_authorized'] is False and s['human_gold_count']==0
        assert s['blind'] is False and s['prior_teacher_exposure'] is True
        assert len(s['entries'])==12
        assert {e['task'] for e in s['entries']}=={'S08_3d_loop_rich_C01__ellipse','S04_3d_unicyclic_small_C06__ellipse','S08_3d_loop_rich_C06__ellipse'}
        assert a['status']=='APPROVED' and a['authorized_operations']==['ai_annotation'] and a['authorized_gates']==[3]
        assert a['scope_sha256']==hashlib.sha256(json.dumps(s,sort_keys=True).encode()).hexdigest()
    except (AssertionError,KeyError,TypeError):errors.append('branch review exact scope mismatch')
    return ValidationReport(not errors,tuple(errors))
