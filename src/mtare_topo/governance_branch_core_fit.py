"""Explicit partial AI-core diagnostic, not full branch qualification."""
import hashlib,json
from mtare_topo.governance import ValidationReport
def validate_card(card):
    errors=[]
    try:
        s=card['scope'];a=card['approval']
        assert card['schema_version']=='gse_branch_core_fit_v1'
        assert (s['observations'],s['parents'],s['unique_frames'],s['updates'],s['effective_batch'])==(12,3,24,500,4)
        assert s['same_pairs']==98699 and s['different_pairs']==4202 and s['unknown_pairs']==1712750
        assert s['variant']=='C' and s['seed']==0 and s['full_branch_qualified'] is False
        assert s['independent_evaluation'] is False and s['unknown_is_negative'] is False
        assert len(s['entries'])==12
        assert {e['task'] for e in s['entries']}=={'S08_3d_loop_rich_C01__ellipse','S04_3d_unicyclic_small_C06__ellipse','S08_3d_loop_rich_C06__ellipse'}
        assert a['status']=='APPROVED' and a['authorized_operations']==['training'] and a['authorized_gates']==[3]
        assert a['scope_sha256']==hashlib.sha256(json.dumps(s,sort_keys=True).encode()).hexdigest()
    except (AssertionError,KeyError,TypeError):errors.append('partial branch core fit scope mismatch')
    return ValidationReport(not errors,tuple(errors))
