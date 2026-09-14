import hashlib,json
from mtare_topo.governance import ValidationReport
def validate_card(card):
    errors=[]
    try:
        s=card['scope'];a=card['approval']
        assert card['schema_version']=='gse_baseline_restart_v1'
        assert s['world']=='tunnel' and s['robots']==s['episodes']==1 and s['runtime_sec']==600 and s['seed']==11
        assert s['method']=='original_mtare' and s['training_steps']==0 and s['test_claim'] is False and s['teacher_map_to_planner'] is False
        assert a['status']=='APPROVED' and a['authorized_operations']==['closed_loop_single'] and a['authorized_gates']==[6]
        assert a['scope_sha256']==hashlib.sha256(json.dumps(s,sort_keys=True).encode()).hexdigest()
    except (AssertionError,KeyError,TypeError):errors.append('baseline restart scope mismatch')
    return ValidationReport(not errors,tuple(errors))
