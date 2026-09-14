import hashlib,json
from mtare_topo.governance import ValidationReport
def validate_card(card):
    errors=[]
    try:
        s=card['scope'];a=card['approval']
        assert card['schema_version']=='gse_directional_context_readout_v1'
        assert s['source']=='results/gate3_semantics/gate3_20260912_gse_structural_token_chain_v1_seed0'
        assert s['raw']=='results/gate3_semantics/gate3_20260912_gse_short_observation_chain_v1_seed0'
        assert s['frames']==12 and s['windows']==8 and s['candidates_per_variant']==32 and s['variants']==list('ABC')
        assert s['model_forwards']==s['teacher_reads']==s['training_steps']==0
        assert a['status']=='APPROVED' and a['authorized_operations']==['data_export'] and a['authorized_gates']==[3]
        assert a['scope_sha256']==hashlib.sha256(json.dumps(s,sort_keys=True).encode()).hexdigest()
    except (AssertionError,KeyError,TypeError):errors.append('exact saved-context readout scope mismatch')
    return ValidationReport(not errors,tuple(errors))
