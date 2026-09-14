import hashlib,json
from mtare_topo.governance import ValidationReport
from mtare_topo.governance_short_chain import validate_card as validate_source

def validate_card(card):
    errors=[]
    try:
        s=card['scope'];a=card['approval']
        assert card['schema_version']=='gse_structural_token_chain_v1'
        assert s['observations']==8 and s['training_steps']==0 and s['variants']==list('ABC')
        assert s['source_scope']['frame_rows']==list(range(120,132))
        assert s['source_scope']['identity']['task']=='S03_flat_unicyclic_small_C07__ellipse'
        assert s['source_run']=='results/gate3_semantics/gate3_20260912_gse_short_observation_chain_v1_seed0'
        assert all(len(s['models'][m]['sha256'])==64 and s['models'][m]['path'].endswith(m+'_final.pt') for m in 'ABC')
        assert s['models']['updates_per_variant']==2000 and s['models']['seed']==0
        assert a['status']=='APPROVED' and a['authorized_operations']==['data_export'] and a['authorized_gates']==[3]
        assert a['scope_sha256']==hashlib.sha256(json.dumps(s,sort_keys=True).encode()).hexdigest()
    except (AssertionError,KeyError,TypeError):errors.append('frozen token integration scope mismatch')
    return ValidationReport(not errors,tuple(errors))
