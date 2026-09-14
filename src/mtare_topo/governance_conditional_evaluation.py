"""Frozen final-checkpoint, full-population development evaluation scope."""
import hashlib
import json


def validate_card(card):
    from .governance import ValidationReport
    errors=[]
    try:
        s,a=card['scope'],card['approval']
        assert card['schema_version']=='gse_conditional_development_evaluation_card_v1'
        assert len(s['entries'])==240 and s['training_steps']==0 and s['variants']==list('ABC')
        assert [e['identity']['case'] for e in s['entries']]==list(range(240))
        expected={f'S{x}_C07' for x in ('03_flat_unicyclic_small','04_3d_unicyclic_small','05_flat_branch_medium','06_3d_branch_medium','10_3d_complex')}
        assert {e['identity']['parent_id'] for e in s['entries']}==expected
        assert all(e['identity']['split']=='development' for e in s['entries'])
        assert s['checkpoint_step']==2000 and s['split_audit']['strict_unseen'] is False
        assert s['target_schema']=='construction_conditioned_geometry_targets_v1'
        expected_hashes=dict(A='247132276bf54ec1f2be79fea2966c9bc481ecbf2de02d6b906350e6eb37378d',B='e2b6691134793a337e571cfa6d19693425b0e71717120ce9e0bf5ec12260210b',C='1000e0ae0ae559915965e9e1cefbe8f08adf9c4aad16a89340fdee60328d995d')
        assert {v:s['models'][v]['sha256'] for v in 'ABC'}==expected_hashes
        assert s['limits']==dict(wall_seconds=1800,host_bytes=8*1024**3,gpu_bytes=28*1024**3,output_bytes=2*1024**3)
        assert a['status']=='APPROVED' and a['authorized_operations']==['data_export'] and a['authorized_gates']==[3]
        assert a['scope_sha256']==hashlib.sha256(json.dumps(s,sort_keys=True).encode()).hexdigest()
    except (KeyError,TypeError,AssertionError):errors.append('fixed conditional development evaluation scope drift')
    return ValidationReport(passed=not errors,errors=errors,warnings=[])
