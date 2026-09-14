import copy
import importlib.util
import sys
from pathlib import Path
from mtare_topo.data.gse_s07_precision_scope import compile_scope
from mtare_topo.governance_s07_precision import SCHEMA,SLUG,POLICY,validate_card
from mtare_topo.governance_surface_selection import digest


def test_exact_original_case_metadata_and_card_mutation():
    root=Path(__file__).resolve().parents[3]
    scope=compile_scope(root)
    assert scope['counts']['observations']==1
    assert scope['counts']['container_observations']==16
    assert len(scope['file_sha256'])==13 and len(scope['metadata_reads_sha256'])==2
    card=dict(schema_version=SCHEMA,card_id=SLUG,operation='data_export',scope=scope,scope_sha256=digest(scope),policy=POLICY,
        approval=dict(status='APPROVED',scope_sha256=digest(scope),authorized_operations=['data_export'],authorized_gates=[3],confirmation_reference='test-only'))
    assert validate_card(card).passed
    bad=copy.deepcopy(card);bad['scope']['entries'][0]['source']['frame_rows'][0]+=1
    assert not validate_card(bad).passed
    bad=copy.deepcopy(card);bad['policy']['optimizer_steps']=1
    assert not validate_card(bad).passed


def test_comparison_reports_withdrawal_without_fabricating_match():
    root=Path(__file__).resolve().parents[3];sys.path.insert(0,str(root/'tools/v3'))
    spec=importlib.util.spec_from_file_location('single_s07_runner',root/'tools/v3/s07_precision_corrective.py')
    module=importlib.util.module_from_spec(spec)
    # Runner adapters modify module globals in their process; isolate this
    # import from other unit tests by restoring the reused module dictionary.
    import v8_original_ten_probe as base
    snapshot=dict(vars(base))
    try:
        spec.loader.exec_module(module)
        old=dict(source_binding={'x':1},record=dict(anchors=[{'position_m':[0,0,0]},{'position_m':[1,0,0]}],
            openings=[],score_region={},source_frame_indices=[1],coordinate_frame='test',membership=[]))
        new=copy.deepcopy(old);new['record']['anchors'].pop(0)
        result=module.compare(old,new)
        assert result['expected_withdrawal_observed']
        assert [x['status'] for x in result['changes']]==['WITHDRAWN','RETAINED']
        assert not result['complete_annotation']
    finally:
        vars(base).clear();vars(base).update(snapshot)
