import sys
from pathlib import Path
import pytest
from test_multiview_teacher_population_v1 import setup_case
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'tools/v3'))
from v8_multiview_population_v1 import export_population,VERSION


def adapted(tmp_path):
    scope,reader,client=setup_case(tmp_path)
    original=client.request
    def request(bundle):
        response=original(bundle);p=response['produced_targets']
        p['producer_version']=VERSION
        p['geometry_evidence_settings']=dict(axial_spacing_m=.05,angular_segments=64,
            field_spacing_m=.025,intersections='original_float32_rays_complete_operand_surfaces',
            recast_first_returns=False,cap_precision_policy='source_float64_float32_interval_v1')
        return response
    client.request=request
    return scope,reader,client


def test_v8_export_preserves_zero_and_unknown_windows(tmp_path):
    scope,reader,client=adapted(tmp_path)
    result=export_population(tmp_path,scope,reader,client,lambda:None,output_cap_bytes=1024**2)
    assert len(result['observations'])==3 and result['summaries']['fit']['zero_anchor_windows']==2
    assert result['summaries']['fit']['unknown_memberships']==1
    assert not result['zero_anchor_is_negative']


@pytest.mark.parametrize('field,value',[('field_spacing_m',.01),('recast_first_returns',True)])
def test_policy_drift_fails_before_evidence_export(tmp_path,field,value):
    scope,reader,client=adapted(tmp_path);request=client.request
    def wrong(bundle):
        response=request(bundle)
        response['produced_targets']['geometry_evidence_settings'][field]=value
        return response
    client.request=wrong
    with pytest.raises(ValueError,match='geometry policy'):
        export_population(tmp_path,scope,reader,client,lambda:None,output_cap_bytes=1024**2)
    assert client.calls==1 and not list((tmp_path/'artifacts').iterdir())
