from copy import deepcopy
import pytest
from mtare_topo.data.gse_v8_probe_comparison import compare


def pair():
    a=dict(source_binding={'source':'fixed'},record=dict(anchors=[{'position_m':[0,0,0],'evidence':'old'}],
        openings=[{'position_m':[1,0,0]}],membership=[[True]],score_region={'complete':False}))
    return a,deepcopy(a)


def test_prose_change_recorded_not_mistaken_for_geometry_change():
    a,b=pair();b['record']['anchors'][0]['evidence']='new observed lateral evidence'
    result=compare(a,b)
    assert len(result['evidence_text_changes'])==1
    assert result['membership_changes']==[]


@pytest.mark.parametrize('defect',['position','known','duplicate','region','new_field','dimensions','integer'])
def test_geometry_and_semantic_guards_retained(defect):
    a,b=pair()
    if defect=='position':b['record']['anchors'][0]['position_m'][2]=1e-14
    elif defect=='known':b['record']['membership']=[[None]]
    elif defect=='duplicate':
        b['record']['anchors'].append(deepcopy(b['record']['anchors'][0]));b['record']['membership']=[[True,None]]
    elif defect=='region':b['record']['score_region']['complete']=True
    elif defect=='new_field':b['record']['anchors'][0]['width_m']=2
    elif defect=='integer':b['record']['membership']=[[1]]
    else:b['record']['membership']=[[]]
    with pytest.raises(ValueError):compare(a,b)
