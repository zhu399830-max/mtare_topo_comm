from copy import deepcopy
import pytest
from mtare_topo.data.gse_joint_cached_interfaces_v1 import compare_prior_targets


def fixture():
    return dict(source_binding={'source':'frozen'},target_record_sha256='old',record=dict(
        anchors=[{'position_m':[0,0,0]}],openings=[{'position_m':[1,0,0]},{'position_m':[2,0,0]}],
        membership=[[True],[None]],source_frame_indices=[0,1,2,3,4]))


def test_unchanged_or_nonowning_removal_records_exact_correspondence():
    old=fixture();new=deepcopy(old)
    assert compare_prior_targets(old,new)['removed_from_positive_only']==[]
    new['record']['openings'].pop(0);new['record']['membership'].pop(0)
    r=compare_prior_targets(old,new)
    assert r['retained_old_indices']==[1] and r['removed_from_positive_only']==[0]


@pytest.mark.parametrize('change',['position','membership','anchor','source'])
def test_unrelated_target_changes_fail(change):
    old=fixture();new=deepcopy(old)
    if change=='position':new['record']['openings'][0]['position_m'][0]=3
    if change=='membership':new['record']['membership'][0]=[None]
    if change=='anchor':new['record']['anchors']=[]
    if change=='source':new['source_binding']={}
    with pytest.raises(ValueError):compare_prior_targets(old,new)
