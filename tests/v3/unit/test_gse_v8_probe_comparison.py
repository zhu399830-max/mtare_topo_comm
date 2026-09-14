from copy import deepcopy
import importlib.util
from pathlib import Path
import sys

import pytest

TOOLS = Path(__file__).resolve().parents[3]/'tools/v3'
sys.path.insert(0,str(TOOLS))
spec = importlib.util.spec_from_file_location('v8_probe_test_runner', TOOLS/'v8_original_ten_probe.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def target():
    return dict(source_binding={'same':True},record=dict(anchors=[{'position_m':[0,0,0]}],
        openings=[{'position_m':[1,0,0]}],score_region={'complete':False},
        source_frame_indices=[0,1,2,3,4],membership=[[None]]))


def test_negative_from_unknown_recorded_without_promotion():
    old=target();new=deepcopy(old);new['record']['membership']=[[False]]
    out=runner.compare(old,new)
    assert out['new_counts']==dict(positive=0,negative=1,unknown=0)
    assert out['membership_changes'][0]['before'] is None
    assert not out['complete_annotation']


@pytest.mark.parametrize('defect',['known','anchor','opening','source','region'])
def test_existing_evidence_cannot_be_silently_changed(defect):
    old=target();old['record']['membership']=[[True]];new=deepcopy(old)
    if defect=='known':new['record']['membership']=[[False]]
    elif defect=='anchor':new['record']['anchors'][0]['position_m']=[0,0,1]
    elif defect=='opening':new['record']['openings']=[]
    elif defect=='source':new['source_binding']={'same':False}
    else:new['record']['score_region']={'complete':True}
    with pytest.raises(ValueError):runner.compare(old,new)
