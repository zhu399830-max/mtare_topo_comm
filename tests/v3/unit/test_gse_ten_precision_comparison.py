import copy
import importlib.util
from pathlib import Path
import sys
import pytest


def test_explicit_added_withdrawn_and_changed_memberships():
    root=Path(__file__).resolve().parents[3];sys.path.insert(0,str(root/'tools/v3'))
    import v8_original_ten_probe as base
    snapshot=dict(vars(base))
    try:
        spec=importlib.util.spec_from_file_location('ten_precision_test',root/'tools/v3/original_ten_precision.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        a=dict(source_binding={},record=dict(anchors=[dict(position_m=[0,0,0]),dict(position_m=[1,0,0])],
            openings=[{}],membership=[[True,None]],score_region={},source_frame_indices=[1],coordinate_frame='test'))
        b=copy.deepcopy(a);b['record']['anchors']=[dict(position_m=[1,0,0]),dict(position_m=[2,0,0])]
        b['record']['membership']=[[False,True]]
        report=module.compare(a,b)
        assert [x['status'] for x in report['changes']]==['WITHDRAWN','RETAINED','ADDED']
        assert report['changes'][1]['old_memberships']==[None]
        assert report['changes'][1]['new_memberships']==[False]
        assert not report['complete_annotation']
        b['record']['score_region']={'changed':True}
        with pytest.raises(ValueError,match='score_region'):module.compare(a,b)
    finally:
        vars(base).clear();vars(base).update(snapshot)
