import gzip,json,hashlib
from copy import deepcopy
import pytest
from mtare_topo.data.gse_joint_v4_cache_v1 import read_v4,compare_versions


def test_direct_sealed_evidence_source_checked(tmp_path):
    source={'task':'synthetic','frame_rows':[0,1,2,3,4]}
    data={'raw_interfaces':{'source':source},'produced_targets':{'record':{}}}
    raw=gzip.compress(json.dumps(data).encode());p=tmp_path/'input.gz';p.write_bytes(raw)
    pin=dict(path='input.gz',sha256=hashlib.sha256(raw).hexdigest());opened={}
    result,old=read_v4(tmp_path,pin,source,opened)
    assert result==data['raw_interfaces'] and opened['input.gz']==pin['sha256']
    with pytest.raises(ValueError,match='source'):
        read_v4(tmp_path,pin,{'task':'other'},opened)


def test_compare_preserves_geometry_and_records_relation_changes():
    old=dict(source_binding={'source':'same'},target_record_sha256='old',record=dict(
        anchors=[{'position_m':[0,0,0]}],openings=[{'position_m':[10,0,0]}],membership=[[None]]))
    new=deepcopy(old);new['target_record_sha256']='new';new['record']['membership']=[[True]]
    r=compare_versions(old,new)
    assert r['membership_changes']==[dict(old_opening_index=0,new_opening_index=0,anchor_index=0,before=None,after=True)]
    assert not r['full_label_qualification']
    new['record']['anchors'][0]['position_m'][0]=1
    with pytest.raises(ValueError,match='non-opening'):compare_versions(old,new)
