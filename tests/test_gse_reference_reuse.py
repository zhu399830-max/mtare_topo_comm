import hashlib,json
import pytest
from mtare_topo.data.gse_reference_reuse import copy_reference_prefix,FILES

def setup(tmp):
    old=tmp/'old';old.mkdir();new=tmp/'new';(new/'artifacts').mkdir(parents=True)
    pins={}
    for name in FILES:
        raw=json.dumps({'case':0}).encode() if name=='summary.json' else b'original'
        (old/name).write_bytes(raw);pins['old/'+name]=hashlib.sha256(raw).hexdigest()
    cases=[dict(case=0,eligible=True,artifacts=pins.copy())]
    return new,cases,pins

def test_copy_immutable_and_origin(tmp_path):
    new,cases,pins=setup(tmp_path);rows=copy_reference_prefix(tmp_path,new,cases,pins)
    assert rows[0]['reused'] and (new/'artifacts/case_000/reuse_origin.json').exists()
    for name in FILES:assert (new/'artifacts/case_000'/name).read_bytes()==(tmp_path/'old'/name).read_bytes()
    with pytest.raises(FileExistsError):copy_reference_prefix(tmp_path,new,cases,pins)

@pytest.mark.parametrize('bad',['hash','partial','prefix','ineligible'])
def test_reject(tmp_path,bad):
    new,cases,pins=setup(tmp_path)
    if bad=='hash':pins['old/targets.npz']='0'*64
    elif bad=='partial':del cases[0]['artifacts']['old/targets.npz']
    elif bad=='prefix':cases[0]['case']=1
    else:cases[0]['eligible']=False
    with pytest.raises(ValueError):copy_reference_prefix(tmp_path,new,cases,pins)
