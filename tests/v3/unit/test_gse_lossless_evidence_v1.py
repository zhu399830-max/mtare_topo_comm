import gzip
import json
import pytest
from mtare_topo.data.gse_lossless_evidence_v1 import encode_evidence,write_evidence


def test_exact_original_json_bytes_and_determinism(tmp_path):
    value={'unknown':None,'evidence':[{'ray_index':2**53+1,'t':0.12345678901234567,'z':-0.0}],
           '中文':['保留',True,False]}
    packed,meta=encode_evidence(value)
    assert (packed,meta)==encode_evidence(value)
    assert gzip.decompress(packed)==(json.dumps(value,ensure_ascii=False,sort_keys=True,allow_nan=False)+'\n').encode()
    assert write_evidence(tmp_path/'e.json.gz',value)==meta
    assert (tmp_path/'e.json.gz').read_bytes()==packed
    with pytest.raises(FileExistsError):write_evidence(tmp_path/'e.json.gz',{'different':True})


def test_invalid_nonfinite_not_persisted(tmp_path):
    with pytest.raises(ValueError):write_evidence(tmp_path/'bad.json.gz',{'bad':float('nan')})
    assert not (tmp_path/'bad.json.gz').exists()
