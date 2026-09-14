from copy import deepcopy
import hashlib
import json
import numpy as np
import pytest
from mtare_topo.integration.streaming_structural_tokens import StreamingStructuralTokens
from tests.v3.unit.test_frozen_structural_token_worker import fixture,FakeOutput


def request(root,index=0,*,bad_pose=False,sequence='live-synthetic'):
    entry,raw,poses=fixture()
    if bad_pose: poses[0,0,0]=2.
    entry['sequence_id']=sequence;entry['window_id']=f'window-{index}'
    for frame in entry['frames']:
        frame['frame_order']+=index*5;frame['stamp_ns']+=index*1000000000
        frame['source_key']+=f'-request-{index}'
    data=root/f'input-{index}.npz'
    np.savez_compressed(data,raw_pointcloud_bytes=raw,world_from_sensor=poses)
    entry['input']=dict(path=data.name,sha256=hashlib.sha256(data.read_bytes()).hexdigest())
    manifest=root/f'manifest-{index}.json'
    manifest.write_text(json.dumps(dict(schema_version='frozen_structural_token_windows_v1',windows=[entry])))
    return dict(schema_version='streaming_structural_token_request_v1',request_id=f'request-{index}',
        manifest=dict(path=manifest.name,sha256=hashlib.sha256(manifest.read_bytes()).hexdigest()))


def producer(root,*,cap=3):
    calls=[]
    class Extractor:
        def extract(self,student,**kw):
            calls.append(('forward',kw))
            return FakeOutput(np.arange(2),np.ones((2,128),dtype=np.float32))
    def load(*a,**kw): calls.append(('load',kw));return Extractor()
    ticks=iter(range(100,1000,10))
    return StreamingStructuralTokens(root,root/'tokens',device='cpu',max_requests=cap,
        loader=load,clock=lambda:next(ticks)),calls


def test_resident_model_two_distinct_windows_source_time_and_output_preserved(tmp_path):
    p,calls=producer(tmp_path);p.preload()
    a=p.process(request(tmp_path,0));b=p.process(request(tmp_path,1))
    assert [c[0] for c in calls]==['load','forward','forward']
    assert a['status']==b['status']=='PROCESSED'
    assert a['window_id']!=b['window_id']
    assert a['server_completed_monotonic_ns']<b['server_received_monotonic_ns']
    assert not a['available_at_native_snapshot'] and not a['control_published']
    assert a['compute_and_evidence_wall_ns']==10
    for response in (a,b):
        token=response['token_ref']
        from pathlib import Path
        assert hashlib.sha256(Path(token['path']).read_bytes()).hexdigest()==token['sha256']
        assert response['training_steps']==0


def test_rejected_pose_never_loads_model_and_next_valid_window_still_works(tmp_path):
    p,calls=producer(tmp_path)
    result=p.process(request(tmp_path,0,bad_pose=True))
    assert result['status']=='REJECTED_RECORD' and not calls
    assert result['token_ref'] is None
    assert p.process(request(tmp_path,1))['status']=='PROCESSED'


def test_duplicate_request_cannot_repeat_forward(tmp_path):
    p,calls=producer(tmp_path);req=request(tmp_path)
    p.process(req)
    with pytest.raises(ValueError,match='DUPLICATE_REQUEST'):p.process(req)
    assert sum(c[0]=='forward' for c in calls)==1


@pytest.mark.parametrize('fault',['segment','frame','capacity','manifest_hash'])
def test_stream_boundary_failures_never_invoke_second_forward(tmp_path,fault):
    p,calls=producer(tmp_path,cap=1 if fault=='capacity' else 3)
    p.process(request(tmp_path,1))
    req=request(tmp_path,0 if fault=='frame' else 2,sequence='other' if fault=='segment' else 'live-synthetic')
    if fault=='manifest_hash':req['manifest']['sha256']='0'*64
    with pytest.raises(ValueError):p.process(req)
    assert sum(c[0]=='forward' for c in calls)==1


def test_source_file_drift_stops_resident_stream_and_preserves_failed_record(tmp_path):
    p,calls=producer(tmp_path);req=request(tmp_path)
    (tmp_path/'input-0.npz').write_bytes(b'changed')
    result=p.process(req)
    assert result['status']=='FATAL_SOURCE_OR_EXECUTION_ERROR' and p.failed and not calls
    with pytest.raises(ValueError,match='STREAM_FATAL'):p.process(request(tmp_path,1))


def test_input_scope_cannot_read_another_repository_population(tmp_path):
    allowed=tmp_path/'only-this-live-input';allowed.mkdir()
    p=StreamingStructuralTokens(tmp_path,tmp_path/'out',device='cpu',max_requests=1,input_root=allowed,
        loader=lambda *a,**kw:pytest.fail('must not load model'))
    with pytest.raises(ValueError,match='MANIFEST_OUTSIDE_INPUT_SCOPE'):
        p.process(request(tmp_path))
