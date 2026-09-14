import gzip
import json
import pytest
from mtare_topo.data.gse_structure_review_v1 import canonical_sha
from mtare_topo.data.multiview_teacher_population_v1 import export_population


def setup_case(tmp_path):
    (tmp_path/'logs').mkdir(); (tmp_path/'artifacts').mkdir()
    sources=[dict(task='synthetic_C01',source_sequence_id=i,frame_rows=list(range(i,i+5))) for i in range(3)]
    scope=dict(counts=dict(observations=3,tasks=1),entries=[dict(task='synthetic_C01',split='fit',observations=sources)])
    class Reader:
        def read_task(self,task):return [dict(source=s) for s in sources]
    class Client:
        calls=0
        def request(self,bundle):
            self.calls+=1;s=bundle['source'];positive=s['source_sequence_id']==1
            record=dict(coordinate_frame='current_sensor_m',source_frame_indices=s['frame_rows'],
                anchors=[dict(position_m=[2.,0.,0.])] if positive else [],openings=[{}],
                membership=[[None]] if positive else [[]])
            produced=dict(record=record,producer_version='joint_partial_reference_v6_reference_competition',
                target_record_sha256=canonical_sha(record),teacher_provenance=dict(terminal_anchor_start=int(positive),anchors=[{}] if positive else []),
                unknown_candidates={'anchors':[{'reason':'synthetic_unknown'}]},supervision_status='PARTIAL',full_training_gate_eligible=False)
            return dict(raw_interfaces=dict(source=s,raw_interface_intersections=[]),produced_targets=produced)
    return scope,Reader(),Client()


def test_every_window_unknowns_and_full_evidence_preserved(tmp_path):
    scope,reader,client=setup_case(tmp_path)
    result=export_population(tmp_path,scope,reader,client,lambda:None,output_cap_bytes=1024**2)
    assert client.calls==3 and len(result['observations'])==3
    assert result['summaries']['fit']['zero_anchor_windows']==2
    assert result['summaries']['fit']['unknown_memberships']==1
    assert result['summaries']['calibration']['junction_x_quantiles'] is None
    assert result['zero_anchor_is_negative'] is False
    saved=json.loads(gzip.decompress((tmp_path/'artifacts/observation_00001.json.gz').read_bytes()))
    assert saved['produced_targets']['record']['membership']==[[None]]
    assert saved['produced_targets']['unknown_candidates']['anchors']
    assert 'raw_interfaces' in saved


def test_development_scope_rejected_before_read_or_call(tmp_path):
    scope,reader,client=setup_case(tmp_path);scope['entries'][0]['split']='development'
    with pytest.raises(ValueError,match='fit/calibration'):
        export_population(tmp_path,scope,reader,client,lambda:None,output_cap_bytes=1024**2)
    assert client.calls==0


def test_failure_stops_without_retry_and_retains_completed_evidence(tmp_path):
    scope,reader,client=setup_case(tmp_path);request=client.request
    def fail(bundle):
        if client.calls==1:raise RuntimeError('teacher failure')
        return request(bundle)
    client.request=fail
    with pytest.raises(RuntimeError,match='teacher failure'):
        export_population(tmp_path,scope,reader,client,lambda:None,output_cap_bytes=1024**2)
    assert client.calls==1
    assert len(list((tmp_path/'artifacts').iterdir()))==1
    log=[json.loads(s) for s in (tmp_path/'logs/observations.jsonl').read_text().splitlines()]
    assert log[-1]['state']=='STARTED' and log[-1]['source']['source_sequence_id']==1


def test_output_cap_does_not_drop_witnesses_to_fit(tmp_path):
    scope,reader,client=setup_case(tmp_path)
    with pytest.raises(OSError,match='output cap'):
        export_population(tmp_path,scope,reader,client,lambda:None,output_cap_bytes=1)
    assert client.calls==1 and not list((tmp_path/'artifacts').iterdir())
