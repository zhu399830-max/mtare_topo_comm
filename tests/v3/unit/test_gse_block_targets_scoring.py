import pytest
import torch
from mtare_topo.data.gse_block_targets import located_targets
from mtare_topo.evaluation.gse_block_structure_scoring import prediction_record,score_record
from mtare_topo.representation.gse_block_structure_readout import BlockStructurePrediction


def reference():
    return dict(anchors=[[0.,0,0]],openings=[dict(position_m=[10.,0,0],direction=[1.,0,0])],complete_for_declared_fixture=True)


def prediction():
    a=torch.zeros(32,3);o=torch.zeros(64,3);o[:,0]=10
    al=torch.full((32,),-10.);ol=torch.full((64,),-10.);al[0]=0;ol[0]=0
    d=torch.zeros(64,3);d[:,0]=1
    return BlockStructurePrediction(a,al,o,ol,d,True)


def test_threshold_half_and_exact_geometry():
    r=score_record(prediction_record(prediction()),reference())
    assert r['anchors']['1.0']['f1']==r['openings']['f1']==1
    assert r['opening_direction_errors_deg']==[0.]


def test_duplicate_predictions_count_false_positive_no_nms():
    p=prediction();p.anchor_presence_logits[1]=1;p.opening_presence_logits[1]=1
    r=score_record(prediction_record(p),reference())
    assert r['anchors']['1.0']['tp']==1 and r['anchors']['1.0']['fp']==1
    assert r['openings']['fp']==1


def test_empty_selection_counts_misses_not_pass():
    p=prediction();p.anchor_presence_logits[:]=-10;p.opening_presence_logits[:]=-10
    r=score_record(prediction_record(p),reference())
    assert r['anchors']['1.0']['fn']==r['openings']['fn']==1
    assert r['openings']['f1']==0


def test_unknown_not_empty_background():
    r=dict(anchors=None,openings=None,complete_for_declared_fixture=False)
    t=located_targets(r)
    assert not t.anchors_complete and len(t.anchors_m)==0
    assert score_record(prediction_record(prediction()),r)['status']=='UNSCORED_INCOMPLETE_REFERENCE'
    r['complete_for_declared_fixture']=True
    with pytest.raises(ValueError):located_targets(r)


def test_unknown_direction_and_capacity():
    r=reference();r['openings'][0]['direction']=None;t=located_targets(r)
    assert not t.direction_known[0] and torch.isnan(t.opening_direction).all()
    r['anchors']=[[0.,0,0]]*33
    with pytest.raises(OverflowError):located_targets(r)


def test_nonfinite_logit_cannot_hide_behind_sigmoid():
    p=prediction();p.anchor_presence_logits[0]=float('inf')
    with pytest.raises(ValueError):prediction_record(p)
