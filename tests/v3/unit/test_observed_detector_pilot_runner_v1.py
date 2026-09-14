from dataclasses import fields
from types import SimpleNamespace
import json
import numpy as np
import torch
import observed_detector_pilot_v1 as runner
from test_conditional_anchor_branch_loss import fixture
from mtare_topo.data.gse_structure_review_v1 import canonical_sha
from mtare_topo.representation.anchor_branch_loss import PartialAnchorBranches


def test_evaluator_saves_real_query_provenance_and_all_nine_scores(tmp_path,monkeypatch):
    p,t,k=fixture()
    # The loss fixture's origin is not a construction reference. Use the
    # established empty-partial-reference scoring fixture, not invented GT.
    t=PartialAnchorBranches(torch.empty((0,3),dtype=torch.float64),(),False,())
    record=dict(source_frame_indices=[0,1,2,3,4],anchors=[],score_region=dict(center_m=[0,0,0],radius_m=10.))
    k['produced_targets']=dict(record=record,source_binding=k['frozen_manifest']['source_binding'],target_record_sha256=canonical_sha(record))
    k['frozen_manifest']=dict(source_binding=k['frozen_manifest']['source_binding'],target_record_sha256=canonical_sha(record),target_reference_indices=())
    observation=SimpleNamespace(source=k['frozen_manifest']['source_binding']['source'],split='fit',
        loss_only=dict(k,target=t))
    output=SimpleNamespace(prediction=p,query_source_indices=torch.tensor([7,9,11]),
        query_positions_m=p.position_m.detach(),residual_m=torch.zeros_like(p.position_m))
    monkeypatch.setattr(runner,'forward_observation',lambda *a:output)
    for name in ('artifacts','metrics','previews'):(tmp_path/name).mkdir()
    saved,rows=runner.evaluate(torch.nn.Identity(),[observation],tmp_path,'initial',lambda:None,set())
    assert len(saved)==1 and len(rows)==9
    record=json.loads((tmp_path/'metrics/initial.jsonl').read_text())
    assert len(record['scores'])==9
    with np.load(tmp_path/'artifacts'/record['prediction_file']) as archive:
        assert archive['query_source_indices'].tolist()==[7,9,11]
        assert set(archive.files)=={f.name for f in fields(p)}|{'query_source_indices','query_positions_m','residual_m'}
    assert {s['matching_radius_m'] for s in record['scores']}=={1,2,4}
    assert {s['matching_angle_deg'] for s in record['scores']}=={5,10,15}


def test_count_aggregation_does_not_treat_unknown_as_tp():
    s=dict(anchors=dict(tp=2,fn=3,known_fp=4,unresolved_predictions=5),
           branches=dict(tp=6,fn=7,known_fp=8,unresolved_predictions=9,observed_reference_recall=.1))
    c=runner.counts(s)
    assert c['anchors_tp']==2 and c['anchors_unresolved_predictions']==5 and len(c)==8
