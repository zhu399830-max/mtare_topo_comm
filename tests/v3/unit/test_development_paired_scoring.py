from types import SimpleNamespace
import torch
import pytest
import mtare_topo.evaluation.development_paired_scoring as m
from mtare_topo.representation.anchor_branch_loss import AnchorBranchPrediction,PartialAnchorBranches


def test_nonfinite_logits_cannot_disappear_during_thresholding():
    p=AnchorBranchPrediction(torch.zeros((1,3)),torch.tensor([float('nan')]),torch.tensor([[[1.,0.,0.]]]),torch.ones((1,1)))
    with pytest.raises(FloatingPointError,match='before selection'):
        m.score_observation(p,None,matching_radius_m=4.,matching_angle_deg=10.)


def test_zero_selected_predictions_count_all_supported_targets_as_missed(monkeypatch):
    def coverage(bundle,grid,positions,**kwargs):
        assert positions.shape==(0,3)
        return dict(query_scoreable_mask=[],possible_unconfirmed_reference_mask=[])
    monkeypatch.setattr(m,'bound_anchor_query_coverage',coverage)
    t=PartialAnchorBranches(torch.zeros((1,3)),(torch.tensor([[1.,0.,0.],[-1.,0.,0.],[0.,1.,0.]]),),False,(True,))
    o=SimpleNamespace(source={'id':'synthetic'},split='fit',loss_only=dict(target=t,bundle={},grid=None,
        produced_targets={},frozen_manifest=dict(source_binding={},target_record_sha256='test')))
    p=AnchorBranchPrediction(torch.zeros((32,3)),torch.full((32,),-2.3),
        torch.tensor([1.,0.,0.]).expand(32,64,3),torch.ones((32,64)))
    r=m.score_observation(p,o,matching_radius_m=4.,matching_angle_deg=10.)
    assert r['anchors']['tp']==0 and r['anchors']['fn']==1
    assert r['branches']['tp']==0 and r['branches']['fn']==3
    assert r['anchors']['known_fp']==r['anchors']['unresolved_predictions']==0
    assert r['branches']['known_fp']==r['branches']['unresolved_predictions']==0
    assert r['selected_anchor_query_indices']==[]
    assert not r['physical_annotation_complete']


def test_bound_masks_and_missing_node_propagate_to_branch_counts(monkeypatch):
    captured={}
    def coverage(bundle,grid,positions,**kwargs):
        captured.update(kwargs)
        return dict(query_scoreable_mask=[False]*len(positions),possible_unconfirmed_reference_mask=[True]*len(positions))
    monkeypatch.setattr(m,'bound_anchor_query_coverage',coverage)
    t=PartialAnchorBranches(torch.zeros((1,3)),(torch.tensor([[1.,0.,0.]]),),False,(True,))
    o=SimpleNamespace(source={'id':'synthetic'},split='development',loss_only=dict(target=t,bundle={},grid=None,
        produced_targets={},frozen_manifest=dict(source_binding={},target_record_sha256='test',target_reference_indices=(0,))))
    p=AnchorBranchPrediction(torch.zeros((1,3)),torch.ones(1),torch.tensor([[[1.,0.,0.]]]),torch.ones((1,1)))
    result=m.score_observation(p,o,matching_radius_m=4.,matching_angle_deg=10.)
    assert result['anchors']['tp']==0 and result['anchors']['unresolved_predictions']==1
    assert result['branches']['fn']==1 and result['branches']['known_fp']==0
    assert result['branches']['f1'] is None and result['provenance_verified']
    assert set(captured['manifest_row'])=={'source_binding','target_record_sha256'}
