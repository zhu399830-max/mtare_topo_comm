from dataclasses import replace
import torch
from test_gse_surface_losses_v1 import prediction,targets
from mtare_topo.evaluation.gse_surface_prediction_scoring_v1 import score_surface_prediction


def score(p,t):
    return score_surface_prediction(p,t,anchor_threshold=.5,opening_threshold=.5,
        opening_maximum_error_m=1.)['observations'][0]


def test_partial_targets_never_become_full_f1():
    p,t=prediction(),targets()
    row=score(p,t)
    assert row['anchor_main_4m']['true_positive']==2
    assert row['anchor_main_4m']['full_f1'] is None
    assert row['anchor_main_4m']['unknown_unmatched_predictions']==30
    assert not row['full_detection_scoring_available']
    assert p.anchor_position_m.grad is None


def test_explicit_complete_empty_region_counts_every_extra_prediction():
    p,t=prediction(),targets()
    p=replace(p,anchor_position_m=torch.zeros_like(p.anchor_position_m),
        opening_position_m=torch.zeros_like(p.opening_position_m))
    t=replace(t,anchor_valid=torch.zeros_like(t.anchor_valid),opening_valid=torch.zeros_like(t.opening_valid),
        membership_valid=torch.zeros_like(t.membership_valid),direction_valid=torch.zeros_like(t.direction_valid),
        dimension_valid=torch.zeros_like(t.dimension_valid),reachability_valid=torch.zeros_like(t.reachability_valid),
        physical_reference_valid=torch.zeros_like(t.physical_reference_valid),
        anchor_region_complete=torch.ones_like(t.anchor_region_complete),
        opening_region_complete=torch.ones_like(t.opening_region_complete))
    row=score(p,t)
    assert row['anchor_main_4m']['known_region_false_positive']==32
    assert row['openings']['known_region_false_positive']==64
    assert row['anchor_main_4m']['full_f1']==0.
    assert row['full_detection_scoring_available']


def test_all_anchor_matching_radii_are_reported_without_best_selection():
    row=score(prediction(),targets())
    assert all(k in row for k in ('anchor_main_4m','anchor_sensitivity_1m','anchor_sensitivity_2m'))
