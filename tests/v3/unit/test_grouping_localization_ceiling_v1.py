import numpy as np
from mtare_topo.evaluation.grouping_localization_ceiling_v1 import localization_funnel


def actual(keep,pairs=(),conflicts=None):
    return dict(query_indices=keep,pairs=list(pairs),tp=len(pairs),
        coverage=dict(possible_unconfirmed_reference_mask=[False]*len(keep) if conflicts is None else conflicts))


def test_one_candidate_cannot_recover_two_references():
    r=localization_funnel([[0,0,0]],[2],[[-.5,0,0],[.5,0,0]],actual([0],[(0,0)]))
    assert r['raw_one_to_one_tp']==1 and r['localization_deficit']==1
    assert r['references'][1]['category']=='ONE_TO_ONE_COMPETITION'


def test_far_candidate_is_localization_failure_not_threshold_failure():
    r=localization_funnel([[2,0,0]],[10],[[0,0,0]],actual([0]))
    assert r['raw_one_to_one_tp']==0 and r['presence_deficit']==0
    assert r['references'][0]['category']=='NO_CANDIDATE_WITHIN_1M'


def test_low_confidence_near_candidate():
    r=localization_funnel([[.2,0,0]],[-2],[[0,0,0]],actual([]))
    assert r['raw_one_to_one_tp']==1 and r['selected_one_to_one_tp']==0 and r['presence_deficit']==1
    assert r['references'][0]['category']=='NEAR_CANDIDATE_BELOW_THRESHOLD'


def test_unknown_competitor_is_not_background():
    r=localization_funnel([[.2,0,0]],[2],[[0,0,0]],actual([0],conflicts=[True]))
    assert r['raw_one_to_one_tp']==r['selected_one_to_one_tp']==1
    assert r['actual_tp']==0 and r['coverage_or_matching_deficit']==1
    assert r['references'][0]['category']=='SELECTED_NEIGHBOR_EXCLUDED_BY_COVERAGE'


def test_cardinality_first_assignment_not_independent_nearests():
    r=localization_funnel([[0,0,0],[1.1,0,0]],[2,2],[[.2,0,0],[-.5,0,0]],actual([0,1],[(0,1),(1,0)]))
    assert r['raw_one_to_one_tp']==2 and r['localization_deficit']==0


def test_empty_targets_and_predictions_count_without_fake_labels():
    r=localization_funnel(np.empty((0,3)),[],np.empty((0,3)),actual([]))
    assert r['reference_count']==r['raw_one_to_one_tp']==0 and r['references']==[]
