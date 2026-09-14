import numpy as np
from mtare_topo.evaluation.branch_direction_scoring import score_direction_set


def test_duplicates_are_not_multiple_true_positives():
    r=score_direction_set([[1,0,0],[1,0,0]],[[1,0,0]],reference_complete=True,matching_angle_deg=10.)
    assert (r['tp'],r['fp'],r['fn'])==(1,1,0)


def test_partial_set_cannot_claim_full_f1_or_punish_unknown():
    r=score_direction_set([[1,0,0],[0,1,0]],[[1,0,0]],reference_complete=False,matching_angle_deg=10.)
    assert r['tp']==1 and r['unresolved_predictions']==1 and r['fp'] is None and r['f1'] is None


def test_opposite_direction_is_wrong_and_missing_prediction_counted():
    r=score_direction_set([[-1,0,0]],[[1,0,0]],reference_complete=True,matching_angle_deg=10.)
    assert (r['tp'],r['fp'],r['fn'])==(0,1,1)
    r=score_direction_set(np.empty((0,3)),[[1,0,0]],reference_complete=True,matching_angle_deg=10.)
    assert r['fn']==1 and r['f1']==0
