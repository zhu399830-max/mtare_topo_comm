import numpy as np
import pytest
from mtare_topo.evaluation.gse_reference_covered_counts_v1 import covered_detection_counts


def kwargs():
    return dict(predicted_xyz_m=[[.1,0,0],[.2,0,0],[8,0,0]],confidence=[.9,.99,.8],
        target_xyz_m=[[0,0,0]],threshold=.5,maximum_error_m=1.,
        score_center_m=[0,0,0],score_radius_m=10.)


def test_duplicate_counted_unknown_not_penalized_geometry_not_confidence():
    r=covered_detection_counts(query_scoreable_mask=[True,True,False],**kwargs())
    assert r['matches'][0][0]==0  # nearer despite lower confidence
    assert (r['true_positive'],r['known_region_false_positive'],r['unknown_unmatched_predictions'])==(1,1,1)
    assert r['full_f1'] is None and not r['full_scoring_available']


def test_query_permutation_keeps_counts():
    a=kwargs();order=[2,0,1]
    a['predicted_xyz_m']=np.array(a['predicted_xyz_m'])[order]
    a['confidence']=np.array(a['confidence'])[order]
    r=covered_detection_counts(query_scoreable_mask=[False,True,True],**a)
    assert (r['true_positive'],r['known_region_false_positive'],r['unknown_unmatched_predictions'])==(1,1,1)


def test_unconfirmed_competitor_retains_duplicate_as_unknown():
    r=covered_detection_counts(query_scoreable_mask=[False,False,False],**kwargs())
    assert r['known_region_false_positive']==0 and r['unknown_unmatched_predictions']==2
    assert r['selected_queries_without_reference_coverage']==3


def test_no_empty_prediction_success_score():
    a=kwargs();a['confidence']=[0.,0.,0.]
    r=covered_detection_counts(query_scoreable_mask=[True,True,False],**a)
    assert r['false_negative']==1 and r['full_f1'] is None


def test_invalid_coverage_rejected():
    with pytest.raises(ValueError):
        covered_detection_counts(query_scoreable_mask=[1,1,0],**kwargs())
    a=kwargs();a['score_radius_m']=4.
    with pytest.raises(ValueError,match='sphere'):
        covered_detection_counts(query_scoreable_mask=[True,True,True],**a)


def test_bound_scoring_uses_same_source_and_reports_all_three_radii():
    from copy import deepcopy
    from test_gse_reference_exclusion_binding_v1 import fixture
    from mtare_topo.data.gse_structure_review_v1 import canonical_sha
    from mtare_topo.evaluation.gse_reference_covered_counts_v1 import score_bound_anchor_references
    b,g,q,binding=fixture()
    record=dict(source_frame_indices=[0,1,2,3,4],anchors=[],
        score_region=dict(center_m=[0,0,0],radius_m=10.))
    manifest=dict(source_binding=binding,target_record_sha256=canonical_sha(record))
    produced=dict(record=record,**deepcopy(manifest))
    r=score_bound_anchor_references(bundle=b,grid=g,produced_targets=produced,
        manifest_row=manifest,predicted_xyz_m=np.concatenate([q,-q]),confidence=[.9,.9],threshold=.5)
    for key in ('anchor_main_4m','anchor_sensitivity_1m','anchor_sensitivity_2m'):
        assert r[key]['counts']['known_region_false_positive']==1
        assert r[key]['counts']['unknown_unmatched_predictions']==1
        assert not r[key]['counts']['reference_coverage_supplied_not_verified']
    assert r['full_f1'] is None


def test_openings_use_own_inventory_and_explicit_matching_radius():
    from copy import deepcopy
    from test_gse_reference_exclusion_binding_v1 import fixture
    from mtare_topo.data.gse_structure_review_v1 import canonical_sha
    from mtare_topo.evaluation.gse_reference_covered_counts_v1 import score_bound_opening_references
    b,g,q,binding=fixture()
    record=dict(source_frame_indices=[0,1,2,3,4],anchors=[],openings=[],
        score_region=dict(center_m=[0,0,0],radius_m=10.))
    manifest=dict(source_binding=binding,target_record_sha256=canonical_sha(record))
    produced=dict(record=record,**deepcopy(manifest))
    options=dict(bundle=b,grid=g,produced_targets=produced,manifest_row=manifest,
                 predicted_xyz_m=np.concatenate([q,-q]),confidence=[.9,.9],threshold=.5)
    with pytest.raises(TypeError):
        score_bound_opening_references(**options)  # no implicit anchor4m
    result=score_bound_opening_references(**options,maximum_error_m=1.)
    assert result['coverage']['reference_kind']=='openings'
    assert result['coverage']['reference_count']==0  # two anchors, no sphere crossings
    assert result['counts']['known_region_false_positive']==1
    assert result['counts']['unknown_unmatched_predictions']==1
    # A confirmed anchor cannot be passed off as an opening reference.
    record['openings']=[dict(position_m=[0.,1.,0.])]
    manifest['target_record_sha256']=canonical_sha(record)
    produced['target_record_sha256']=manifest['target_record_sha256']
    with pytest.raises(ValueError,match='uniquely match'):
        score_bound_opening_references(**options,maximum_error_m=1.)
