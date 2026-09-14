from mtare_topo.teacher.gse_reference_query_coverage_v1 import reference_query_coverage


def test_bound_empty_selection_preserves_integer_grid_states():
    import numpy as np
    from copy import deepcopy
    from test_gse_reference_exclusion_binding_v1 import fixture
    from mtare_topo.data.gse_structure_review_v1 import canonical_sha
    from mtare_topo.teacher.gse_reference_query_coverage_v1 import bound_anchor_query_coverage as v1
    from mtare_topo.teacher.gse_reference_query_coverage_v2 import bound_anchor_query_coverage as v2
    b,g,_,binding=fixture()
    record=dict(source_frame_indices=[0,1,2,3,4],anchors=[],
                score_region=dict(center_m=[0,0,0],radius_m=10.))
    manifest=dict(source_binding=deepcopy(binding),target_record_sha256=canonical_sha(record))
    produced=dict(record=record,**deepcopy(manifest))
    for bound in (v1,v2):
        r=bound(b,g,np.empty((0,3)),produced_targets=produced,
                manifest_row=manifest,matching_radius_m=4.)
        assert r['query_scoreable_mask']==[]
        assert r['possible_unconfirmed_reference_mask']==[]
        assert r['reference_negative_mask']==[]
        assert r['reference_count']==2
        assert not r['physical_annotation_complete']


def score(refs, confirmed, queries, states):
    return reference_query_coverage(query_xyz_m=queries,observed_state=states,
        all_reference_xyz_m=refs,confirmed_reference_indices=confirmed,matching_radius_m=1.,
        score_center_m=[0,0,0],score_radius_m=10.)


def test_repeated_predictions_near_confirmed_reference_are_scoreable():
    r=score([[0,0,0]],[0],[[.1,0,0],[.2,0,0]],[1,1])
    assert r['query_scoreable_mask']==[True,True]
    assert r['reference_negative_mask']==[False,False]
    assert r['full_f1'] is None


def test_unconfirmed_competitor_prevents_false_penalty():
    r=score([[0,0,0],[.5,0,0]],[0],[[.1,0,0]],[1])
    assert r['query_scoreable_mask']==[False]


def test_unobserved_and_outside_sphere_stay_unknown():
    r=score([[0,0,0]],[0],[[.1,0,0],[8,8,0]],[0,1])
    assert r['query_scoreable_mask']==[False,False]


def test_reference_exclusion_is_a_subset_of_scoreable_observed_space():
    r=score([[0,0,0]],[],[[3,0,0],[.1,0,0]],[1,1])
    assert r['query_scoreable_mask']==[True,False]


def test_full_source_grid_and_positive_manifest_binding():
    import pytest
    from copy import deepcopy
    from test_gse_reference_exclusion_binding_v1 import fixture
    from mtare_topo.data.gse_structure_review_v1 import canonical_sha
    from mtare_topo.teacher.gse_reference_query_coverage_v1 import bound_anchor_query_coverage
    b,g,q,binding=fixture()
    record=dict(source_frame_indices=[0,1,2,3,4],anchors=[],
                score_region=dict(center_m=[0,0,0],radius_m=10.))
    manifest=dict(source_binding=deepcopy(binding),target_record_sha256=canonical_sha(record))
    produced=dict(record=record,**deepcopy(manifest))
    result=bound_anchor_query_coverage(b,g,q,produced_targets=produced,
        manifest_row=manifest,matching_radius_m=4.)
    assert result['reference_count']==2 and result['confirmed_reference_count']==0
    assert result['query_scoreable_mask']==[True]
    assert not result['reference_inventory_complete_supplied_not_verified']
    produced['record']['source_frame_indices'][-1]=99
    with pytest.raises(ValueError,match='binding'):
        bound_anchor_query_coverage(b,g,q,produced_targets=produced,
            manifest_row=manifest,matching_radius_m=4.)
