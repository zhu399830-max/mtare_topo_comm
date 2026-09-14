import numpy as np
from mtare_topo.teacher.gse_reference_query_coverage_v1 import reference_query_coverage as old
from mtare_topo.teacher.gse_reference_query_coverage_v2 import reference_query_coverage as new
from mtare_topo.evaluation.gse_reference_covered_counts_v1 import covered_detection_counts


def args():
    return dict(query_xyz_m=[[0.,0.,0.],[.1,0,0]],observed_state=[0,0],
        all_reference_xyz_m=[[0.,0.,0.]],confirmed_reference_indices=[0],
        matching_radius_m=1.,score_center_m=[0,0,0],score_radius_m=10.)


def test_duplicate_confirmed_structure_does_not_require_return_at_center():
    a=args()
    assert old(**a)['query_scoreable_mask']==[False,False]  # preserve old failure
    coverage=new(**a)
    assert coverage['query_scoreable_mask']==[True,True]
    assert coverage['observed_inside_score_region_mask']==[False,False]
    assert coverage['reference_negative_mask']==[False,False]
    count=covered_detection_counts(query_scoreable_mask=coverage['query_scoreable_mask'],
        predicted_xyz_m=a['query_xyz_m'],confidence=[1,1],target_xyz_m=a['all_reference_xyz_m'],
        threshold=.5,maximum_error_m=1.,score_center_m=[0,0,0],score_radius_m=10.)
    assert (count['true_positive'],count['known_region_false_positive'])==(1,1)
    assert count['full_f1'] is None


def test_hidden_competitor_still_vetoes_even_exact_confirmed_query():
    a=args(); a['all_reference_xyz_m'].append([.2,0,0])
    assert new(**a)['query_scoreable_mask']==[False,False]


def test_no_confirmation_does_not_invent_observation():
    a=args();a['confirmed_reference_indices']=[]
    assert new(**a)['query_scoreable_mask']==[False,False]
    a['query_xyz_m']=[[5.,0.,0.],[6.,0.,0.]];a['observed_state']=[1,0]
    assert new(**a)['query_scoreable_mask']==[True,False]


def test_confirmed_region_cannot_expand_score_sphere():
    a=args();a['query_xyz_m']=[[10.1,0,0],[9.9,0,0]]
    a['all_reference_xyz_m']=[[10.,0,0]]
    assert new(**a)['query_scoreable_mask']==[False,True]


def test_source_bound_confirmation_and_manifest_remain_required():
    from copy import deepcopy
    import pytest
    from test_gse_reference_exclusion_binding_v1 import fixture
    from mtare_topo.data.gse_structure_review_v1 import canonical_sha
    from mtare_topo.teacher.gse_reference_query_coverage_v2 import bound_anchor_query_coverage
    b,g,_,binding=fixture()
    record=dict(source_frame_indices=[0,1,2,3,4],anchors=[dict(position_m=[0.,1.,0.])],
        score_region=dict(center_m=[0,0,0],radius_m=10.))
    manifest=dict(source_binding=binding,target_record_sha256=canonical_sha(record))
    target=dict(record=record,**deepcopy(manifest))
    kw=dict(produced_targets=target,manifest_row=manifest,matching_radius_m=1.)
    r=bound_anchor_query_coverage(b,g,np.asarray([[0.,1.,0.],[0.,1.1,0.]]),**kw)
    assert r['query_scoreable_mask']==[True,True]
    from mtare_topo.evaluation.gse_reference_covered_counts_v2 import score_bound_references
    result=score_bound_references(kind='anchors',bundle=b,grid=g,
        produced_targets=target,manifest_row=manifest,maximum_error_m=1.,
        predicted_xyz_m=np.asarray([[0.,1.,0.],[0.,1.1,0.]]),confidence=[1,1],threshold=.5)
    assert result['counts']['true_positive']==1
    assert result['counts']['known_region_false_positive']==1
    assert not result['counts']['reference_coverage_supplied_not_verified']
    target['record']['anchors'][0]['position_m']=[0.,2.,0.]
    with pytest.raises(ValueError,match='binding'):
        bound_anchor_query_coverage(b,g,np.asarray([[0.,1.,0.]]),**kw)
