import pytest
from test_gse_surface_target_adapter_v1 import record
from mtare_topo.data.gse_surface_review_v1 import SurfaceReview, validate_bundle
from mtare_topo.data.gse_structure_review_v1 import canonical_sha


def bundle():
    return dict(schema="gse_surface_review_bundle_v1",observation_id="opaque01",
        coordinate_frame="current_sensor_m",source_frame_indices=[1,2,3,4,5],
        points_xyz_m=[[.1,.2,.3]],point_history_slots=[4])


def test_one_observation_independent_opening_unknown_and_locked_review():
    b=bundle();r=SurfaceReview(b,reviewer="synthetic_test_only")
    a=record();a['anchors']=[];a['membership']=[[]]
    with pytest.raises(ValueError):r.reveal({})
    r.commit_blind(a);a['openings'].clear()
    with pytest.raises(ValueError):r.commit_blind(record())
    r.reveal(dict(bundle_sha256=canonical_sha(b),construction_reference={}))
    r.finish("synthetic protocol test, not human labeling")
    out=r.export()
    assert len(out['blind_annotation']['openings'])==1
    assert not out['automatic_training_eligibility'] and not out['human_identity_authenticated']
    assert out['review_protocol_complete']
    out['blind_annotation']['openings'].clear()
    assert len(r.export()['blind_annotation']['openings'])==1


@pytest.mark.parametrize('fault',['teacher','future','slot','nan'])
def test_invalid_blind_input(fault):
    b=bundle()
    if fault=='teacher':b['node_id']='hidden'
    elif fault=='future':b['source_frame_indices']=[1,2,6,4,5]
    elif fault=='slot':b['point_history_slots']=[5]
    else:b['points_xyz_m']=[[float('nan'),0,0]]
    with pytest.raises(ValueError):validate_bundle(b)


def test_wrong_reference_and_source_rejected():
    r=SurfaceReview(bundle(),reviewer="synthetic")
    a=record();a['source_frame_indices']=[2,3,4,5,6]
    with pytest.raises(ValueError):r.commit_blind(a)
    r.commit_blind(record())
    with pytest.raises(ValueError):r.reveal(dict(bundle_sha256='wrong',construction_reference={}))
