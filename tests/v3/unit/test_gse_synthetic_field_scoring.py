from mtare_topo.data.gse_synthetic_matrix import matrix
from mtare_topo.evaluation.gse_synthetic_field_scoring import expected_geometry,score_case,match_positions


def record(anchors,openings):
    return dict(coordinate_frame='current_sensor_m',anchors=[dict(position_m=x) for x in anchors],openings=openings,
        score_region=dict(anchors_complete=False,openings_complete=False))


def test_coverage_fixed_and_unknown_not_pass():
    rows=matrix();covered=[c for c in rows if expected_geometry(c)['complete_for_declared_fixture']]
    assert len(covered)==45
    unknown=next(c for c in rows if not expected_geometry(c)['complete_for_declared_fixture'])
    assert score_case(unknown,record([],[]))['status']=='UNSCORED_NOT_PASS'
    for c in covered:
        oracle=expected_geometry(c)
        assert score_case(c,record(oracle['anchors'],oracle['openings']))['status']=='FIXTURE_GEOMETRY_PASS'
        assert score_case(c,record([],[]))['status']=='FIXTURE_GEOMETRY_FAIL'


def test_extra_and_duplicate_predictions_count_as_false_positives():
    r=match_positions([[0,0,0]],[[0,0,0],[0,0,0],[20,0,0]],4.)
    assert (r['tp'],r['fp'],r['fn'])==(1,2,0)
    assert not r['exact_count_and_matching_pass']


def test_matching_maximizes_valid_cardinality_before_distance():
    r=match_positions([[0,0,0],[1,0,0]],[[.1,0,0],[-.8,0,0]],1.)
    assert r['tp']==2 and r['fp']==r['fn']==0
