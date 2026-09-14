import pytest
from mtare_topo.data.gse_synthetic_matrix import matrix
from mtare_topo.evaluation.gse_synthetic_straight_oracle import visible_straight_cap_rays


def test_analytic_positive_controls_for_three_sections_and_registered_views():
    for case in matrix():
        kind=case['program']['type']
        if kind not in ('straight','terminal','visible_blocker'):continue
        result=visible_straight_cap_rays(case)
        assert not result[0]
        expected_visible=kind!='straight' and not case['case_id'].endswith('view0')
        assert bool(result[1])==expected_visible,case['case_id']
        assert all(0<=r<57600 for r in result[1])


def test_oracle_does_not_claim_general_junction_visibility():
    case=next(c for c in matrix() if c['program']['type']=='T')
    with pytest.raises(ValueError,match='only covers'):visible_straight_cap_rays(case)
