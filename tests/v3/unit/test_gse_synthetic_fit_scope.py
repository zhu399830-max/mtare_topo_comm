from collections import Counter
from mtare_topo.data.gse_synthetic_fit_scope import declared_cases


def test_scope_is_declared_not_selected_by_prediction_or_score():
    cases=declared_cases()
    assert len(cases)==45 and len({c['case_id'] for c in cases})==45
    assert Counter(c['program']['type'] for c in cases)==dict(
        straight=12,terminal=12,visible_blocker=12,T=3,Y=3,four_way=3)
    assert all(c['case_id'].endswith('view2') for c in cases if c['program']['type'] in ('T','Y','four_way'))


def test_callers_cannot_mutate_future_declarations():
    first=declared_cases();first[0]['poses_world_m'][0][0]=999
    assert declared_cases()[0]['poses_world_m'][0][0]!=999
