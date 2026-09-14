from copy import deepcopy
from mtare_topo.teacher.gse_window_opening_diagnostic_v1 import resolve_competition


def row(geometry, source, surface):
    return dict(geometric_outward_crossing_ray_indices=geometry,
                outward_crossing_ray_indices=source,surface_return_ray_indices=surface)


def test_unsupported_other_source_still_competes():
    rows=[row([1,2],[1,2],[3]),row([1],[],[])]
    original=deepcopy(rows);out=resolve_competition(rows)
    assert rows==original
    assert out[0]['exclusive_outward_crossing_ray_indices']==[2]
    assert out[0]['competing_crossing_ray_indices']==[1]
    assert out[0]['proposal_supported_after_competition']
    assert not out[1]['proposal_supported_after_competition']


def test_duplicate_geometry_cannot_make_two_supported_openings():
    out=resolve_competition([row([1],[1],[2]),row([1],[1],[2])])
    assert not any(r['proposal_supported_after_competition'] for r in out)


def test_candidate_permutation_does_not_choose_a_winner():
    rows=[row([1],[1],[2]),row([1,3],[3],[4])]
    assert resolve_competition(rows)==list(reversed(resolve_competition(list(reversed(rows)))))
