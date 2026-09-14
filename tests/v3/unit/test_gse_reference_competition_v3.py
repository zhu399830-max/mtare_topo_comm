from copy import deepcopy
from mtare_topo.teacher.gse_window_opening_diagnostic_v3 import resolve_reference_competition


def proposal(source='p', owned=True):
    return dict(primitive_id_teacher_only=source, axis_reference_owned=owned,
        geometric_outward_crossing_ray_indices=[1, 2], outward_crossing_ray_indices=[1, 2],
        surface_return_ray_indices=[3])


def test_incidental_same_source_contour_preserved_not_competing_reference():
    rows = [proposal(), proposal(owned=False)]; old = deepcopy(rows)
    result = resolve_reference_competition(rows)
    assert rows == old and len(result) == 2
    assert result[0]['exclusive_outward_crossing_ray_indices'] == [1, 2]
    assert result[0]['same_source_nonreference_contour_indices'] == [1]


def test_same_source_distinct_owning_arcs_still_compete():
    assert not resolve_reference_competition([proposal(), proposal()])[0]['proposal_supported_after_competition']


def test_other_source_even_nonowning_still_competes():
    assert not resolve_reference_competition([proposal(), proposal('other', False)])[0]['proposal_supported_after_competition']
