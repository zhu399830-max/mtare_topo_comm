import copy
import pytest

from test_gse_structural_inventory_v1 import fixture
from mtare_topo.data.gse_structural_supplement_v1 import (
    nominate_parent, choose_position, merge_source_requests)
from mtare_topo.data.gse_surface_selection_v1 import VARIANTS


def inputs():
    report, groups = fixture()
    nominations = nominate_parent(report, 'fit', groups)
    nomination = next(n for n in nominations if n['kind'] == 'junction')
    positions = {v: [[8., 0., 0.], [1., 0., 0.], [4., 0., 0.]] for v in VARIANTS}
    anchors = {v: [0., 0., 0.] for v in VARIANTS}
    return nomination, positions, anchors


def test_nomination_order_and_missing_history():
    report, groups = fixture()
    expected = nominate_parent(report, 'fit', groups)
    report['intervals'].reverse()
    for record in report['intervals']:
        record['variants'].reverse()
    for g in groups.values():
        g.reverse()
    assert nominate_parent(report, 'fit', groups) == expected
    assert len(expected) == 3
    assert sum(n['status'] == 'NO_CAUSAL_HISTORY' for n in expected) == 1


def test_actual_positions_not_direction_suffix_choose_window():
    n, p, a = inputs()
    out = choose_position(n, decision_xyz_by_variant=p, anchor_xyz_by_variant=a)
    assert out['decision_index'] == 1
    assert len({s['source_sequence_id'] for s in out['sources']}) == 1
    assert all(len(s['frame_rows']) == 5 for s in out['sources'])
    assert not out['observable_label'] and not out['continuous_route_evidence']


def test_worst_variant_and_height_not_xy_only():
    n, p, a = inputs()
    p['ellipse'][1] = [0., 0., 20.]
    out = choose_position(n, decision_xyz_by_variant=p, anchor_xyz_by_variant=a)
    assert out['decision_index'] == 2


def test_rigid_transform_invariance():
    n, p, a = inputs()
    before = choose_position(n, decision_xyz_by_variant=p, anchor_xyz_by_variant=a)
    def transform(x):
        return [-x[1] + 3., x[0] - 4., x[2] + 7.]
    after = choose_position(n, decision_xyz_by_variant={v: list(map(transform, p[v])) for v in VARIANTS},
                            anchor_xyz_by_variant={v: transform(a[v]) for v in VARIANTS})
    assert before == after


def test_far_samples_retained_without_resampling():
    n, p, a = inputs()
    out = choose_position(n, decision_xyz_by_variant=p,
                          anchor_xyz_by_variant={v: [0., 0., 30.] for v in VARIANTS})
    assert len(out['sources']) == 3 and not out['all_variants_within_10m']
    assert out['traversal_id'] == n['traversal_id']


@pytest.mark.parametrize('mode', ['nan', 'missing', 'row_count'])
def test_invalid_poses_fail(mode):
    n, p, a = inputs()
    if mode == 'nan':
        p['ellipse'][0][0] = float('nan')
    elif mode == 'missing':
        del p['ellipse']
    else:
        p['ellipse'].pop()
    with pytest.raises(ValueError):
        choose_position(n, decision_xyz_by_variant=p, anchor_xyz_by_variant=a)


def test_shared_windows_dedup_not_independent_entities():
    n, p, a = inputs()
    s = choose_position(n, decision_xyz_by_variant=p, anchor_xyz_by_variant=a)
    other = dict(s, node_id_teacher_only='second_node')
    out = merge_source_requests(s['sources'], [s, other])
    assert len(out) == 3
    assert all(len(r['sampling_roles']) == 3 for r in out)
    assert out == merge_source_requests(list(reversed(s['sources'])), [other, s])
    bad = copy.deepcopy(s)
    bad['sources'][0]['frame_rows'][0] += 1
    with pytest.raises(ValueError, match='conflicting'):
        merge_source_requests(s['sources'], [bad])
