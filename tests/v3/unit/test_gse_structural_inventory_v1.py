import copy
import pytest
from test_gse_surface_selection_v1 import fixture_report
from mtare_topo.data.gse_structural_inventory_v1 import inventory_parent
from mtare_topo.data.gse_surface_selection_v1 import VARIANTS


def fixture():
    report = fixture_report(edges=3)
    for interval in report['intervals']:
        interval['traversal_id'] = interval['traversal_id'].replace(':edge', ':edge_')
    def group(node, keys):
        return dict(node_id_teacher_only=node, paths=[dict(endpoint_key=k) for k in keys])
    groups = [group('junction', [('primitive:edge_000', 0), ('primitive:edge_001', 0), ('primitive:edge_002', 0)]),
              group('terminal', [('primitive:edge_000', 1)]),
              group('ordinary', [('primitive:edge_001', 1), ('primitive:edge_002', 1)]),
              group('no_history', [('primitive:edge_999', 0)])]
    return report, {v: copy.deepcopy(groups) for v in VARIANTS}


def test_entities_not_frames_and_background_not_negative():
    report, groups = fixture()
    output = inventory_parent(report, 'fit', groups)
    assert len(output['entities']) == 3
    entities = {e['node_id_teacher_only']: e for e in output['entities']}
    assert len(entities['junction']['causal_traversals']) == 6
    assert len(entities['terminal']['causal_traversals']) == 2
    assert entities['no_history']['causal_traversals'] == []
    assert not output['background_is_negative_label']
    assert output['observations_selected'] == 0
    assert all(not e['observable_label'] for e in output['entities'])


def test_order_invariant():
    report, groups = fixture()
    expected = inventory_parent(report, 'fit', groups)
    report['intervals'].reverse()
    for g in groups.values():
        g.reverse()
        for node in g:
            node['paths'].reverse()
    assert inventory_parent(report, 'fit', groups) == expected


@pytest.mark.parametrize('change', ['variant_missing', 'topology_drift', 'duplicate_incidence', 'test_parent'])
def test_invalid_scope_rejected(change):
    report, groups = fixture()
    if change == 'variant_missing':
        del groups['ellipse']
    elif change == 'topology_drift':
        groups['ellipse'][0]['paths'].pop()
    elif change == 'duplicate_incidence':
        groups['ellipse'][1]['paths'][0]['endpoint_key'] = ('primitive:edge_000', 0)
    else:
        report['parent_id'] = 'S01_flat_tree_small_C08'
    with pytest.raises(ValueError):
        inventory_parent(report, 'fit', groups)
