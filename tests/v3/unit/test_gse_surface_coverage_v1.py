from mtare_topo.data.gse_surface_coverage_v1 import nearby_references, aggregate_coverage


def test_strict_roi_and_degree_two_separation():
    groups = [dict(node_id_teacher_only=str(i), anchor_world_m=(x, 0, 0), paths=[None]*d)
              for i, (x, d) in enumerate([(0, 1), (1, 2), (2, 3), (10, 1)])]
    r = nearby_references(groups, [0, 0, 0])
    assert [x['kind'] for x in r] == ['terminal', 'construction_segment', 'junction']


def test_variants_do_not_inflate_independent_entities():
    reference = dict(kind='terminal', node_id_teacher_only='n')
    outputs = [dict(observations=[dict(source=dict(split='fit', parent_id='p'), nearby=[reference])])]*3
    r = aggregate_coverage(outputs)['fit']['by_kind']['terminal']
    assert r == dict(independent_entities=1, parents=1, observations=3)


def test_same_node_name_in_different_parent_is_not_same_entity():
    outputs = [dict(observations=[dict(source=dict(split='fit', parent_id=p),
                    nearby=[dict(kind='junction', node_id_teacher_only='n')])]) for p in ('p', 'q')]
    assert aggregate_coverage(outputs)['fit']['by_kind']['junction']['independent_entities'] == 2
