from dataclasses import replace
import numpy as np
import pytest
from mtare_topo.integration.structural_token_retrieval import FrameBinding, LocalTokenRecord, retrieve_structural_candidates


def record(name, order, vectors, xyz=None, segment='s'):
    vectors = np.asarray(vectors, dtype=np.float32)
    return LocalTokenRecord(name, segment, order, (name + ':scan',),
        np.array(xyz if xyz is not None else [[i, 0, 0] for i in range(len(vectors))]), vectors,
        (FrameBinding(name + ':scan', order * 1000, '0' * 64),), '1' * 64, '2' * 64)


def test_features_change_retrieval_not_registration_or_graph():
    a, b = np.eye(128, dtype=np.float32)[:2]
    history = [record('a', 0, [a]), record('b', 1, [b])]
    first = retrieve_structural_candidates(record('q', 2, [a]), history)
    second = retrieve_structural_candidates(record('q', 2, [b]), history)
    assert first['candidates'][0]['record_id'] == 'a'
    assert second['candidates'][0]['record_id'] == 'b'
    assert not first['graph_mutated'] and first['registration_required']
    assert not first['candidates'][0]['association_verified']


def test_no_global_mean_collapse_and_order_invariance():
    a, b = np.eye(128, dtype=np.float32)[:2]
    x = record('x', 0, [a, -a])
    y = record('y', 1, [b, -b])
    q = record('q', 2, [a, -a])
    result = retrieve_structural_candidates(q, [x, y])
    swapped = replace(q, xyz_current_sensor_m=q.xyz_current_sensor_m[::-1], tokens=q.tokens[::-1])
    assert result == retrieve_structural_candidates(swapped, [y, x])
    assert result['candidates'][0]['record_id'] == 'x'
    assert result['candidates'][0]['score'] > result['candidates'][1]['score']


def test_five_only_and_ties_do_not_mean_unique_identity():
    a = np.eye(128, dtype=np.float32)[0]
    result = retrieve_structural_candidates(record('q', 10, [a]), [record(str(i), i, [a]) for i in range(8)])
    assert len(result['candidates']) == 5 and result['eligible_records'] == 8
    assert {r['score'] for r in result['candidates']} == {1.0}
    assert all(not r['association_verified'] for r in result['candidates'])


@pytest.mark.parametrize('vectors,xyz', [([np.zeros(128)], [[0, 0, 0]]), ([np.ones(128)], [[11, 0, 0]])])
def test_unsupported_is_unknown(vectors, xyz):
    a = np.eye(128)[0]
    result = retrieve_structural_candidates(record('q', 2, vectors, xyz), [record('past', 0, [a])])
    assert not result['candidates'] and result['unknown']


def test_future_cross_segment_and_mutability():
    a = np.eye(128)[0]
    q = record('q', 1, [a])
    with pytest.raises(ValueError, match='past'):
        retrieve_structural_candidates(q, [record('future', 2, [a])])
    result = retrieve_structural_candidates(q, [record('past', 0, [a], segment='another')])
    assert result['unknown'][0]['reason'] == 'different_segment'
    with pytest.raises(ValueError):
        q.tokens[0, 0] = 0
    with pytest.raises(ValueError):
        q.tokens.setflags(write=True)


def test_false_order_does_not_override_timestamp_or_version():
    a = np.eye(128)[0]
    q = record('q', 3, [a])
    future = replace(record('future', 9, [a]), order=1)
    with pytest.raises(ValueError, match='past'):
        retrieve_structural_candidates(q, [future])
    with pytest.raises(ValueError, match='versions'):
        retrieve_structural_candidates(q, [replace(record('p', 0, [a]), token_model_sha256='3' * 64)])
