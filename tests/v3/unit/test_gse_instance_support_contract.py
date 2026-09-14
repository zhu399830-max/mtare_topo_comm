import numpy as np
import pytest
from mtare_topo.teacher.gse_instance_support_contract import inspect_instance_support


def inputs():
    return np.array([[0., 0., 0.], [1., 0., 0.]]), np.array([[0., 1., 0.], [0., -1., 0.]])


def test_centers_alone_cannot_supply_membership():
    with pytest.raises(ValueError, match='explicit'):
        inspect_instance_support(*inputs(), positive=None, known=None)


def test_unknown_stays_unknown_not_nearest_center():
    z = np.zeros((2, 2), bool)
    r = inspect_instance_support(*inputs(), positive=z, known=z)
    assert r['unknown_pairs'] == 4 and r['negative_pairs'] == 0
    assert r['unsupported_anchor_indices'] == [0, 1]
    assert not r['center_vote_supervision_available']


def test_multiple_support_not_forced_unique():
    p = np.array([[True, True], [False, False]])
    r = inspect_instance_support(*inputs(), positive=p, known=p)
    assert r['multiply_supported_points'] == 1
    assert r['center_vote_supervision_available']
    assert not r['teacher_qualified'] and not r['training_authorized']


def test_explicit_negative_is_distinct_from_unknown():
    p = np.eye(2, dtype=bool)
    r = inspect_instance_support(*inputs(), positive=p, known=np.ones((2, 2), bool))
    assert (r['positive_pairs'], r['negative_pairs'], r['unknown_pairs']) == (2, 2, 0)


def test_unknown_positive_rejected():
    with pytest.raises(ValueError, match='unknown'):
        inspect_instance_support(*inputs(), positive=np.eye(2, dtype=bool), known=np.zeros((2, 2), bool))


def test_same_centers_allow_distinct_support_assignments():
    x, a = inputs()
    p = np.eye(2, dtype=bool)
    q = p[:, ::-1].copy()
    assert inspect_instance_support(x, a, positive=p, known=np.ones_like(p)) == inspect_instance_support(x, a, positive=q, known=np.ones_like(q))
    # Both contracts are well formed; centers alone cannot choose which
    # distinct point-wise offsets are scientifically justified.
    assert not np.array_equal(a[np.argmax(p, axis=1)] - x, a[np.argmax(q, axis=1)] - x)
