import numpy as np
import pytest
from mtare_topo.teacher.gse_terminal_reference_v1 import terminal_reference_evidence
from mtare_topo.teacher.gse_portal_ray_evidence import CausalRaySegments
from mtare_topo.teacher.swept_superellipse_field import SweptSuperellipsePrimitive


def fixture():
    p = SweptSuperellipsePrimitive('p', np.array([[-2., 0., 0.], [2., 0., 0.]]),
                                  ((1., 1.), (1., 1.)), (2., 2.))
    # Nonradial triangulation-edge return; exactly on the end plane.
    rays = CausalRaySegments(np.array([[0., .2, .03]]), np.array([[1., 0., 0.]]),
                            np.array([2.]), np.array([True]), np.array([4]), 4, 0.)
    group = dict(node_id_teacher_only='end', anchor_world_m=(2., 0., 0.),
                 paths=[dict(endpoint_key=('p', 1))])
    return [group], [p], rays


def run(groups, primitives, rays, sets=None):
    return terminal_reference_evidence(groups, primitives, rays=rays,
        membership_codes=np.array([1], dtype=np.uint16), source_sets=[[], [0]] if sets is None else sets)


def test_degree_one_cap_and_return_source_bound():
    groups, primitives, rays = fixture()
    r = run(groups, primitives, rays)
    assert r['terminal_references'][0]['terminal_reference_observed']
    assert not r['training_eligible'] and not r['model_input_fields']


def test_degree_two_cut_is_not_a_terminal():
    groups, primitives, rays = fixture()
    groups[0]['paths'].append(dict(endpoint_key=('p', 0)))
    assert run(groups, primitives, rays)['terminal_references'] == []


def test_multisource_return_cannot_be_forced_to_terminal():
    groups, primitives, rays = fixture()
    primitives.append(SweptSuperellipsePrimitive('q', np.array([[-2., 0., 0.], [2., 0., 0.]]),
                                                  ((1., 1.), (1., 1.)), (2., 2.)))
    r = run(groups, primitives, rays, [[], [0, 1]])['terminal_references'][0]
    assert not r['terminal_reference_observed']
    assert len(r['nonunique_or_other_source_witnesses']) == 1


def test_duplicate_endpoint_rejected_before_result():
    groups, primitives, rays = fixture()
    groups.append({**groups[0], 'node_id_teacher_only': 'other'})
    with pytest.raises(ValueError, match='ownership'):
        run(groups, primitives, rays)


def test_roi_boundary_is_excluded_without_creating_a_new_endpoint():
    groups, primitives, rays = fixture()
    result = terminal_reference_evidence(groups, primitives, rays=rays,
        membership_codes=np.array([1], dtype=np.uint16), source_sets=[[], [0]],
        local_center_m=[-8., 0., 0.])
    assert result['terminal_references'] == []


def test_denied_source_does_not_invalidate_scan():
    groups,primitives,rays=fixture()
    out=terminal_reference_evidence(groups,primitives,rays=rays,
        membership_codes=np.array([1],dtype=np.uint16),source_sets=[[],[0]],
        source_permission=np.array([False]))['terminal_references'][0]
    assert not out['terminal_reference_observed']
    assert len(out['nonunique_or_other_source_witnesses'])==1
    assert rays.valid.tolist()==[True]
