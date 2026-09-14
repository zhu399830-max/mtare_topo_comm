import numpy as np
import pytest
from mtare_topo.teacher import gse_terminal_reference_v2 as module
from test_gse_terminal_reference_v1 import fixture


def call(groups, primitives, rays, **kwargs):
    return module.terminal_reference_evidence(groups, primitives, rays=rays,
        membership_codes=np.array([1],dtype=np.uint16),source_sets=kwargs.pop('source_sets',[[],[0]]),
        source_directions=kwargs.pop('directions',rays.directions),**kwargs)


def test_exact_witness_preserves_frame_and_unknown_contract(monkeypatch):
    g,p,r=fixture()
    monkeypatch.setattr(module,'replay_cap',lambda *a,**k:dict(exact_cap_witnesses=[dict(ray_index=0,triangle_index=1,stored_t=2.)],backend_version='synthetic'))
    o=call(g,p,r)['terminal_references'][0]
    assert o['terminal_reference_observed']
    assert o['accepted_witnesses'][0]['source_frame_index']==4
    assert o['semantic_label'] is None and not o['complete_region']


def test_no_backend_match_is_unknown(monkeypatch):
    g,p,r=fixture()
    monkeypatch.setattr(module,'replay_cap',lambda *a,**k:dict(exact_cap_witnesses=[],backend_version='synthetic'))
    assert not call(g,p,r)['terminal_references'][0]['terminal_reference_observed']


def test_wrong_source_direction_rejected_before_backend():
    g,p,r=fixture()
    with pytest.raises(ValueError,match='normalization'):
        call(g,p,r,directions=np.array([[0.,1.,0.]]))


def test_degree_two_does_not_invoke_backend(monkeypatch):
    g,p,r=fixture();g[0]['paths'].append(dict(endpoint_key=('p',0)))
    monkeypatch.setattr(module,'replay_cap',lambda *a,**k:pytest.fail('not a terminal'))
    assert call(g,p,r)['terminal_references']==[]


def test_multi_source_exact_hit_is_not_forced_to_one_endpoint(monkeypatch):
    from dataclasses import replace
    g,p,r=fixture();p.append(replace(p[0],primitive_id='q'))
    monkeypatch.setattr(module,'replay_cap',lambda *a,**k:dict(exact_cap_witnesses=[dict(ray_index=0,triangle_index=1,stored_t=2.)],backend_version='synthetic'))
    o=call(g,p,r,source_sets=[[],[0,1]])['terminal_references'][0]
    assert not o['terminal_reference_observed']
    assert len(o['nonunique_or_other_source_witnesses'])==1


def test_denied_source_preserves_valid_return_but_not_terminal(monkeypatch):
    g,p,r=fixture()
    monkeypatch.setattr(module,'replay_cap',lambda *a,**k:dict(exact_cap_witnesses=[dict(ray_index=0,triangle_index=1,stored_t=2.)],backend_version='synthetic'))
    o=call(g,p,r,source_permission=np.array([False]))['terminal_references'][0]
    assert not o['terminal_reference_observed'] and not o['accepted_witnesses']
    assert len(o['nonunique_or_other_source_witnesses'])==1
    assert r.valid.tolist()==[True] and r.first_return_m.tolist()==[2.]


def test_permission_shape_rejected_before_geometry(monkeypatch):
    g,p,r=fixture()
    monkeypatch.setattr(module,'replay_cap',lambda *a,**k:pytest.fail('invalid mask reached backend'))
    with pytest.raises(ValueError,match='permission'):
        call(g,p,r,source_permission=np.array([False,True]))
