"""Source settings are mandatory and shared by both evidence branches."""
import inspect

import pytest

from mtare_topo.teacher import gse_reference_anchor_targets_v3 as anchor
from mtare_topo.teacher import gse_joint_reference_targets_v8 as joint
from mtare_topo.teacher.gse_interior_branch_evidence_v1 import bound_interior_evidence
from mtare_topo.teacher.gse_lateral_branch_evidence_v1 import bound_lateral_evidence


@pytest.mark.parametrize('producer', [anchor.produce_junction_reference_targets,
                                    joint.produce_joint_reference_targets,
                                    bound_lateral_evidence])
def test_new_producers_require_explicit_field_settings(producer):
    with pytest.raises(TypeError, match='field_spacing_m'):
        producer({}, {}, axial_spacing_m=.05, angular_segments=64)


def test_legacy_interior_default_unchanged():
    assert inspect.signature(bound_interior_evidence).parameters['field_spacing_m'].default == .01


def test_anchor_forwards_original_spacing_to_both_branches(monkeypatch):
    calls = []

    def interior(b, r, **settings):
        calls.append(('interior', settings))
        return {'interior': True}

    def lateral(b, r, **settings):
        calls.append(('lateral', settings))
        return {'ambiguous_ray_indices': []}

    monkeypatch.setattr(anchor, 'bound_interior_evidence', interior)
    monkeypatch.setattr(anchor, 'bound_lateral_evidence', lateral)
    monkeypatch.setattr(anchor, '_produce_junction_reference_targets',
                        lambda b, r, i, additional: {})
    anchor.produce_junction_reference_targets({}, {}, axial_spacing_m=.05,
        angular_segments=64, field_spacing_m=.025)
    assert calls == [('interior', {'field_spacing_m': .025}),
                     ('lateral', {'axial_spacing_m': .05, 'angular_segments': 64,
                                  'field_spacing_m': .025})]


@pytest.mark.parametrize('guard',[False,True])
def test_joint_forwards_and_records_settings_without_changing_record(monkeypatch,guard):
    calls = []

    def fake_anchor(b, r, **settings):
        calls.append(settings)

    def base(b, r, *, anchor_producer, **kwargs):
        anchor_producer(b, r)
        return {'record': {'fixed': True}}

    monkeypatch.setattr(joint, 'produce_junction_reference_targets', fake_anchor)
    monkeypatch.setattr(joint, '_produce_joint_v2', base)
    monkeypatch.setattr(joint, '_add_interior_correspondence', lambda *a, **k: None)
    terminal_calls = []
    def terminal(b, r, producer, **settings):
        terminal_calls.append(settings)
        return producer(b, r)
    monkeypatch.setattr(joint, '_produce_terminal_joint', terminal)
    monkeypatch.setattr(joint, '_produce_terminal_exclusion', terminal)
    result = joint.produce_joint_reference_targets({}, {}, axial_spacing_m=.05,
        angular_segments=64, field_spacing_m=.025,qualify_cap_precision=guard)
    anchor_options={'qualify_cap_precision':True} if guard else {}
    terminal_options={'cap_source_settings':dict(axial_spacing_m=.05,angular_segments=64)} if guard else {}
    assert calls == [{'axial_spacing_m': .05, 'angular_segments': 64, 'field_spacing_m': .025,**anchor_options}]
    assert terminal_calls == [dict(departure_field='surface_departure_witness_ray_indices',field_spacing_m=.025,**terminal_options),
                              dict(field_spacing_m=.025)]
    assert result['record'] == {'fixed': True}
    assert result['geometry_evidence_settings']['field_spacing_m'] == .025
    assert result['geometry_evidence_settings']['recast_first_returns'] is False


def test_every_field_constructor_in_v8_chain_has_explicit_spacing():
    import ast
    from pathlib import Path
    root = Path(joint.__file__).parent
    modules = ['gse_joint_reference_targets_v5.py', 'gse_joint_reference_targets_v7.py',
               'gse_interior_branch_evidence_v1.py', 'gse_lateral_branch_evidence_v1.py']
    constructors = 0
    for name in modules:
        tree = ast.parse((root/name).read_text())
        for call in ast.walk(tree):
            if isinstance(call,ast.Call) and isinstance(call.func,ast.Name) and call.func.id == 'SweptSuperellipseProvenanceField':
                constructors += 1
                values = [k.value for k in call.keywords if k.arg == 'spacing_m']
                assert len(values) == 1, name
                assert isinstance(values[0],ast.Name) and values[0].id == 'field_spacing_m', name
    assert constructors == 4
