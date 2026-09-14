"""Synthetic wiring only: bound evidence producers are separately tested."""
from copy import deepcopy
import numpy as np
import pytest
from mtare_topo.teacher import gse_joint_reference_targets_v1 as module
from mtare_topo.teacher.gse_surface_target_adapter_v1 import observed_targets


def fixture(monkeypatch):
    record = dict(schema='gse_surface_observed_targets_v1', coordinate_frame='current_sensor_m',
        source_frame_indices=[1,2,3,4,5], anchors=[], openings=[], membership=[],
        score_region=dict(center_m=[0.,0.,0.], radius_m=10., anchors_complete=False,
            openings_complete=False, evidence='partial only'))
    anchor = dict(record=deepcopy(record), unknown_candidates=[], teacher_provenance=[
        dict(node_id_teacher_only='n', interface_ids=[0], witness_ray_indices=[[3]])])
    anchor['record']['anchors'] = [dict(position_m=[2.,0.,0.], evidence='partial')]
    opening = dict(record=deepcopy(record), unknown_candidates=[], teacher_provenance=[
        dict(primitive_id_teacher_only='p', crossing_ray_indices=[3])])
    opening['record']['openings'] = [dict(position_m=[10.,0.,0.], direction=[1.,0.,0.],
        width_m=None, height_m=None, evidence='partial')]
    raw = dict(interfaces_teacher_only=[dict(interface_id_teacher_only=0, node_id_teacher_only='n')],
        raw_interface_intersections=[dict(ray_index=3, inside_roi=True, intersection_world_m=[2.,0.,0.],
            interface_id_teacher_only=0, source_key_teacher_only='p')])
    bundle = dict(sensor_teacher_only=dict(sensor_xyz_m=np.zeros((5,3)), yaw_deg=np.zeros(5)))
    monkeypatch.setattr(module, 'produce_junction_reference_targets', lambda b,r: deepcopy(anchor))
    monkeypatch.setattr(module, 'produce_opening_reference_targets', lambda b: deepcopy(opening))
    monkeypatch.setattr(module, 'produce_terminal_reference_targets', lambda b:
        dict(record=deepcopy(record), teacher_provenance=[], unknown_candidates=[]))
    return bundle, raw, anchor, opening


def test_shared_direct_witness_reaches_loss_mask(monkeypatch):
    bundle, raw, _, _ = fixture(monkeypatch)
    result = module.produce_joint_reference_targets(bundle, raw)
    assert result['record']['membership'] == [[True]]
    targets = observed_targets([result['record']])
    assert targets.membership_valid.tolist() == [[[True]]]
    assert not result['full_training_gate_eligible']
    assert 'node_id_teacher_only' not in str(result['record'])


@pytest.mark.parametrize('change', ['different_ray', 'different_source', 'after_opening'])
def test_source_identity_alone_cannot_create_membership(monkeypatch, change):
    bundle, raw, _, _ = fixture(monkeypatch)
    hit = raw['raw_interface_intersections'][0]
    if change == 'different_ray': hit['ray_index'] = 4
    if change == 'different_source': hit['source_key_teacher_only'] = 'other'
    if change == 'after_opening': hit['intersection_world_m'] = [11.,0.,0.]
    assert module.produce_joint_reference_targets(bundle, raw)['record']['membership'] == [[None]]


def test_unlabelled_intermediate_node_still_blocks_merge(monkeypatch):
    bundle, raw, _, _ = fixture(monkeypatch)
    raw['interfaces_teacher_only'].append(dict(interface_id_teacher_only=1, node_id_teacher_only='unlabelled'))
    raw['raw_interface_intersections'].append(dict(raw['raw_interface_intersections'][0],
        interface_id_teacher_only=1, intersection_world_m=[5.,0.,0.]))
    result = module.produce_joint_reference_targets(bundle, raw)
    assert result['record']['membership'] == [[None]]
    assert not observed_targets([result['record']]).membership_valid.any()


def test_duplicate_triangle_hits_do_not_change_correspondence(monkeypatch):
    bundle, raw, _, _ = fixture(monkeypatch)
    before = module.produce_joint_reference_targets(bundle, raw)
    raw['raw_interface_intersections'] *= 2
    assert module.produce_joint_reference_targets(bundle, raw) == before


def test_conflicting_rays_cannot_select_favourable_anchor(monkeypatch):
    bundle, raw, anchor, opening = fixture(monkeypatch)
    opening['teacher_provenance'][0]['crossing_ray_indices'].append(4)
    anchor['record']['anchors'].append(dict(position_m=[3.,0.,0.], evidence='partial'))
    anchor['teacher_provenance'].append(dict(node_id_teacher_only='second',
        interface_ids=[1], witness_ray_indices=[[4]]))
    raw['interfaces_teacher_only'].append(dict(interface_id_teacher_only=1, node_id_teacher_only='second'))
    raw['raw_interface_intersections'].append(dict(raw['raw_interface_intersections'][0],
        ray_index=4, interface_id_teacher_only=1, intersection_world_m=[3.,0.,0.]))
    assert module.produce_joint_reference_targets(bundle, raw)['record']['membership'] == [[None, None]]


def test_unmatched_competing_ray_not_discarded(monkeypatch):
    bundle, raw, _, opening = fixture(monkeypatch)
    opening['teacher_provenance'][0]['crossing_ray_indices'].append(4)
    assert module.produce_joint_reference_targets(bundle, raw)['record']['membership'] == [[None]]


def test_terminal_appended_without_inventing_opening_correspondence(monkeypatch):
    bundle, raw, anchor, _ = fixture(monkeypatch)
    terminal = deepcopy(anchor)
    terminal['record']['anchors'][0]['position_m'] = [-3.,0.,0.]
    monkeypatch.setattr(module, 'produce_terminal_reference_targets', lambda b: terminal)
    result = module.produce_joint_reference_targets(bundle, raw)
    assert result['record']['membership'] == [[True, None]]
    assert observed_targets([result['record']]).anchor_valid.tolist() == [[True, True]]
    assert result['teacher_provenance']['terminal_anchor_start'] == 1


def test_interior_anchor_witness_does_not_bypass_membership_entry_contract(monkeypatch):
    bundle, raw, anchor, _ = fixture(monkeypatch)
    anchor['teacher_provenance'][0]['entering_witness_ray_indices'] = [[]]
    anchor['teacher_provenance'][0]['interior_witness_ray_indices'] = [[3]]
    assert module.produce_joint_reference_targets(bundle,raw)['record']['membership'] == [[None]]
