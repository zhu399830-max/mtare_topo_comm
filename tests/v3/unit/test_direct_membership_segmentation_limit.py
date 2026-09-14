"""Document current direct-source limitation; these tests do not fix the teacher."""
import numpy as np
from mtare_topo.teacher.gse_joint_reference_targets_v2 import _add_interior_correspondence


def fixture():
    result = {
        'record': {'openings': [{'position_m': [5., 0., 0.], 'direction': [1., 0., 0.]}],
                   'anchors': [{'position_m': [0., 0., 0.]}], 'membership': [[None]]},
        'teacher_provenance': {
            'openings': [{'primitive_id_teacher_only': 'branch', 'crossing_ray_indices': [0]}],
            'anchors': [{'node_id_teacher_only': 'junction', 'interface_ids': [0],
                         'surface_entry_witness_ray_indices': [[0]]}],
            'relations': [{'status': 'UNKNOWN'}]}}
    bundle = {'sensor_teacher_only': {'sensor_xyz_m': np.zeros((5, 3)), 'yaw_deg': np.zeros(5)}}
    raw = {'interfaces_teacher_only': [{'interface_id_teacher_only': 0,
            'endpoint_key_teacher_only': ['branch', 0], 'node_id_teacher_only': 'junction'}],
           'raw_interface_intersections': []}
    return result, bundle, raw


def apply(result, bundle, raw):
    _add_interior_correspondence(result, bundle, raw,
                                witness_field='surface_entry_witness_ray_indices')
    return result['record']['membership'][0][0]


def test_direct_source_correspondence_supported():
    result, bundle, raw = fixture()
    assert apply(result, bundle, raw) is True


def test_same_ray_on_nonincident_segment_is_not_propagated():
    result, bundle, raw = fixture()
    # Source segmentation is the only change to this post-processing input.
    # It has no API for an observed chain through a degree-two node.
    result['teacher_provenance']['openings'][0]['primitive_id_teacher_only'] = 'earlier_segment'
    assert apply(result, bundle, raw) is None
    assert result['teacher_provenance']['relations'][0]['status'] == 'UNKNOWN'


def test_other_node_veto_is_not_silently_removed():
    result, bundle, raw = fixture()
    raw['interfaces_teacher_only'].append({'interface_id_teacher_only': 1,
        'endpoint_key_teacher_only': ['earlier_segment', 1], 'node_id_teacher_only': 'intermediate'})
    raw['raw_interface_intersections'] = [{'inside_roi': True, 'ray_index': 0,
        'intersection_world_m': [1., 0., 0.], 'interface_id_teacher_only': 1}]
    assert apply(result, bundle, raw) is None
    assert result['teacher_provenance']['relations'][0]['status'] == 'UNKNOWN_COMPETING_INTERIOR_CORRESPONDENCE'
