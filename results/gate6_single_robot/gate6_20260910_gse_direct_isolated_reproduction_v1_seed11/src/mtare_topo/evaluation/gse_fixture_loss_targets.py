"""Declared-prototype oracle for software loss checks, NEVER a real-data teacher.

No production target is accepted or rewritten. No membership, dimension or
reachability answers are invented from these fixture center/heading oracles.
"""
from mtare_topo.data.gse_synthetic_matrix import matrix
from mtare_topo.data.gse_structure_review_v1 import canonical_sha
from mtare_topo.evaluation.gse_synthetic_field_scoring import expected_geometry
from mtare_topo.teacher.gse_surface_target_adapter_v1 import observed_targets


def fixture_loss_targets(case, source_frame_indices, *, device='cpu'):
    registered = [c for c in matrix() if c['case_id'] == case.get('case_id')]
    if len(registered) != 1 or canonical_sha(case) != canonical_sha(registered[0]):
        raise ValueError('exact declared synthetic fixture required; no real-world inputs')
    truth = expected_geometry(registered[0])
    if not truth['complete_for_declared_fixture']:
        raise ValueError('fixture has no independent complete geometry oracle')
    evidence = 'Software-only declared fixture oracle: ' + truth['oracle']
    record = dict(schema='gse_surface_observed_targets_v1', coordinate_frame='current_sensor_m',
        source_frame_indices=source_frame_indices,
        anchors=[dict(position_m=p,evidence=evidence) for p in truth['anchors']],
        openings=[dict(position_m=list(map(float,p['position_m'])),
                       direction=list(map(float,p['direction'])),
                       width_m=None,height_m=None,evidence=evidence) for p in truth['openings']],
        membership=[[None for _ in truth['anchors']] for _ in truth['openings']],
        score_region=dict(center_m=[0.,0.,0.],radius_m=10.,anchors_complete=True,
                          openings_complete=True,evidence=evidence))
    return observed_targets([record],device=device)
