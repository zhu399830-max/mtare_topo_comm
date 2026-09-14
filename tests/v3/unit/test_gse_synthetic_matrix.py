import collections
import numpy as np
from mtare_topo.data.gse_synthetic_matrix import matrix,TYPES,construction_document


def test_closed_144_matrix_and_reference_not_visibility_truth():
    rows=matrix();assert len(rows)==144 and len({r['case_id'] for r in rows})==144
    assert collections.Counter(r['program']['type'] for r in rows)==dict.fromkeys(TYPES,12)
    assert sum(r['paired_control'] is not None for r in rows)==12
    assert all(len(r['poses_world_m'])==5 for r in rows)
    assert all(r['expected']['observed_target_counts']=='NOT_ASSUMED_FROM_REFERENCE_INVENTORY' for r in rows)
    assert matrix()==rows


def test_declared_sensor_poses_inside_centerline_inscribed_tube():
    # Independent conservative containment: inscribed circular tube, not the
    # label producer or its source identity rules. No ray rendering needed.
    for row in matrix():
        for pose in np.array(row['poses_world_m']):
            distances=[]
            for edge in row['program']['edges']:
                points=np.array(edge['points'])
                for a,b in zip(points[:-1],points[1:]):
                    u=np.dot(pose-a,b-a)/np.dot(b-a,b-a)
                    if 0<=u<=1:distances.append(np.linalg.norm(pose-a-u*(b-a)))
            assert distances and min(distances)<min(row['half_axes_m']),row['case_id']


def test_no_false_structural_nodes_in_plain_corridors_or_hidden_junction():
    for row in matrix():
        if row['program']['type'] in ('straight','parallel','stacked','ramp_connection','hidden_branch'):
            assert not row['reference_inventory']


def test_all_constructions_decode_and_controls_preserve_observed_approach():
    from mtare_topo.data.primitive_relation_materialization import load_p1a_realized_construction
    for case in matrix():
        document=construction_document(case)
        _,primitives=load_p1a_realized_construction(document)
        assert len(primitives)==len(case['program']['edges'])
        if case['paired_control']:
            control=construction_document(case,hidden_control=True)
            assert control['realized_primitives']==[r for r in document['realized_primitives'] if r['primitive_id']!='hidden']
            _,other=load_p1a_realized_construction(control)
            assert len(other)==len(primitives)-1
