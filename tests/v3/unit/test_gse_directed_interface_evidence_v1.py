import copy
import numpy as np
from mtare_topo.teacher.gse_directed_interface_evidence_v1 import directed_evidence


def fixture():
    interfaces=[dict(interface_id_teacher_only=i,node_id_teacher_only='node',
        source_key_teacher_only='same',inward_direction=[sign,0.,0.]) for i,sign in enumerate((1.,-1.))]
    records=[dict(ray_index=0,interface_id_teacher_only=i,t=1.,inside_roi=True) for i in range(2)]
    args=dict(directions=np.array([[1.,0.,0.]]),first_return=np.array([3.],dtype=np.float32),
              valid=np.array([True]),return_sources=[['same']])
    return records,interfaces,args


def test_coincident_opposite_arcs_not_collapsed_or_ordered():
    records,interfaces,args=fixture();r=directed_evidence(records,interfaces,**args)
    assert r['interfaces'][0]['entering_ray_indices']==[0]
    assert r['interfaces'][1]['leaving_ray_indices']==[0]
    assert r==directed_evidence(records[::-1],interfaces,**args)
    assert r['qualified_labels']==0 and not r['complete_region']


def test_same_inward_source_ambiguous_not_two_branches():
    records,interfaces,args=fixture();interfaces[1]['inward_direction']=[1.,0.,0.]
    r=directed_evidence(records,interfaces,**args)
    assert all(v['entering_ray_indices']==[] and v['unknown_ray_indices']==[0] for v in r['interfaces'].values())


def test_duplicate_triangle_does_not_duplicate_ray():
    records,interfaces,args=fixture()
    assert directed_evidence(records,interfaces,**args)==directed_evidence(records+copy.deepcopy(records),interfaces,**args)


def test_occlusion_and_multisource_are_unknown():
    records,interfaces,args=fixture();args['first_return'][0]=.5
    r=directed_evidence(records,interfaces,**args)
    assert all(v['unknown_ray_indices']==[0] for v in r['interfaces'].values())
    args['first_return'][0]=3.;args['return_sources']=[['same','other']]
    assert directed_evidence(records,interfaces,**args)['interfaces'][0]['unknown_ray_indices']==[0]
