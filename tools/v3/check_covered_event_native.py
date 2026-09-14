"""Fixed prior failure and native contact/gap checks; no sensor export."""
import json
import numpy as np
from check_interval_v2_native import box
from mtare_topo.data.gse_synthetic_matrix import matrix, construction_document
from mtare_topo.data.cano_sensor_smoke import lidar_local_directions, world_directions
from mtare_topo.data.primitive_relation_materialization import load_p1a_realized_construction
from mtare_topo.teacher.csg_mesh_provenance import mesh_swept_superellipse
from mtare_topo.teacher.covered_event_diagnostic import CoveredEventDiagnostic
from mtare_topo.teacher.exact_reference_diagnostic import ExactReferenceDiagnostic
from mtare_topo.evaluation.gse_convex_ray_union import declared_union_exit


def main():
    case=next(c for c in matrix() if c['case_id']=='double_junction__circle__view1')
    origin=np.asarray(case['poses_world_m'][4])
    direction=world_directions(lidar_local_directions().reshape(-1,3).astype(np.float64),case['yaw_deg'][4])[7470]
    _,primitives=load_p1a_realized_construction(construction_document(case))
    meshes=[mesh_swept_superellipse(p,axial_spacing_m=.05,angular_segments=64) for p in primitives]
    old=ExactReferenceDiagnostic(meshes).query(origin,direction)
    new=CoveredEventDiagnostic(meshes).query(origin,direction)
    unit=direction/np.linalg.norm(direction)
    expected,supported,_=declared_union_exit(case,origin[None],unit[None])
    print(json.dumps(dict(case=case['case_id'],frame=4,ray=7470,old=old,new=new,expected_m=float(expected[0]))),flush=True)
    assert old['status']=='needs_reference' and old['reason']=='coincident_entry_exit'
    assert supported.all() and new['status']=='candidate'
    assert abs(new['distance_m']-float(expected[0]))<=1e-5
    assert new['sources']==['left','middle']
    assert new['covered_mixed_events'][0]['persistent_operands']==[1]
    for name,gap,cover in [('contact',0.,False),('gap',.0001,False),('covered_contact',0.,True)]:
        boxes=[box('a',-1.,2.),box('b',2.+gap,4.)]
        if cover:boxes.append(box('cover',-1.,3.))
        result=CoveredEventDiagnostic(boxes).query([0.,.13,.17],[1.,0.,0.])
        print(json.dumps(dict(case=name,result=result)),flush=True)
        if name=='contact':
            assert result['status']=='needs_reference' and result['reason']=='coincident_entry_exit'
        else:
            assert result['status']=='candidate'
            assert abs(result['distance_m']-(4. if cover else 2.))<=1e-5
            assert result['sources']==(['b'] if cover else ['a'])
    print(json.dumps(dict(native_checks=4,status='PASS_NOT_FULL_SCAN',labels_exported=0,optimizer_steps=0)),flush=True)


if __name__=='__main__':main()
