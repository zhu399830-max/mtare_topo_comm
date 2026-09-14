"""Independent declared-prototype controls, not a general real-world teacher."""
import numpy as np
from scipy.optimize import linear_sum_assignment
from .gse_synthetic_straight_oracle import visible_straight_cap_rays


def expected_geometry(case):
    kind=case['program']['type'];origin=np.asarray(case['poses_world_m'][-1])
    if any(case['yaw_deg']):raise ValueError('registered zero-yaw prototype controls required')
    radius=np.sqrt(100.-origin[2]**2)
    if kind in ('straight','terminal','visible_blocker'):
        a,b=np.asarray(case['program']['edges'][0]['points'])
        caps=visible_straight_cap_rays(case)
        anchors=[(p-origin).tolist() for side,p in enumerate((a,b)) if caps[side]]
        openings=[]
        for sign in (-1.,1.):
            x=origin[0]+sign*radius
            if a[0]<x<b[0]:openings.append(dict(position_m=[sign*radius,0.,-origin[2]],direction=[sign,0.,0.]))
        return dict(anchors=anchors,openings=openings,oracle='convex_straight_tube_analytic',complete_for_declared_fixture=True)
    if kind in ('T','Y','four_way') and case['case_id'].endswith('view2'):
        # Center view lies inside each incident convex branch; every branch
        # extends beyond the 10m window. No occluder or layered duplicate exists
        # in these three declared prototypes. Other viewpoints are NOT inferred.
        if not np.array_equal(origin[:2],np.zeros(2)):raise ValueError('exact central prototype view required')
        openings=[]
        for edge in case['program']['edges']:
            points=np.asarray(edge['points']);d=points[-1]/np.linalg.norm(points[-1])
            openings.append(dict(position_m=(radius*d-origin).tolist(),direction=d.tolist()))
        return dict(anchors=[(-origin).tolist()],openings=openings,
            oracle='declared_unoccluded_convex_star_center',complete_for_declared_fixture=True)
    return dict(anchors=None,openings=None,oracle='UNRESOLVED_OBSERVABILITY',complete_for_declared_fixture=False)


def match_positions(expected,actual,threshold):
    x=np.asarray(expected,dtype=float).reshape(-1,3);y=np.asarray(actual,dtype=float).reshape(-1,3)
    if not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('finite scoring geometry required')
    pairs=[]
    if len(x) and len(y):
        distances=np.linalg.norm(x[:,None,:]-y[None,:,:],axis=-1)
        # Maximum in-threshold cardinality first, distance second. No class or
        # confidence enters matching; a far assignment cannot steal a valid one.
        penalty=(max(len(x),len(y))+1)*(threshold+1)
        rows,cols=linear_sum_assignment(np.where(distances<=threshold,distances,penalty))
        pairs=[(int(i),int(j),float(distances[i,j])) for i,j in zip(rows,cols) if distances[i,j]<=threshold]
    tp=len(pairs);fp=len(y)-tp;fn=len(x)-tp
    return dict(tp=tp,fp=fp,fn=fn,f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
                exact_count_and_matching_pass=fp==fn==0,pairs=pairs)


def score_case(case,record):
    expected=expected_geometry(case)
    if record['coordinate_frame']!='current_sensor_m':raise ValueError('current sensor frame required')
    if record['score_region']['anchors_complete'] or record['score_region']['openings_complete']:
        raise ValueError('producer may not promote partial labels to complete real-world annotation')
    report=dict(case_id=case['case_id'],oracle=expected['oracle'],independent_oracle_covered=expected['complete_for_declared_fixture'],
        scientific_gate_pass=False,real_training_eligible=False)
    if expected['anchors'] is None:
        report.update(status='UNSCORED_NOT_PASS',anchors=None,openings=None)
        return report
    anchors=[a['position_m'] for a in record['anchors']];openings=[o['position_m'] for o in record['openings']]
    report['anchors']={str(t):match_positions(expected['anchors'],anchors,t) for t in (1.,2.,4.)}
    report['openings']=match_positions([x['position_m'] for x in expected['openings']],openings,1.)
    errors=[]
    for i,j,_ in report['openings']['pairs']:
        actual=record['openings'][j]['direction'];truth=np.asarray(expected['openings'][i]['direction'])
        if actual is None:errors.append(None);continue
        actual=np.asarray(actual,dtype=float)
        if not np.isfinite(actual).all() or np.linalg.norm(actual)==0:raise ValueError('invalid opening direction')
        errors.append(float(np.rad2deg(np.arccos(np.clip(np.dot(actual/np.linalg.norm(actual),truth),-1,1)))))
    report['opening_direction_errors_deg']=errors
    passed=report['anchors']['4.0']['exact_count_and_matching_pass'] and report['openings']['exact_count_and_matching_pass']
    report['status']='FIXTURE_GEOMETRY_PASS' if passed else 'FIXTURE_GEOMETRY_FAIL'
    return report
