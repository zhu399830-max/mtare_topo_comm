"""Fixed synthetic supervision matrix; declarations only, no data reads/render.

Reference inventory is not visibility truth. Observation-dependent expectations
must be checked independently, never filled from the target producer output.
"""
from copy import deepcopy
import numpy as np

SECTIONS={'circle':((2.,2.),2.),'ellipse':((2.,1.5),2.),'rounded_rectangle':((2.,1.5),8.)}
TYPES=('straight','terminal','T','Y','four_way','double_junction','parallel',
       'stacked','ramp_connection','visible_blocker','hidden_branch','overlap_ambiguity')


def program(kind):
    if kind not in TYPES:raise ValueError('unregistered synthetic type')
    edges=[];anchors={}
    def edge(key,points,left,right):
        edges.append(dict(id=key,points=points,nodes=[left,right]))
    def star(prefix='',offset=(0.,0.,0.),directions=((1.,0.,0.),(-1.,0.,0.),(0.,-1.,0.))):
        center=np.asarray(offset);node=prefix+'J';anchors[node]=center.tolist()
        for i,d in enumerate(directions):
            d=np.asarray(d);end=prefix+'end'+str(i)
            points=[(center-d).tolist(),(center+30*d).tolist()]
            anchors[end]=points[-1];edge(prefix+'p'+str(i),points,node,end)
    if kind in ('T','Y','four_way','overlap_ambiguity'):
        directions=((1.,0.,0.),(-1.,0.,0.),(0.,-1.,0.))
        if kind=='Y':directions=((1.,0.,0.),(-.5,np.sqrt(3)/2,0.),(-.5,-np.sqrt(3)/2,0.))
        if kind=='four_way':directions=(*directions,(0.,1.,0.))
        star(directions=directions)
        if kind=='overlap_ambiguity':star('other_',(.5,0.,0.))
    elif kind=='double_junction':
        anchors.update(L=[-30.,0.,0.],J0=[-4.,0.,0.],J1=[4.,0.,0.],R=[30.,0.,0.],B0=[-4.,-30.,0.],B1=[4.,-30.,0.])
        edge('left',[anchors['L'],[-3.,0.,0.]],'L','J0')
        edge('middle',[[-5.,0.,0.],[5.,0.,0.]],'J0','J1')
        edge('right',[[3.,0.,0.],anchors['R']],'J1','R')
        for i in (0,1):edge('branch'+str(i),[[-4.+8*i,1.,0.],anchors['B'+str(i)]],'J'+str(i),'B'+str(i))
    elif kind=='hidden_branch':
        anchors.update(L=[-30.,0.,0.],J=[8.,16.,0.],R=[8.,40.,0.],B=[30.,16.,0.])
        edge('approach',[anchors['L'],[8.,0.,0.],[8.,17.,0.]],'L','J')
        edge('continuation',[[8.,15.,0.],anchors['R']],'J','R')
        edge('hidden',[[7.,16.,0.],anchors['B']],'J','B')
    else:
        end=6. if kind in ('terminal','visible_blocker') else 30.
        anchors.update(L=[-30.,0.,0.],R=[end,0.,0.])
        points=[anchors['L'],anchors['R']]
        if kind=='ramp_connection':
            anchors['R']=[30.,0.,3.];points=[anchors['L'],[0.,0.,0.],[12.,0.,3.],anchors['R']]
        edge('main',points,'L','R')
        if kind in ('parallel','stacked'):
            delta=np.array([0.,6.,0.] if kind=='parallel' else [0.,0.,6.])
            anchors.update(other_L=(np.array(anchors['L'])+delta).tolist(),other_R=(np.array(anchors['R'])+delta).tolist())
            edge('other',[anchors['other_L'],anchors['other_R']],'other_L','other_R')
        if kind=='visible_blocker':
            anchors.update(behind_L=[6.5,0.,0.],behind_R=[30.,0.,0.])
            edge('behind_wall',[anchors['behind_L'],anchors['behind_R']],'behind_L','behind_R')
    return dict(type=kind,anchors=anchors,edges=edges)


def matrix():
    cases=[]
    for kind in TYPES:
        geom=program(kind)
        degree={n:sum(n in e['nodes'] for e in geom['edges']) for n in geom['anchors']}
        for section,(axes,exponent) in SECTIONS.items():
            for view,distance in enumerate((-4.,-2.,0.,2.)):
                # Y approach uses its +X arm; other prototypes use the X trunk.
                current=-distance if kind=='Y' else distance
                xs=[current-.4+.1*i for i in range(5)]
                if kind=='Y':xs=[current+.4-.1*i for i in range(5)]
                poses=[[x,0.,.04+(max(0.,x)*.25 if kind=='ramp_connection' else 0.)] for x in xs]
                # At the negative Y view, use its negative-X ray rather than
                # silently placing the sensor outside all three Y arms.
                if kind=='Y' and current<0:
                    poses=[[x*.5,-x*np.sqrt(3)/2,.04] for x in xs]
                origin=np.array(poses[-1])
                inventory=[dict(node=n,position_m=p,degree=degree[n]) for n,p in geom['anchors'].items()
                    if degree[n]!=2 and np.linalg.norm(np.array(p)-origin)<10.]
                cases.append(dict(case_id=f'{kind}__{section}__view{view}',program=deepcopy(geom),section=section,
                    half_axes_m=list(axes),shape_exponent=exponent,poses_world_m=poses,yaw_deg=[0.]*5,
                    reference_inventory=inventory,
                    expected=dict(no_anchors_outside_reference_inventory=True,
                        no_full_background_promotion=True,no_physical_reachability_claim=True,
                        overlap_must_not_force_identity=kind=='overlap_ambiguity',
                        hidden_branch_not_visible_positive=kind=='hidden_branch',
                        observed_target_counts='NOT_ASSUMED_FROM_REFERENCE_INVENTORY'),
                    paired_control='remove_hidden_edge_same_observation_contract' if kind=='hidden_branch' else None))
    return cases


def construction_document(case, *, hidden_control=False):
    from mtare_topo.teacher.primitive_construction_supervisor import (
        PrimitiveEndpoint, SweptPrimitive, EndpointComposition, PrimitiveConstructionGraph)
    geom=deepcopy(case['program'])
    if hidden_control:
        if geom['type']!='hidden_branch':raise ValueError('only registered hidden-branch control allowed')
        geom['edges']=[e for e in geom['edges'] if e['id']!='hidden']
        del geom['anchors']['B']
    memberships={n:[] for n in geom['anchors']};primitives=[];realized=[]
    for e in geom['edges']:
        points=np.asarray(e['points'],dtype=np.float64)
        endpoints=tuple(PrimitiveEndpoint(e['id'],side,node,tuple(points[0 if side==0 else -1]),tuple(geom['anchors'][node]))
                        for side,node in enumerate(e['nodes']))
        for endpoint in endpoints:memberships[endpoint.node_id].append(endpoint)
        errors=tuple(float(np.linalg.norm(np.array(p.xyz_m)-np.array(p.composition_anchor_xyz_m))) for p in endpoints)
        primitives.append(SweptPrimitive(e['id'],e['id'],e['id'],points,max(case['half_axes_m']),errors,endpoints))
        realized.append(dict(primitive_id=e['id'],centerline_xyz_m=points.tolist(),
            endpoint_half_axes_m=[case['half_axes_m'],case['half_axes_m']],
            endpoint_shape_exponent=[case['shape_exponent']]*2))
    graph=PrimitiveConstructionGraph('cano_world',tuple(primitives),tuple(
        EndpointComposition(n,tuple(rows),tuple(geom['anchors'][n])) for n,rows in memberships.items()),
        endpoint_attachment_mode='free_space_overlap',node_degree_source='edge_incidence')
    return dict(schema_version='primitive_relation_realized_construction_v1',parent_id='synthetic_'+geom['type'],
        geometry_realization=case['section'],base_construction=graph.as_dict(),realized_primitives=realized)
