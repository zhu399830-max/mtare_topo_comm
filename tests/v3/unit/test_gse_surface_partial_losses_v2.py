from copy import deepcopy
from dataclasses import replace,fields
import torch
from test_gse_reference_exclusion_binding_v1 import fixture
from test_gse_surface_losses_v1 import prediction
from test_gse_surface_target_adapter_v1 import record
from mtare_topo.data.gse_structure_review_v1 import canonical_sha
from mtare_topo.representation.gse_surface_partial_losses_v1 import bound_partial_surface_losses as old
from mtare_topo.representation.gse_surface_partial_losses_v2 import bound_partial_surface_losses as new


def setup(confirmed_both=True):
    b,g,_,binding=fixture();p=prediction()
    p=replace(p,**{f.name:getattr(p,f.name).float().detach().requires_grad_()
        for f in fields(p) if getattr(p,f.name).dtype==torch.float64})
    positions=torch.full_like(p.anchor_position_m,20.)
    positions[0,0]=torch.tensor([0.,1.,0.]);positions[0,1]=torch.tensor([0.,1.1,0.])
    positions[0,2]=torch.tensor([2.,0.,0.])
    p=replace(p,anchor_position_m=positions.requires_grad_())
    row=record();row['source_frame_indices']=[0,1,2,3,4]
    row['anchors']=[dict(position_m=[0.,1.,0.],evidence='synthetic')]
    if confirmed_both: row['anchors'].append(dict(position_m=[2.,0.,0.],evidence='synthetic'))
    row['openings']=[];row['membership']=[]
    manifest=dict(source_binding=binding,target_record_sha256=canonical_sha(row))
    target=dict(record=row,**deepcopy(manifest))
    return p,dict(produced_targets=[target],manifest_rows=[manifest],bundles=[b],grids=[g])


def test_duplicate_gradient_added_once_positive_and_unknown_preserved():
    p,kw=setup();prior,_=old(p,**kw)
    assert prior.denominators['anchor_presence']==2
    r,e=new(p,**kw)
    assert r.denominators['anchor_presence']==3
    assert e[0]['confirmed_structure_duplicate_evidence_v2']['added_duplicate_negative_query_indices']==[1]
    r.total.backward();grad=p.anchor_presence_logits.grad[0]
    assert grad[0]<0 and grad[2]<0 and grad[1]>0
    assert torch.count_nonzero(grad)==3
    assert r.assignments['anchor'].tolist()==[[0,2]]


def test_unknown_competitor_prevents_duplicate_gradient():
    p,kw=setup(False);r,e=new(p,**kw);r.total.backward()
    assert e[0]['confirmed_structure_duplicate_evidence_v2']['added_duplicate_negative_query_indices']==[]
    assert p.anchor_presence_logits.grad[0,1]==0
    assert r.denominators['anchor_presence']==1


def test_unsupported_observation_gets_no_duplicate_supervision():
    import pytest
    p,kw=setup();p=replace(p,observation_supported=torch.zeros(1,dtype=torch.bool))
    with pytest.raises(ValueError,match='without observation support'):
        new(p,**kw)
    row=kw['produced_targets'][0]['record'];row['anchors']=[]
    sha=canonical_sha(row)
    kw['produced_targets'][0]['target_record_sha256']=sha
    kw['manifest_rows'][0]['target_record_sha256']=sha
    r,e=new(p,**kw)
    assert e[0]['confirmed_structure_duplicate_evidence_v2']['added_duplicate_negative_query_indices']==[]
    assert r.denominators['anchor_presence']==0


def opening_setup():
    """Synthetic serialization/binding fixture, not physical ray correctness."""
    import json
    import numpy as np
    from mtare_topo.teacher.primitive_construction_supervisor import (
        PrimitiveEndpoint,SweptPrimitive,PrimitiveConstructionGraph,EndpointComposition)
    p,kw=setup();b=kw['bundles'][0]
    ends=tuple(PrimitiveEndpoint('p',i,n,xyz,xyz) for i,n,xyz in (
        (0,'a',(0.,0.,0.)),(1,'b',(12.,0.,0.))))
    primitive=SweptPrimitive('p','edge','tunnel',np.array([[0.,0.,0.],[12.,0.,0.]]),2.,(1.,0.),ends)
    graph=PrimitiveConstructionGraph('cano_world',(primitive,),tuple(
        EndpointComposition(e.node_id,(e,),e.composition_anchor_xyz_m) for e in ends),
        endpoint_attachment_mode='free_space_overlap',node_degree_source='edge_incidence')
    doc=json.loads(json.dumps(dict(schema_version='primitive_relation_realized_construction_v1',
        parent_id='synthetic',geometry_realization='ellipse',base_construction=graph.as_dict(),
        realized_primitives=[dict(primitive_id='p',centerline_xyz_m=[[0,0,0],[12,0,0]],
            endpoint_half_axes_m=[[2,2],[2,2]],endpoint_shape_exponent=[2,2])])) )
    b['construction_teacher_only']=doc
    row=record();row['source_frame_indices']=[0,1,2,3,4];row['anchors']=[]
    row['openings']=[dict(position_m=[10.,0.,0.],direction=None,width_m=None,height_m=None,evidence='synthetic')]
    row['membership']=[[]]
    manifest=dict(source_binding=dict(source=deepcopy(b['source']),construction_sha256=canonical_sha(doc)),
        target_record_sha256=canonical_sha(row))
    kw.update(produced_targets=[dict(record=row,**deepcopy(manifest))],manifest_rows=[manifest],opening_matching_radius_m=1.)
    pos=torch.full_like(p.opening_position_m,20.);pos[0,0]=torch.tensor([10.,0.,0.]);pos[0,1]=torch.tensor([9.9,0.,0.])
    return replace(p,opening_position_m=pos.requires_grad_()),kw


def test_opening_duplicate_uses_independent_reference_and_explicit_radius():
    p,kw=opening_setup();prior,_=old(p,**kw);r,e=new(p,**kw)
    assert prior.denominators['opening_presence']==1
    assert r.denominators['opening_presence']==2
    assert e[0]['opening_exclusion']['confirmed_structure_duplicate_evidence_v2']['reference_kind']=='openings'
    assert e[0]['opening_exclusion']['confirmed_structure_duplicate_evidence_v2']['added_duplicate_negative_query_indices']==[1]
    r.total.backward()
    assert p.opening_presence_logits.grad[0,0]<0 and p.opening_presence_logits.grad[0,1]>0
    assert torch.count_nonzero(p.opening_presence_logits.grad)==2


def test_mixed_batch_empty_observation_does_not_change_denominator_or_gradient():
    p,kw=opening_setup()
    p=replace(p,**{f.name:torch.cat([getattr(p,f.name).detach()]*2).requires_grad_(getattr(p,f.name).requires_grad)
        for f in fields(p)})
    p.observation_supported[1]=False
    empty=deepcopy(kw['produced_targets'][0]);empty['record']['openings']=[];empty['record']['membership']=[]
    empty['target_record_sha256']=canonical_sha(empty['record'])
    empty_manifest={k:deepcopy(empty[k]) for k in ('source_binding','target_record_sha256')}
    kw['produced_targets'].append(empty);kw['manifest_rows'].append(empty_manifest)
    kw['bundles']*=2;kw['grids']*=2
    r,e=new(p,**kw);r.total.backward()
    assert r.denominators['opening_presence']==2
    assert torch.count_nonzero(p.opening_presence_logits.grad[1])==0
    assert e[1]['opening_exclusion']['confirmed_structure_duplicate_evidence_v2']['added_duplicate_negative_query_indices']==[]
