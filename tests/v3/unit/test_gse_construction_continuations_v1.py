import copy
import json
import numpy as np
import pytest
from mtare_topo.data.gse_structure_review_v1 import canonical_sha
from mtare_topo.teacher.gse_construction_continuations_v1 import construction_reference_continuations
from mtare_topo.teacher.primitive_construction_supervisor import (
    PrimitiveEndpoint,SweptPrimitive,EndpointComposition,PrimitiveConstructionGraph)


def document():
    primitives=[];members={};realized=[]
    for name,nodes,line in [('a',('start','split'),[[-2.,0.,0.],[0.,0.,0.]]),
                            ('b',('split','end'),[[0.,0.,0.],[10.,0.,0.]])]:
        ends=tuple(PrimitiveEndpoint(name,i,node,tuple(line[i]),tuple(line[i])) for i,node in enumerate(nodes))
        for e in ends:members.setdefault(e.node_id,[]).append(e)
        primitives.append(SweptPrimitive(name,name,'same_tunnel',np.array(line),1.,(1.,0.),ends))
        realized.append(dict(primitive_id=name,centerline_xyz_m=line,
            endpoint_half_axes_m=[[1.,1.],[1.,1.]],endpoint_shape_exponent=[2.,2.]))
    graph=PrimitiveConstructionGraph('cano_world',tuple(primitives),tuple(
        EndpointComposition(node,tuple(ends),ends[0].composition_anchor_xyz_m) for node,ends in members.items()),
        endpoint_attachment_mode='free_space_overlap',node_degree_source='edge_incidence')
    return json.loads(json.dumps(dict(schema_version='primitive_relation_realized_construction_v1',
        base_construction=graph.as_dict(),realized_primitives=realized)))


def test_complete_real_serializer_roundtrip_and_nonmutation():
    d=document();before=copy.deepcopy(d)
    r=construction_reference_continuations(d,expected_document_sha256=canonical_sha(d))
    assert d==before and r['endpoint_incidence_count']==4
    assert r['continuations'][0].source_ids==('a','b')
    assert not r['observation_support_verified'] and not r['physical_separation_certified']


def test_missing_endpoint_inventory_fails_even_if_remaining_groups_valid():
    d=document();d['base_construction']['composition_operations'].pop()
    with pytest.raises(ValueError,match='both original endpoint owners'):
        construction_reference_continuations(d,expected_document_sha256=canonical_sha(d))


def test_realized_displacement_not_repaired_using_base_axis_or_connector():
    d=document();d['realized_primitives'][1]['centerline_xyz_m'][0][0]=1e-14
    r=construction_reference_continuations(d,expected_document_sha256=canonical_sha(d))
    assert len(r['continuations'])==2
    assert all(x.unresolved_degree_two_nodes==('split',) for x in r['continuations'])


def test_pinned_document_drift_rejected():
    d=document();pin=canonical_sha(d);d['realized_primitives'][1]['centerline_xyz_m'][0][0]=1.
    with pytest.raises(ValueError,match='pinned'):
        construction_reference_continuations(d,expected_document_sha256=pin)


def test_source_order_changes_do_not_change_continuation():
    d=document();r=construction_reference_continuations(d,expected_document_sha256=canonical_sha(d))
    d['realized_primitives'].reverse();d['base_construction']['primitives'].reverse()
    q=construction_reference_continuations(d,expected_document_sha256=canonical_sha(d))
    assert r['continuations']==q['continuations']
