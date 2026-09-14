"""Counterexamples using actual upstream affinity and existing teacher code.

No data export: constructed records are software fixtures, not training labels.
"""
import json
import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
import check_supercluster_affinity as upstream
from mtare_topo.data.primitive_relation_targets import primitive_relation_window_targets
from mtare_topo.teacher.primitive_construction_supervisor import (
    EndpointComposition,PrimitiveConstructionGraph,PrimitiveEndpoint)


def main():
    endpoint=lambda p,e,n:PrimitiveEndpoint(p,e,n,(float(e),0.,0.))
    graph=PrimitiveConstructionGraph(coordinate_frame='world',primitives=(),compositions=(
        EndpointComposition('left',(endpoint('a',1,'left'),endpoint('b',0,'left'))),
        EndpointComposition('right',(endpoint('b',1,'right'),endpoint('c',0,'right')))))
    frames=[[(0,),(1,),(2,)]]*5
    target=primitive_relation_window_targets(construction=graph,primitive_ids=('a','b','c'),
        source_sets_by_frame=frames,maximum_slots=3)
    assert target.endpoint_attachment[0,1,1,0]==1
    assert target.endpoint_attachment[1,1,2,0]==1
    assert target.endpoint_attachment[0,1,2,0]==0
    cls=upstream.resolve('src.data.instance','InstanceData')
    # Pure surface chunks a,b,c: object identity supervision says NOT same object,
    # although a-b and b-c have different physical endpoint attachments.
    obj=cls(torch.tensor([0,1,2,3]),torch.tensor([0,1,2]),torch.ones(3,dtype=torch.long),torch.zeros(3,dtype=torch.long))
    _,aff=obj.instance_graph(torch.tensor([[0,1],[1,2]]),num_classes=1,smooth_affinity=False)
    torch.testing.assert_close(aff,torch.zeros(2))
    # Two chunks of b are the same object, but b's endpoints attach to two places.
    same=cls(torch.tensor([0,1,2]),torch.tensor([1,1]),torch.ones(2,dtype=torch.long),torch.zeros(2,dtype=torch.long))
    _,same_aff=same.instance_graph(torch.tensor([[0],[1]]),num_classes=1,smooth_affinity=False)
    assert same_aff.item()==1
    # Identical source observations do not establish which endpoint is visible.
    hidden_graph=PrimitiveConstructionGraph(coordinate_frame='world',primitives=(),compositions=())
    hidden=primitive_relation_window_targets(construction=hidden_graph,primitive_ids=('a','b','c'),
        source_sets_by_frame=frames,maximum_slots=3)
    assert hidden.endpoint_attachment.sum()==0 and target.endpoint_attachment.sum()==4
    print(json.dumps(dict(status='TARGET_NON_EQUIVALENCE_COUNTEREXAMPLES_CONFIRMED',
        connected_tunnel_object_affinity=aff.tolist(),same_tunnel_two_ends_affinity=same_aff.item(),
        physical_directed_attachment_count=int(target.endpoint_attachment.sum()),
        same_source_different_hidden_construction_count=int(hidden.endpoint_attachment.sum()),
        conclusion='instance_equivalence_is_not_endpoint_incidence; visibility mask still required',
        training_targets_created=0,research_observations=0,optimizer_steps=0)))


if __name__=='__main__':main()
