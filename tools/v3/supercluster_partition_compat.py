"""Explicit local compatibility changes, not an unmodified official entrypoint.

Keep the upstream objective/native solver. Change only singleton-logit shape
and the documented native edge-list memory representation. Solve disconnected
components separately, retaining isolated thing nodes. Stuff-class global
merging is deliberately unsupported. No edges are added or reweighted.
"""
import ast
import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from grid_graph import edge_list_to_forward_star
import check_supercluster_affinity as upstream


def forward_star(num_nodes, edges):
    edges=np.asarray(edges)
    if edges.ndim!=2 or edges.shape[1]!=2:
        raise ValueError('expected upstream E-by-2 edge list')
    first,target,old_to_new=edge_list_to_forward_star(num_nodes,np.ascontiguousarray(edges).reshape(-1))
    # Native output maps original edge -> CSR slot; upstream gathers weights
    # and therefore needs the inverse, CSR slot -> original edge.
    return first,target,np.argsort(old_to_new)


def load_partitioner(**kwargs):
    space=upstream.namespace('src.utils.instance')
    tree=ast.parse(ast.unparse(space.nodes['_instance_cut_pursuit']))
    changes=0
    for node in ast.walk(tree):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='squeeze' and isinstance(node.func.value,ast.Name) and node.func.value.id=='edge_affinity_logits':
            assert not node.args and not node.keywords
            node.func.attr='reshape';node.args=[ast.Constant(-1)];changes+=1
    assert changes==1,'upstream drift: expected exactly one singleton squeeze'
    exec(compile(ast.fix_missing_locations(tree),'<explicit-singleton-shape-compat>','exec'),space)
    space['edge_list_to_forward_star']=forward_star
    model=upstream.resolve('src.nn.instance','InstancePartitioner')(**kwargs)
    def partition(batch,x,logits,stuff,size,edges,affinity):
        if len(stuff):raise ValueError('stuff-class global merging not qualified')
        n=x.shape[0]
        if edges.ndim!=2 or edges.shape[0]!=2 or affinity.shape!=(edges.shape[1],):
            raise ValueError('invalid edge or affinity shape')
        if logits.shape[0]!=n or size.shape!=(n,):raise ValueError('node shape mismatch')
        if edges.numel() and (edges.min()<0 or edges.max()>=n):raise ValueError('invalid endpoint')
        if batch is not None:
            if batch.shape!=(n,):raise ValueError('batch shape mismatch')
            if edges.numel() and (batch[edges[0]]!=batch[edges[1]]).any():
                raise ValueError('cross-batch edge')
        if x.shape[0]==0:
            if logits.shape[0] or size.numel() or edges.numel() or affinity.numel():
                raise ValueError('nonempty fields for empty graph')
            return torch.empty(0,dtype=torch.long,device=x.device)
        if edges.numel()==0:
            if affinity.numel():raise ValueError('affinity without edges')
            if len(stuff):raise ValueError('empty-edge stuff merging not qualified')
            return torch.arange(x.shape[0],device=x.device)
        pairs=edges.detach().cpu().numpy()
        graph=coo_matrix((np.ones(edges.shape[1]),(pairs[0],pairs[1])),shape=(n,n))
        count,labels=connected_components(graph,directed=False)
        result=torch.empty(n,dtype=torch.long,device=x.device)
        offset=0
        for component in range(count):
            ids=torch.as_tensor(np.flatnonzero(labels==component),device=x.device)
            mask=torch.as_tensor(labels[pairs[0]]==component,device=edges.device)
            if ids.numel()==1:
                result[ids]=offset;offset+=1;continue
            inverse=torch.full((n,),-1,dtype=torch.long,device=edges.device)
            inverse[ids]=torch.arange(ids.numel(),device=edges.device)
            local=model(None,x[ids],logits[ids],[],size[ids],inverse[edges[:,mask]],affinity[mask])
            _,local=torch.unique(local,sorted=True,return_inverse=True)
            result[ids]=local+offset;offset+=int(local.max())+1
        return result
    return partition
