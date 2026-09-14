"""Synthetic compatibility checks: no research data, no learning."""
import json
import subprocess
import numpy as np
import torch
import check_supercluster_affinity as upstream
from supercluster_partition_compat import forward_star,load_partitioner


def main():
    assert subprocess.check_output(['git','-C',str(upstream.ROOT),'rev-parse','HEAD'],text=True).strip()==upstream.COMMIT
    assert not subprocess.check_output(['git','-C',str(upstream.ROOT),'status','--porcelain'],text=True).strip()
    # Decode native CSR independently and verify weighted edge identities.
    for pairs in [[],[[1,2]],[[1,2],[0,1]],[[2,0],[0,2],[1,0],[0,1]]]:
        arr=np.asarray(pairs,dtype=np.int64).reshape(-1,2)
        first,target,reindex=forward_star(3,arr)
        decoded=np.column_stack((np.repeat(np.arange(3),np.diff(first).astype(np.int64)),target))
        np.testing.assert_array_equal(decoded,arr[reindex])
        assert sorted(reindex.tolist())==list(range(len(pairs)))
    model=load_partitioner(regularization=.01,x_weight=1,p_weight=1,parallel=False)
    def run(xs,pairs,prob):
        x=torch.tensor(xs,dtype=torch.float32).reshape(-1,3);n=len(x)
        return model(None,x,torch.zeros(n,2),[],torch.ones(n)*4,
                     torch.tensor(pairs,dtype=torch.long).reshape(-1,2).T,
                     torch.logit(torch.tensor(prob,dtype=torch.float32)))
    x=[[0,0,0],[.1,0,0],[2,0,0]]
    assert run([],[],[]).numel()==0
    assert run(x[:1],[],[]).tolist()==[0]
    no_edges=run(x,[],[])
    assert no_edges.unique().numel()==3,no_edges
    single=run(x,[[0,1]],[.875])
    two=run(x,[[0,1],[1,2]],[.875,.125])
    expected=torch.tensor([[1,1,0],[1,1,0],[0,0,1]],dtype=torch.bool)
    for out in [single,two,run(x,[[0,1],[1,2],[0,2]],[.875,.125,0.])]:
        assert torch.equal(out[:,None]==out[None,:],expected),out
    assert run(x,[[0,1],[1,2]],[.9999,.9999]).unique().numel()==1
    for _ in range(3):assert torch.equal(run(x,[[0,1],[1,2]],[.875,.125]),two)
    shuffled=run(x,[[1,2],[0,2],[0,1]],[.125,0.,.875])
    assert torch.equal(shuffled[:,None]==shuffled[None,:],expected),shuffled
    # Two geometrically identical components must remain separate.
    disconnected=run(x[:2]+x[:2],[[0,1],[2,3]],[.875,.875])
    assert disconnected[0]==disconnected[1] and disconnected[2]==disconnected[3]
    assert disconnected[0]!=disconnected[2],disconnected
    perm=[2,0,1];inverse=np.argsort(perm)
    renamed=run([x[i] for i in perm],inverse[np.array([[0,1],[1,2]])].tolist(),[.875,.125])[inverse]
    assert torch.equal(renamed[:,None]==renamed[None,:],expected),renamed
    print(json.dumps(dict(status='EXPLICIT_COMPAT_SYNTHETIC_PASS',
        edge_counts=[0,1,2,3,4],empty_graph=True,no_edge_partition=no_edges.tolist(),
        two_edge_partition=two.tolist(),disconnected_partition=disconnected.tolist(),
        energy_formula_changed=False,component_separation_enforced=True,edges_added=0,
        changes=['flatten_native_edge_memory','invert_native_edge_permutation','reshape_singleton_logits','empty_graph_return','isolated_thing_nodes_retained'],
        source_sha256=upstream.READS,optimizer_steps=0,research_observations=0)))


if __name__=='__main__':main()
