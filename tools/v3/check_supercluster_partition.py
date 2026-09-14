"""Known-answer CPU smoke of unchanged official InstancePartitioner."""
import json
import subprocess
import importlib.metadata
import torch
import check_supercluster_affinity as upstream


def main():
    assert subprocess.check_output(['git','-C',str(upstream.ROOT),'rev-parse','HEAD'],text=True).strip()==upstream.COMMIT
    assert not subprocess.check_output(['git','-C',str(upstream.ROOT),'status','--porcelain'],text=True).strip()
    partitioner=upstream.resolve('src.nn.instance','InstancePartitioner')(
        regularization=.01,x_weight=1,p_weight=1,parallel=False)
    # Equal semantics: only spatial fidelity and affinities drive grouping.
    x=torch.tensor([[0.,0.,0.],[.1,0.,0.],[2.,0.,0.]])
    edges=torch.tensor([[0,1],[1,2]])
    def run(prob,positions=x,edge=edges):
        return partitioner(None,positions,torch.zeros(3,2),[],torch.ones(3)*4,
                           edge,torch.logit(torch.tensor(prob)))
    # Preserve the two-edge compatibility defect, do not silently alter upstream.
    compatibility_error=None
    try:
        run([.875,.125])
    except TypeError as exc:
        if 'each edge contiguously' not in str(exc):raise
        compatibility_error=str(exc)
    # Separate diagnostic: explicit zero-weight third edge changes no objective
    # term, but avoids the native extension's ambiguous 2-by-2 array layout.
    edges=torch.tensor([[0,1,0],[1,2,2]])
    def run(prob,positions=x,edge=edges):
        return partitioner(None,positions,torch.zeros(3,2),[],torch.ones(3)*4,
                           edge,torch.logit(torch.tensor(prob)))
    result=run([.875,.125,0.])
    expected=torch.tensor([[True,True,False],[True,True,False],[False,False,True]])
    assert torch.equal(result[:,None]==result[None,:],expected),result
    for _ in range(3):
        assert torch.equal(run([.875,.125,0.]),result)
    # Strong links should merge even the spatially separated third block.
    merged=run([.9999,.9999,0.])
    assert merged.unique().numel()==1,merged
    # Rename nodes, remap edges, then undo permutation before comparison.
    perm=torch.tensor([2,0,1]); inv=torch.argsort(perm)
    renamed=run([.875,.125,0.],x[perm],inv[edges])[inv]
    assert torch.equal(renamed[:,None]==renamed[None,:],expected),renamed
    print(json.dumps(dict(status='PARTITION_DIAGNOSTIC_WITH_TWO_EDGE_COMPATIBILITY_GAP',
        two_edge_error=compatibility_error,zero_weight_third_edge_diagnostic=True,
        commit=upstream.COMMIT,partition=result.tolist(),strong_links=merged.tolist(),
        source_sha256=upstream.READS,executed_definitions=upstream.EXECUTED,
        dependencies={k:importlib.metadata.version(k) for k in
          ['torch','torch-geometric','torch-scatter','pycut-pursuit','pygrid-graph','hydra-core']},
        optimizer_steps=0,research_observations=0,neural_model_forward=False)))


if __name__=='__main__':main()
