import json
from copy import deepcopy
import pytest
from mtare_topo.evaluation.native_candidate_graph import validate_native_graph


def graph():
    return dict(scope='fiveframe_candidate_graph_not_verified',rays=1,below_min_ray_count=0,
        observed_esdf=2,allocated_unknown=2,hallucinated=0,missing_tsdf_support=0,nodes=2,edges=1,
        vertices=[dict(id=0,xyz=[0,0,0]),dict(id=1,xyz=[1,0,0])],connections=[[0,1]],
        edge_audit=[dict(edge_id=0,endpoints=[0,1],samples=9,unknown_samples=0,
                        below_0_4m_samples=0,minimum_sampled_esdf_m=.5)],audit_spacing_m=.125,
        audit_is_continuous_safety_proof=False)


def test_parallel_native_edges_preserved():
    g=graph();g['edges']=2;g['connections'].append([0,1]);g['edge_audit'].append(deepcopy(g['edge_audit'][0]))
    g['edge_audit'][1]['edge_id']=1
    assert len(validate_native_graph(json.dumps(g))['connections'])==2


@pytest.mark.parametrize('field,value',[('nodes',3),('hallucinated',1),('rays',57601)])
def test_bad_counts(field,value):
    g=graph();g[field]=value
    with pytest.raises(ValueError):validate_native_graph(json.dumps(g))


def test_unknown_and_clearance_consistency():
    g=graph();a=g['edge_audit'][0];a['unknown_samples']=9;a['minimum_sampled_esdf_m']=None
    validate_native_graph(json.dumps(g))
    a['minimum_sampled_esdf_m']=.5
    with pytest.raises(ValueError):validate_native_graph(json.dumps(g))
    g=graph();g['edge_audit'][0]['minimum_sampled_esdf_m']=.3
    with pytest.raises(ValueError):validate_native_graph(json.dumps(g))
