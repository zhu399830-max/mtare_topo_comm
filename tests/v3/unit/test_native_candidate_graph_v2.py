import json
from copy import deepcopy
import pytest
from mtare_topo.evaluation.native_candidate_graph import validate_native_graph as v1
from mtare_topo.evaluation.native_candidate_graph_v2 import validate_native_graph as v2
from test_native_candidate_graph import graph


def test_self_loop_preserved_not_rewritten():
    g=graph();g['connections'][0]=[0,0];g['edge_audit'][0]['endpoints']=[0,0]
    assert v2(json.dumps(g))==g
    with pytest.raises(ValueError):v1(json.dumps(g))


@pytest.mark.parametrize('defect',['missing_endpoint','mismatch','nonfinite','duplicate_audit','unsupported'])
def test_other_contracts_not_relaxed(defect):
    g=graph()
    if defect=='missing_endpoint':g['connections'][0]=[0,99]
    elif defect=='mismatch':g['edge_audit'][0]['endpoints']=[1,0]
    elif defect=='nonfinite':g['vertices'][0]['xyz'][0]=float('nan')
    elif defect=='unsupported':g['missing_tsdf_support']=1
    else:
        g['edges']=2;g['connections']*=2;g['edge_audit'].append(deepcopy(g['edge_audit'][0]))
    with pytest.raises(ValueError):v2(json.dumps(g))


def test_valid_original_contract_unchanged():
    raw=json.dumps(graph())
    assert v1(raw)==v2(raw)
