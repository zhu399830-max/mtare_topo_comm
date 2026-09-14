from pathlib import Path
import copy
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'tools/v3'))
from conditional_development_statistics import summarize


def population():
    identities=[dict(case=i,parent_id=p,task=p+'__ellipse',physical_edge_id=p+':'+str(i)) for i,p in enumerate(['p','p','q'])]
    rows=[]
    for v in 'ABC':
        for i in identities:
            val=(1. if i['parent_id']=='p' else 3.)+(0. if v=='C' else 1.)
            m=dict(count=2,axis_mae=val,height_mae_m=val,constant_axis_mae=1.,constant_height_mae_m=2.)
            rows.append(dict(case=i['case'],variant=v,parent=i['parent_id'],task=i['task'],physical_edge=i['physical_edge_id'],
                unknown_relations=3,output_relations=5,metrics={c:dict(m) for c in ('all','same','cross')}))
    return rows,identities


def test_parent_pairing_not_patch_population():
    rows,identities=population();s=summarize(rows,identities)
    assert s['parents']==2 and s['observations']==3
    p=s['paired_cross_component']['C_minus_B_height_mae_m']
    assert p['mean']==-1 and p['improved_parents']==2
    assert p['descriptive_bootstrap_95']==[-1,-1]
    assert s==summarize(list(reversed(rows)),identities)


@pytest.mark.parametrize('defect',['missing','duplicate','identity','reference','nonfinite'])
def test_invalid_comparison_rejected(defect):
    rows,ids=population()
    if defect=='missing':rows.pop()
    elif defect=='duplicate':rows.append(copy.deepcopy(rows[0]))
    elif defect=='identity':rows[0]['parent']='other'
    elif defect=='reference':rows[0]['metrics']['cross']['constant_axis_mae']=9.
    else:rows[0]['metrics']['cross']['height_mae_m']=float('nan')
    with pytest.raises(ValueError):summarize(rows,ids)


def test_unknown_parent_not_perfect_prediction():
    rows,ids=population()
    for r in rows:
        if r['parent']=='q':r['metrics']['cross']=dict(count=0,axis_mae=None,height_mae_m=None,constant_axis_mae=None,constant_height_mae_m=None)
    s=summarize(rows,ids);p=s['paired_cross_component']['C_minus_B_axis_mae']
    assert p['parents']==1 and p['descriptive_bootstrap_95'] is None
    q=[r for r in s['per_parent'] if r['parent']=='q'][0]
    assert q['metrics']['cross']['observations_scored']==0
    assert q['metrics']['cross']['axis_mae'] is None
