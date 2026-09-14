from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'tools/v3'))
from run_conditional_development_evaluation import require_complete_reference


def summary():
    return dict(error=None,completed=240,windows=[dict(case=i) for i in range(240)])


def test_complete_population_required():
    require_complete_reference(dict(state='COMPLETED'),summary())


@pytest.mark.parametrize('state',['RUNNING','FAILED','CREATED_NOT_EXECUTED'])
def test_nonterminal_or_failed_cannot_freeze_evaluation(state):
    with pytest.raises(ValueError):require_complete_reference(dict(state=state),summary())


def test_partial_and_duplicate_rejected():
    for defect in ('partial','duplicate','error'):
        s=summary()
        if defect=='partial':s['completed']=239
        elif defect=='duplicate':s['windows'][-1]['case']=0
        else:s['error']='failed worker'
        with pytest.raises(ValueError):require_complete_reference(dict(state='COMPLETED'),s)
