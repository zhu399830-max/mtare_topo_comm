import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'tools/v3'))
from evaluate_observation_axis_core import metrics


def test_undefined_known_gets_worst_error_unknown_not_negative():
    m=metrics(np.array([np.nan,.2,.9]),np.array([False,True,True]),
        np.array([.1,.2,0.]),np.array([True,True,False]),
        np.array([False,False,False]),np.array([.5,.5,0.]))['all']
    assert m['full_population_error']==.5
    assert m['valid_only_mae']==0
    assert m['weighted_coverage']==.5
    assert m['reference_count']==2


def test_all_undefined_never_perfect_score():
    m=metrics(np.array([np.nan]),np.array([False]),np.array([1.]),
        np.array([True]),np.array([True]),np.array([1.]))
    assert m['all']['full_population_error']==1
    assert m['all']['valid_only_mae'] is None
    assert m['cross']['full_population_error'] is None
