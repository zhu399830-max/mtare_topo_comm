import numpy as np
from mtare_topo.evaluation.gse_affinity_metrics import affinity_metrics


def test_constant_positive_does_not_hide_negative_failure():
    y=np.array([1]*99+[0]+[np.nan]);k=np.isfinite(y);v=np.ones(101,dtype=bool)
    m=affinity_metrics(np.ones(101),y,k,v)
    assert m['positive_recall']==1 and m['negative_recall']==0 and m['balanced_accuracy']==.5
    assert m['unknown_predicted_positive']==1


def test_absent_class_is_not_perfect_score():
    m=affinity_metrics(np.ones(2),np.ones(2),np.ones(2,dtype=bool),np.ones(2,dtype=bool))
    assert m['negative_recall'] is None and m['balanced_accuracy'] is None
