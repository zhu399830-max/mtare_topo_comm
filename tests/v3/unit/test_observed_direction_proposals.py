import numpy as np
import pytest
from mtare_topo.semantics.observed_direction_proposals import sector_column_mask,range_proposals
from mtare_topo.semantics.range_exit_baseline import RangeExitBaseline
from mtare_topo.semantics.primitive_relation_nonlearning import ELEVATION_DEG

def test_asymmetric_peak_does_not_recenter_support():
    s=dict(start_column=100,end_column=199,angular_width_deg=50.,heading_robot_deg=99.)
    assert np.flatnonzero(sector_column_mask(s)).tolist()==list(range(100,200))

def test_wrapped_sector_preserves_exact_columns():
    s=dict(start_column=700,end_column=19,angular_width_deg=20.)
    assert set(np.flatnonzero(sector_column_mask(s)))==set(range(700,720))|set(range(20))

def test_conflicting_metadata_rejected():
    with pytest.raises(ValueError):sector_column_mask(dict(start_column=0,end_column=9,angular_width_deg=20.))

def test_real_detector_proposal_sources_inside_original_component():
    x=np.ones((16,720),np.float32)*3;x[:,100:200]=15;x[:,190:195]=30
    valid=np.ones_like(x,bool)
    sectors=RangeExitBaseline().predict(x,valid,np.asarray(ELEVATION_DEG))['sectors']
    predictions=range_proposals(x,valid,np.eye(4),'f')
    assert len(predictions)==len(sectors)>0
    for p,s in zip(predictions,sectors):
        allowed=sector_column_mask(s)
        assert all(allowed[int(r.rsplit('/ray:',1)[1])%720] for r in p['source_refs'])
