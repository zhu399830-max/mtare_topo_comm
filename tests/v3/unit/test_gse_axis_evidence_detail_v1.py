from dataclasses import replace

import numpy as np
import pytest

from mtare_topo.representation.gse_surface_ray_evidence_v1 import build_surface_ray_grid
from mtare_topo.teacher.gse_observed_axis_support_v1 import observed_axis_support
from mtare_topo.teacher.gse_axis_evidence_detail_v1 import axis_evidence_detail


def test_missing_cells_are_explicit_not_obstacle_labels():
    g=build_surface_ray_grid(np.array([[.1,.1,.1]]),np.array([[.4,.1,.1]]),np.array([True]),np.array([2]))
    s=observed_axis_support(g,np.array([[.11,.1,.1],[1.4,.1,.1]]))
    d=axis_evidence_detail(g,s)
    assert d['status_unchanged']==s.status=='UNKNOWN'
    assert d['semantic_label'] is None
    assert sum(c['state']==0 for c in d['cells'])==s.unknown_cells>0
    assert all(c['free_frame_bits']==0 and c['occupied_frame_bits']==0 for c in d['cells'] if c['state']==0)


def test_all_free_boundary_stays_distinct_from_missing_observation():
    origins=np.array([[-.4,-.1,.1],[-.4,.1,.1]])
    ends=np.array([[.4,-.1,.1],[.4,.1,.1]])
    g=build_surface_ray_grid(origins,ends,np.array([True,True]),np.array([0,1]))
    s=observed_axis_support(g,np.array([[-.1,0.,.1],[.1,0.,.1]]))
    d=axis_evidence_detail(g,s)
    assert s.status=='UNKNOWN' and s.unknown_cells==s.occupied_cells==0
    assert all(c['state']==1 and c['query_boundary_ambiguous'] for c in d['cells'])


def test_wrong_grid_or_summary_rejected():
    g=build_surface_ray_grid(np.array([[.1,.1,.1]]),np.array([[1.4,.1,.1]]),np.array([True]),np.array([0]))
    s=observed_axis_support(g,np.array([[.11,.1,.1],[1.,.1,.1]]))
    for bad in (replace(s,grid_content_sha256='wrong'),replace(s,free_cells=s.free_cells+1)):
        with pytest.raises(ValueError):axis_evidence_detail(g,bad)
