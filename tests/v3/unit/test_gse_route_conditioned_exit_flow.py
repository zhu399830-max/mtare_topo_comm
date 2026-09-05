from __future__ import annotations
import importlib.util
from pathlib import Path
import numpy as np
SOURCE=Path(__file__).resolve().parents[3]/"tools/v3/execute_gse_route_conditioned_exit_flow_audit_v1.py";SPEC=importlib.util.spec_from_file_location("route_flow",SOURCE);MODULE=importlib.util.module_from_spec(SPEC);assert SPEC.loader is not None;SPEC.loader.exec_module(MODULE)
def test_backward_exit_has_positive_stop_score():
 raw=np.zeros((1,3,6,40),dtype=np.float32);raw[...,0]=1;raw[...,2]=-1;value=MODULE.route_flow(raw);np.testing.assert_allclose(value["route_stop_score"],1)
def test_forward_exit_has_negative_stop_score():
 raw=np.zeros((1,3,6,40),dtype=np.float32);raw[...,0]=1;raw[...,2]=1;value=MODULE.route_flow(raw);np.testing.assert_allclose(value["route_stop_score"],-1)
def test_token_permutation_does_not_change_flow():
 rng=np.random.default_rng(2);raw=rng.normal(size=(4,3,6,40));raw[...,0]=rng.uniform(0,1,size=(4,3,6));first=MODULE.route_flow(raw);second=MODULE.route_flow(raw[:,:,[3,0,5,1,4,2]]);np.testing.assert_allclose(first["route_stop_score"],second["route_stop_score"])
def test_auc_handles_ties():
 assert MODULE.auc(np.asarray([0,0,1,1]),np.asarray([0,1,1,2]))==.875
