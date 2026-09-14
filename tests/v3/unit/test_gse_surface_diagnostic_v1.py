from types import SimpleNamespace

import numpy as np
import pytest

from mtare_topo.teacher import gse_surface_diagnostic_v1 as module
from mtare_topo.teacher.swept_superellipse_field import SweptSuperellipsePrimitive


def bundle():
    return dict(sensor_teacher_only=dict(sensor_xyz_m=np.tile([.013,0,0],(5,1)),yaw_deg=np.zeros(5)),
        student=dict(ranges_m=np.full((5,16,720),30,dtype=np.float32),valid_mask=np.ones((5,16,720),dtype=np.uint8)),
        source=dict(frame_rows=[0,1,2,3,4]),construction_teacher_only={})


def primitive(name="p"):
    return SweptSuperellipsePrimitive(name,np.array([[-20.,0,0],[20,0,0]]),((2.,2.),(2.,2.)),(2.,2.))


def test_full_projection_mesh_and_ray_pipeline_stays_diagnostic(monkeypatch):
    monkeypatch.setattr(module,"load_p1a_realized_construction",lambda doc:(SimpleNamespace(compositions=[]),(primitive(),)))
    r=module.diagnose_observation(bundle(),range_error_bound_m=0.)
    assert len(r["candidate_sections"])==2 and r["section_positive_count"]==2
    assert r["labels_generated"]==0 and not r["teacher_complete"]


def test_all_source_competition_not_gt_incident_selection(monkeypatch):
    monkeypatch.setattr(module,"load_p1a_realized_construction",lambda doc:(SimpleNamespace(compositions=[]),(primitive(),primitive("q"))))
    r=module.diagnose_observation(bundle(),range_error_bound_m=0.)
    assert len(r["candidate_sections"])==4 and r["section_positive_count"]==0
    assert all(row["section_local_witness_indices"] for row in r["candidate_sections"])


def test_occluded_sections_are_unknown_not_negative(monkeypatch):
    monkeypatch.setattr(module,"load_p1a_realized_construction",lambda doc:(SimpleNamespace(compositions=[]),(primitive(),)))
    b=bundle();b["student"]["ranges_m"][:]=3
    r=module.diagnose_observation(b,range_error_bound_m=0.)
    assert r["section_positive_count"]==0
    assert all(row["semantic_opening_label"] is None for row in r["candidate_sections"])
