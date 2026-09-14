import numpy as np

from test_gse_construction_paths_v2 import document
from mtare_topo.teacher.gse_construction_paths_v2 import construction_incident_paths as old_paths
from mtare_topo.teacher.gse_construction_paths_v3 import construction_incident_paths


def test_source_axis_index_preserves_paths_and_offset_exactly():
    d=document(); before=old_paths(d); after=construction_incident_paths(d)
    assert [g["paths"][0]["axis_start_index"] for g in after]==[1,0]
    for a,b in zip(before,after):
        pa,pb=a["paths"][0],b["paths"][0]
        np.testing.assert_array_equal(pa["points_world_m"],pb["points_world_m"])
        assert pa["anchor_axis_offset_m"]==pb["anchor_axis_offset_m"]
        assert pa["endpoint_key"]==pb["endpoint_key"]


def test_tiny_connector_not_reinterpreted_by_tolerance():
    d=document();g=d["base_construction"]["composition_operations"][0]
    g["anchor_xyz_m"]=[0.,1e-14,0.]
    g["member_endpoints"][0]["composition_anchor_xyz_m"]=g["anchor_xyz_m"][:]
    p=construction_incident_paths(d)[0]["paths"][0]
    assert p["axis_start_index"]==1
    np.testing.assert_array_equal(p["points_world_m"][1:3],[[0,0,0],[2,0,0]])
