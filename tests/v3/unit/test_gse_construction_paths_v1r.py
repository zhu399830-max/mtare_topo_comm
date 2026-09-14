import numpy as np
import pytest

from mtare_topo.teacher.gse_construction_paths_v1r import construction_incident_paths


def document():
    line=[[0,0,0],[2,0,0]]
    return dict(base_construction=dict(coordinate_frame="cano_world",primitives=[dict(primitive_id="p",centerline_xyz_m=line)],
        compositions=[dict(node_id="a",anchor_xyz_m=[0,1,0],degree=1,member_endpoints=[dict(
            primitive_id="p",endpoint_index=0,node_id="a",xyz_m=[0,0,0],composition_anchor_xyz_m=[0,1,0])]),
            dict(node_id="b",anchor_xyz_m=[2,0,0],degree=1,member_endpoints=[dict(
                primitive_id="p",endpoint_index=1,node_id="b",xyz_m=[2,0,0],composition_anchor_xyz_m=[2,0,0])])]))


def test_offset_is_preserved_not_snapped():
    result=construction_incident_paths(document())
    a=result[0]["paths"][0]
    np.testing.assert_array_equal(a["points_world_m"],[[0,1,0],[0,0,0],[2,0,0]])
    assert a["anchor_axis_offset_m"]==1 and a["connector_is_observation_probe_only"]


def test_reverse_incidence_not_collapsed_by_primitive_identity():
    result=construction_incident_paths(document())
    np.testing.assert_array_equal(result[1]["paths"][0]["points_world_m"],[[2,0,0],[0,0,0]])
    assert [r["paths"][0]["endpoint_key"] for r in result]==[("p",0),("p",1)]


@pytest.mark.parametrize("field,value",[("xyz_m",[0,0,3]),("composition_anchor_xyz_m",[0,1,3]),("node_id","b")])
def test_no_xy_or_nearest_repair(field,value):
    d=document();d["base_construction"]["compositions"][0]["member_endpoints"][0][field]=value
    with pytest.raises(ValueError):construction_incident_paths(d)


def test_output_does_not_alias_source():
    d=document();p=construction_incident_paths(d)[0]["paths"][0]["points_world_m"]
    d["base_construction"]["primitives"][0]["centerline_xyz_m"][0][0]=8
    assert p[1,0]==0
    with pytest.raises(ValueError):p[0,0]=5


def test_original_frame_name_required_not_generic_world():
    d=document();d["base_construction"]["coordinate_frame"]="world"
    with pytest.raises(ValueError,match="cano_world"):construction_incident_paths(d)
