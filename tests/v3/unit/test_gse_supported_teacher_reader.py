import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import zarr

from mtare_topo.data.gse_supported_teacher_reader import ScopedSupportedTeacherReader,selected_source_index
from mtare_topo.governance_supported_teacher import TEACHER_FIELDS,SENSOR_FIELDS
from tests.v3.unit.test_gse_supported_teacher_card import card


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(tmp_path):
    value=card()
    for root in value["source_roots"].values():(tmp_path/root).mkdir(parents=True,exist_ok=True)
    task=value["tasks"][0];common={"parent_id":task.split("__")[0],"partition":"fit","geometry_realization":"c1_mixed"}
    t=zarr.open_group(str(tmp_path/value["source_roots"]["teacher"]/(task+".zarr")),mode="w")
    t.attrs.update(**common,student_identity_input_forbidden=True,window_frames=5)
    mask=(np.arange(32)[None]<np.full((18,1),8)).astype("u1")
    indices=np.tile(np.arange(32),(18,1));indices[mask==0]=-1
    arrays={"frame_row":np.arange(90,dtype="i8").reshape(18,5),"source_global_sequence_index":np.arange(18,dtype="i8"),
        "primitive_index":indices,"primitive_mask":mask,"axis_control_current_sensor_m":np.zeros((18,32,3,3),"f4"),
        "endpoint_half_axes_m":np.ones((18,32,2,2),"f4"),"endpoint_shape_exponent":np.ones((18,32,2),"f4"),
        "support_ray_count":mask.astype("i4")*5,"temporal_visibility":np.repeat(mask[:,None],5,axis=1),
        "endpoint_neighbor":np.full((18,32,2,3),-1,"i1"),"disconnected_overlap_packed":np.zeros((18,32,4),"u1"),
        "relative_translation_current_sensor_m":np.zeros((18,5,3),"f4"),"relative_yaw_current_sensor_deg":np.zeros((18,5),"f4")}
    for key,data in arrays.items():t.create_dataset(key,data=data)
    s=zarr.open_group(str(tmp_path/value["source_roots"]["sensor"]/(task+".zarr")),mode="w")
    s.attrs.update(**common,sensor_shape=[16,720],maximum_range_m=50.,student_pose_input_forbidden=True)
    for key,data in {"range_m":np.full((95,16,720),5,"f4"),"valid_mask":np.ones((95,16,720),"u1"),
        "primitive_membership_code":np.ones((95,16,720),"u2"),"sensor_xyz_m":np.zeros((95,3),"f8"),"yaw_deg":np.zeros(95,"f8")}.items():
        s.create_dataset(key,data=data,chunks=(5,*data.shape[1:]))
    for chosen in value["tasks"]:
        identity={"parent_id":chosen.split("__")[0],"geometry_realization":"c1_mixed"}
        documents={"construction":{**identity,"realized_primitives":[{"primitive_id":str(i)} for i in range(8)],"base_construction":{"composition_operations":[]}},
                   "codebook":{**identity,"primitive_ids":[str(i) for i in range(8)],"source_sets":[[],[0]]}}
        for role,document in documents.items():
            path=tmp_path/value["source_roots"][role]/(chosen+".json");path.write_text(json.dumps(document))
            value["task_json_sha256"][str(path.relative_to(tmp_path))]=sha(path)
    reseal(tmp_path,value)
    return value,t,s


def reseal(tmp_path,value):
    for role,seal in value["source_seals"].items():
        roots=[value["source_roots"][role]]
        if role=="sensor":roots += [value["source_roots"][r] for r in ("construction","codebook")]
        files=[p for root in roots for p in (tmp_path/root).rglob("*") if p.is_file()]
        path=tmp_path/seal
        path.write_text("".join(f"{sha(p)}  {p.relative_to(tmp_path)}\n" for p in sorted(files))
            +"0"*64+"  nonexistent/S01_bad_C07__c1_mixed.zarr/range_m/0.0.0\n"
            +"0"*64+"  /absolute/C10/checkpoint.pt\n")
        value["sealed_sources"][seal]=sha(path)


def reader(tmp_path,value):
    return ScopedSupportedTeacherReader(tmp_path,value,expected_sha256=selected_source_index(tmp_path,value))


def test_only_exact_selected_rows_fields_and_json_are_read(tmp_path,monkeypatch):
    value,t,s=fixture(tmp_path)
    original=Path.resolve
    def resolving(path,*args,**kwargs):
        assert "C07" not in str(path) and "C10" not in str(path)
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,"resolve",resolving)
    source=reader(tmp_path,value);out=source.read_task(value["tasks"][0])
    assert set(out["teacher"])==set(TEACHER_FIELDS) and set(out["sensor"])==set(SENSOR_FIELDS)
    assert out["sensor_frame_rows"].tolist()==list(range(90))
    assert out["sensor"]["range_m"].shape==(90,16,720)
    assert out["frame_rows"].shape==(18,5)
    assert out["construction"]["parent_id"]==value["worlds"][0]
    assert not hasattr(source,"student")
    accessed=list(source.opened)
    assert not any("range_m/18." in path for path in accessed)  # source frame90+ not decoded
    assert all(value["tasks"][0] in path for path in accessed)


def test_c02_record_rejected_before_any_io():
    value=card();value["selected_rows"][0]["task"]=value["selected_rows"][0]["task"].replace("C01","C02")
    with pytest.raises(ValueError):ScopedSupportedTeacherReader("/nonexistent",value,expected_sha256={})


def test_unselected_task_rejected_without_reads(tmp_path):
    value,_,_=fixture(tmp_path);source=reader(tmp_path,value)
    with pytest.raises(PermissionError):source.read_task(value["tasks"][0].replace("C01","C02"))
    assert source.opened=={}


@pytest.mark.parametrize("field",["primitive_membership_code","sensor_xyz_m","yaw_deg"])
def test_new_sensor_field_hash_drift_stops(tmp_path,field):
    value,_,s=fixture(tmp_path);source=reader(tmp_path,value)
    s[field][0]=s[field][0]+1
    with pytest.raises(ValueError,match="drift|unsealed"):source.read_task(value["tasks"][0])


def test_missing_sealed_chunk_not_filled_silently(tmp_path):
    value,_,_=fixture(tmp_path);source=reader(tmp_path,value)
    path=tmp_path/value["source_roots"]["sensor"]/(value["tasks"][0]+".zarr")/"primitive_membership_code/0.0.0"
    path.unlink()  # Synthetic tmp fixture only.
    with pytest.raises(ValueError,match="sealed chunk missing"):source.read_task(value["tasks"][0])


@pytest.mark.parametrize("issue",["valid_nonbinary","code_invalid","code_oob","pose_nan","relative_last","source_id","float_frame_rows","unsigned_reverse","near_range","far_range"])
def test_invalid_resealed_values_still_fail_contract(tmp_path,issue):
    value,t,s=fixture(tmp_path)
    if issue=="valid_nonbinary":s["valid_mask"][0]=2
    elif issue=="code_invalid":s["primitive_membership_code"][0]=0
    elif issue=="code_oob":s["primitive_membership_code"][0]=2
    elif issue=="pose_nan":s["sensor_xyz_m"][0]=float("nan")
    elif issue=="relative_last":t["relative_yaw_current_sensor_deg"][0,-1]=1
    elif issue=="source_id":t["source_global_sequence_index"][0]=123
    elif issue=="near_range":s["range_m"][0]=.2
    elif issue=="far_range":s["range_m"][0]=50.00001
    elif issue=="unsigned_reverse":
        data=t["frame_row"][:].astype("u8");data[0]=data[0,::-1];del t["frame_row"];t.create_dataset("frame_row",data=data)
    else:
        data=t["frame_row"][:].astype("f4");del t["frame_row"];t.create_dataset("frame_row",data=data)
    reseal(tmp_path,value)
    with pytest.raises(ValueError):reader(tmp_path,value).read_task(value["tasks"][0])


@pytest.mark.parametrize("issue",["primitive_order","duplicate_source","bad_source_type","unsorted_source","wrong_parent"])
def test_codebook_semantics_strict_even_when_resealed(tmp_path,issue):
    value,_,_=fixture(tmp_path);relative=value["source_roots"]["codebook"]+"/"+value["tasks"][0]+".json"
    path=tmp_path/relative;book=json.loads(path.read_text())
    if issue=="primitive_order":book["primitive_ids"].reverse()
    elif issue=="duplicate_source":book["source_sets"][1]=[0,0]
    elif issue=="bad_source_type":book["source_sets"][1]=[False]
    elif issue=="unsorted_source":book["source_sets"][1]=[1,0]
    else:book["parent_id"]=value["worlds"][1]
    path.write_text(json.dumps(book));value["task_json_sha256"][relative]=sha(path);reseal(tmp_path,value)
    with pytest.raises(ValueError):reader(tmp_path,value).read_task(value["tasks"][0])


def test_reader_does_not_authorize_writes_or_extra_fields(tmp_path):
    value,_,_=fixture(tmp_path);source=reader(tmp_path,value)
    group=source._open("sensor",value["tasks"][0],SENSOR_FIELDS)
    with pytest.raises(PermissionError):group.store["range_m/0.0.0"]=b"bad"
    with pytest.raises(PermissionError):group.store["axis_xyz_m/.zarray"]
