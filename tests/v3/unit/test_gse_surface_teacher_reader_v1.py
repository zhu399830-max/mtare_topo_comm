import numpy as np
import pytest
from mtare_topo.data.gse_surface_teacher_reader_v1 import verify_alignment
from mtare_topo.data.primitive_relation_sequences import causal_relative_odometry


def inputs():
    xyz = np.array([[i*.3,i*.1,2] for i in range(5)],dtype=np.float64)
    yaw = np.array([179,180,181,182,183],dtype=np.float64)
    motion = causal_relative_odometry(xyz,yaw)
    sensor = dict(sensor_xyz_m=xyz,yaw_deg=yaw,primitive_membership_code=np.ones((5,16,720),dtype=np.uint16))
    student = dict(valid_mask=np.ones((5,16,720),dtype=np.uint8),
        relative_translation_current_sensor_m=motion.translation_current_sensor_m.astype(np.float32),
        relative_yaw_current_sensor_deg=motion.yaw_current_sensor_deg.astype(np.float32))
    construction = dict(parent_id="S01_flat_tree_small_C01",geometry_realization="ellipse",realized_primitives=[dict(primitive_id="p"),dict(primitive_id="q")])
    codebook = dict(parent_id=construction["parent_id"],geometry_realization="ellipse",primitive_ids=["p","q"],source_sets=[[],[0,1]])
    source = dict(task="S01_flat_tree_small_C01__ellipse")
    return sensor,student,construction,codebook,source


def test_original_float32_motion_and_multiple_owners_preserved():
    values = inputs(); verify_alignment(*values)
    assert values[3]["source_sets"][1]==[0,1]


def test_one_ulp_motion_mismatch_rejected():
    values = inputs();a = values[1]["relative_translation_current_sensor_m"]
    a[0,0] = np.nextafter(a[0,0],np.float32(np.inf))
    with pytest.raises(ValueError,match="storage precision"): verify_alignment(*values)


def test_invalid_return_cannot_have_source_owner():
    values = inputs(); values[1]["valid_mask"][0,0,0]=0
    with pytest.raises(ValueError,match="validity"): verify_alignment(*values)


@pytest.mark.parametrize("change",["parent","variant","order","source_code"])
def test_teacher_identity_and_code_mismatch(change):
    values = inputs()
    if change=="parent":values[2]["parent_id"]="other"
    if change=="variant":values[3]["geometry_realization"]="c1_mixed"
    if change=="order":values[3]["primitive_ids"]=["q","p"]
    if change=="source_code":values[0]["primitive_membership_code"][0,0,0]=2
    with pytest.raises(ValueError):verify_alignment(*values)


@pytest.mark.parametrize("damage",[None,"missing_chunk","changed_chunk","wrong_input_row"])
def test_scoped_reader_real_codec_and_pins(tmp_path,monkeypatch,damage):
    import hashlib,json,zarr
    from mtare_topo.data import gse_surface_teacher_reader_v1 as module
    from mtare_topo.data.gse_surface_teacher_scope_v1 import plan_field
    sensor,student,construction,codebook,source = inputs()
    task = source["task"];source.update(frame_rows=[0,1,2,3,4],source_frame_count=5,source_sequence_id=42)
    files,plans = {},{}
    def pin(path):files[str(path.relative_to(tmp_path))]=hashlib.sha256(path.read_bytes()).hexdigest()
    for field,data in sensor.items():
        path = tmp_path/module.P1A/"artifacts/dataset/fit"/(task+".zarr")/field
        chunk = (16,*data.shape[1:]) if field=="primitive_membership_code" else data.shape
        zarr.array(data,store=str(path),chunks=chunk,overwrite=False)
        prefix = str(path.relative_to(tmp_path))
        h = json.loads((path/".zarray").read_text())
        plan = plan_field(h,field,source["frame_rows"],5);plans[prefix]=plan
        for name in (".zarray",*plan["chunk_keys"]):pin(path/name)
    for role,doc in (("constructions",construction),("codebooks",codebook)):
        path=tmp_path/module.P1A/"artifacts"/role/"fit"/(task+".json")
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(doc));pin(path)
    path=tmp_path/"input.npz"
    np.savez(path,ranges_m=np.ones((1,5,16,720),dtype=np.float32),
        frame_rows=np.array([[0,1,2,3,4]],dtype=np.int32),source_sequence_ids=np.array([43 if damage=="wrong_input_row" else 42]),
        **{k:v[None] for k,v in student.items()})
    scope=dict(selected=[dict(task=task,source=source,input_path="input.npz",input_row=0)],
        file_sha256=files,array_access=plans,existing_sensor_inputs_sha256={"input.npz":hashlib.sha256(path.read_bytes()).hexdigest()})
    monkeypatch.setattr(module,"compile_teacher_source_scope",lambda root:scope)
    reader=module.SurfaceTeacherReader(tmp_path,scope)
    chunk_path=next(tmp_path/p for p in files if p.endswith("sensor_xyz_m/0.0"))
    if damage=="missing_chunk":chunk_path.unlink()
    if damage=="changed_chunk":chunk_path.write_bytes(b"drift")
    if damage:
        with pytest.raises(ValueError):reader.read_task(task)
        assert not reader.completed
    else:
        result=reader.read_task(task)
        np.testing.assert_array_equal(result["sensor_teacher_only"]["sensor_xyz_m"],sensor["sensor_xyz_m"])
        assert set(reader.opened)==set(files)|{"input.npz"}
        with pytest.raises(ValueError):reader.read_task(task)
    with pytest.raises(ValueError):reader.read_task("S01_flat_tree_small_C10__ellipse")
