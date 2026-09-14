"""Temporary synthetic Zarr only; never use the actual selected corpus."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import zarr

from mtare_topo.data.gse_surface_input_export_v1 import (
    SurfaceInputReader, plan_array_access, SENSOR_FIELDS, WINDOW_FIELDS,
)


def fixture(tmp_path, cohort="C01", split="fit", variant="ellipse"):
    parent = "S01_flat_tree_small_" + cohort; task = parent + "__" + variant
    partition = "c07" if cohort == "C07" else "fit"
    sources = {task: {"sensor": f"sensor/{partition}/{task}.zarr", "teacher": f"teacher/{partition}/{task}.zarr",
                      "parent_id": parent, "variant": variant, "partition": partition}}
    sensor = zarr.open_group(str(tmp_path / sources[task]["sensor"]), mode="w")
    teacher = zarr.open_group(str(tmp_path / sources[task]["teacher"]), mode="w")
    common = dict(parent_id=parent, partition=partition, geometry_realization=variant)
    sensor.attrs.update(common, schema_version="primitive_relation_p1a_sensor_shard_v1", sensor_shape=[16,720],
                        maximum_range_m=50., student_pose_input_forbidden=True)
    teacher.attrs.update(common, schema_version="primitive_relation_p1b_teacher_shard_v1", window_frames=5,
                         maximum_slots=32, student_identity_input_forbidden=True)
    ranges = np.broadcast_to(np.arange(113, dtype=np.float32)[:, None, None] / 10, (113,16,720)).copy()
    valid = np.ones(ranges.shape, dtype=np.uint8); valid[::3, 0, 0] = 0
    sensor.create_dataset("range_m", data=ranges, chunks=(16,16,720), compressor=None)
    sensor.create_dataset("valid_mask", data=valid, chunks=(32,16,720), compressor=None)
    # Forbidden source arrays exist, so a broad reader is observable in tests.
    sensor.create_dataset("sensor_xyz_m", data=np.zeros((113,3)), chunks=(113,3), compressor=None)
    frame_rows = np.zeros((300,5), dtype=np.int32)
    sequence = np.arange(1000,1300, dtype=np.int64)
    translation = np.zeros((300,5,3), dtype=np.float32); translation[:, :, 0] = [-4,-3,-2,-1,0]
    yaw = np.zeros((300,5), dtype=np.float32); yaw[:, :4] = [4,3,2,1]
    selection = []
    for i in range(16):
        row = 18 * i; frames = list(range(i * 7, i * 7 + 5)); frame_rows[row] = frames
        selection.append(dict(task=task, parent_id=parent, variant=variant, split=split,
            sequence_row=row, source_sequence_id=1000 + row, source_frame_count=113,
            source_sequence_count=300, frame_rows=frames))
    for name, data in (("frame_row",frame_rows), ("source_global_sequence_index",sequence),
                       ("relative_translation_current_sensor_m",translation), ("relative_yaw_current_sensor_deg",yaw)):
        teacher.create_dataset(name, data=data, chunks=(256,*data.shape[1:]), compressor=None)
    teacher.create_dataset("primitive_mask", data=np.ones((300,32),dtype=np.uint8), chunks=(256,32), compressor=None)
    selected = selection[::-1]  # Preserve original selected order, not sorted source order.
    sealed = {}
    for role, names in (("sensor",SENSOR_FIELDS),("teacher",WINDOW_FIELDS)):
        prefix = sources[task][role]; path = tmp_path / prefix
        keys = {".zgroup", ".zattrs"}
        indices = [f for r in selected for f in r["frame_rows"]] if role == "sensor" else [r["sequence_row"] for r in selected]
        for name in names:
            header_key = name + "/.zarray"; keys.add(header_key)
            header = json.loads((path / header_key).read_text())
            keys.update(name + "/" + k for k in plan_array_access(header,name,role,indices)["chunk_keys"])
        for key in keys:
            sealed[prefix + "/" + key] = hashlib.sha256((path / key).read_bytes()).hexdigest()
    return task, sources, selected, sealed, ranges, valid


def reader(tmp_path, values):
    _, sources, selection, sealed, *_ = values
    return SurfaceInputReader(tmp_path, task_sources=sources, selection=selection, sealed_keys=sealed)


def test_real_zarr_exact_payload_order_chunk_once_and_no_teacher_reads(tmp_path, monkeypatch):
    data = fixture(tmp_path); task, sources, selection, sealed, ranges, valid = data
    reads = []; original = Path.read_bytes
    def traced(path):
        reads.append(str(path.relative_to(tmp_path)))
        return original(path)
    monkeypatch.setattr(Path, "read_bytes", traced)
    r = reader(tmp_path, data); result = r.read_task(task)
    frames = np.array([s["frame_rows"] for s in selection])
    np.testing.assert_array_equal(result.ranges_m, ranges[frames])
    np.testing.assert_array_equal(result.valid_mask, valid[frames])
    np.testing.assert_array_equal(result.frame_rows, frames)
    np.testing.assert_array_equal(result.source_sequence_ids, [s["source_sequence_id"] for s in selection])
    assert result.ranges_m.dtype == np.float32 and result.valid_mask.dtype == np.uint8
    assert result.relative_translation_current_sensor_m.shape == (16,5,3)
    assert result.relative_yaw_current_sensor_deg.shape == (16,5)
    assert set(reads) == set(sealed) == set(r.opened) and len(reads) == len(set(reads))
    assert all("primitive_mask" not in p and "sensor_xyz_m" not in p for p in reads)
    assert not result.ranges_m.flags.writeable and not result.frame_rows.flags.writeable
    report = result.read_report
    assert report["selected_observations"] == 16 and report["selected_unique_sensor_frames"] == 80
    assert report["arrays"]["sensor"]["range_m"]["decoded_actual_rows"] == 112
    assert report["arrays"]["sensor"]["valid_mask"]["decoded_actual_rows"] == 113
    assert report["arrays"]["teacher"]["frame_row"]["decoded_actual_rows"] == 300
    assert report["role_row_coverage"]["sensor"] == {
        "decoded_actual_unique_rows":113, "selected_unique_rows":80, "collateral_actual_unique_rows":33}
    assert report["role_row_coverage"]["teacher"] == {
        "decoded_actual_unique_rows":300, "selected_unique_rows":16, "collateral_actual_unique_rows":284}
    assert report["neighbor_rows_are_training_samples"] is False
    assert report["new_labels"] == report["model_windows"] == 0
    with pytest.raises(ValueError, match="already"):
        r.read_task(task)


@pytest.mark.parametrize("split", ["calibration", "development"])
def test_c07_partition_and_three_variants_are_not_restricted_to_old_mixed_fit(tmp_path, split):
    data = fixture(tmp_path, cohort="C07", split=split, variant="rounded_rectangle")
    result = reader(tmp_path, data).read_task(data[0])
    assert result.ranges_m.shape == (16,5,16,720)


def header(field="range_m", role="sensor", n=33):
    definitions = {"range_m":([16,720],"<f4",16),"valid_mask":([16,720],"|u1",32),
                   "frame_row":([5],"<i4",256),"source_global_sequence_index":([],"<i8",256),
                   "relative_translation_current_sensor_m":([5,3],"<f4",256),
                   "relative_yaw_current_sensor_deg":([5],"<f4",256)}
    tail, dtype, c = definitions[field]
    return dict(zarr_format=2, shape=[n,*tail], chunks=[c if role=="sensor" else min(c,n),*tail],
                dtype=dtype, order="C", compressor=None, filters=None, fill_value=0)


def test_access_plan_exact_tail_chunk_bound_and_unique_rows():
    p = plan_array_access(header(),"range_m","sensor",[0,0,15,32])
    assert p["chunk_keys"] == ["0.0.0", "2.0.0"]
    assert p["decoded_row_intervals"] == [[0,16],[32,33]]
    assert p["selected_unique_rows"] == 3 and p["decoded_actual_rows"] == 17
    assert p["decoded_actual_bytes"] == 17*16*720*4
    assert p["decoded_padded_bytes"] == 32*16*720*4
    motion = plan_array_access(header("relative_translation_current_sensor_m","teacher",300),
                               "relative_translation_current_sensor_m","teacher",[0,299])
    assert motion["chunk_keys"] == ["0.0.0","1.0.0"]
    assert motion["decoded_actual_rows"] == 300


@pytest.mark.parametrize("key,value", [
    ("dtype","<f8"),("chunks",[16,8,720]),("chunks",[32,16,720]),("shape",[33,8,720]),
    ("order","F"),("dimension_separator","/"),("zarr_format",True),("shape",[True,16,720]),
])
def test_bad_headers_rejected_before_decode(key,value):
    h = header(); h[key] = value
    with pytest.raises(ValueError): plan_array_access(h,"range_m","sensor",[0])


@pytest.mark.parametrize("rows", [[],[-1],[33],[True],[1.]])
def test_invalid_selected_row_types_and_bounds(rows):
    with pytest.raises(ValueError): plan_array_access(header(),"range_m","sensor",rows)


def test_no_geometry_field_read_permission():
    with pytest.raises(ValueError): plan_array_access(header(),"axis_control_current_sensor_m","teacher",[0])


def rehash(tmp_path,sealed,path):
    sealed[path] = hashlib.sha256((tmp_path / path).read_bytes()).hexdigest()


@pytest.mark.parametrize("case", ["schema","shape","dtype","chunk","unsealed_extra","unsealed_missing"])
def test_header_or_exact_scope_drift_has_zero_payload_reads(tmp_path, monkeypatch,case):
    values = fixture(tmp_path); task, sources, selection, sealed, *_ = values
    prefix = sources[task]["teacher"]
    if case in ("unsealed_extra","unsealed_missing"):
        if case == "unsealed_extra":
            path = prefix + "/primitive_mask/.zarray"; rehash(tmp_path,sealed,path)
        else:
            del sealed[prefix + "/frame_row/0.0"]
    else:
        path = prefix + ("/.zattrs" if case=="schema" else "/frame_row/.zarray")
        content = json.loads((tmp_path/path).read_text())
        if case == "schema": content["schema_version"] = "wrong"
        elif case == "shape": content["shape"][0] += 1
        elif case == "dtype": content["dtype"] = "<i8"
        else: content["chunks"][0] = 128
        (tmp_path/path).write_text(json.dumps(content)); rehash(tmp_path,sealed,path)
    reads=[]; original=Path.read_bytes
    def traced(path):
        reads.append(path.name)
        return original(path)
    monkeypatch.setattr(Path,"read_bytes",traced)
    with pytest.raises(ValueError): reader(tmp_path,values).read_task(task)
    assert reads and all(name.startswith(".") for name in reads)


@pytest.mark.parametrize("case", ["missing","byte_drift","symlink"])
def test_payload_missing_drift_symlink_do_not_silently_fill(tmp_path,case):
    values=fixture(tmp_path);task,sources,_,_,*_=values
    path=tmp_path/sources[task]["sensor"]/"range_m/0.0.0"
    if case=="missing": path.unlink()
    elif case=="byte_drift": path.write_bytes(b"not the sealed bytes")
    else:
        target=tmp_path/"replacement";path.rename(target);path.symlink_to(target)
    with pytest.raises((ValueError,PermissionError)):reader(tmp_path,values).read_task(task)


@pytest.mark.parametrize("field,value", [
    ("frame_row",99),("source_global_sequence_index",999999),
    ("relative_translation_current_sensor_m",float("nan")),("relative_yaw_current_sensor_deg",float("inf")),
    ("relative_translation_current_sensor_m",1.),("relative_yaw_current_sensor_deg",1.),
    ("valid_mask",2),("range_m",float("nan")),("range_m",50.1),
])
def test_selected_identity_motion_and_sensor_values_validated_after_lossless_read(tmp_path,field,value):
    values=fixture(tmp_path);task,sources,_,sealed,*_=values
    role="sensor" if field in SENSOR_FIELDS else "teacher"
    prefix=sources[task][role];group=zarr.open_group(str(tmp_path/prefix),mode="a")
    a=group[field]
    if field in ("range_m","valid_mask"):a[0,0,0]=value
    elif field=="frame_row":a[0,0]=value
    elif field=="source_global_sequence_index":a[0]=value
    elif field=="relative_translation_current_sensor_m":a[0,-1,0]=value
    else:a[0,-1]=value
    for path in tuple(sealed):
        if path.startswith(prefix+"/"+field+"/"):rehash(tmp_path,sealed,path)
    with pytest.raises(ValueError):reader(tmp_path,values).read_task(task)


@pytest.mark.parametrize("case", ["C08","duplicate","history","population","partition"])
def test_scope_rejected_without_source_payload(tmp_path,case):
    values=fixture(tmp_path);task,sources,selection,sealed,*_=values
    if case=="C08":
        sources[task.replace("C01","C08")]=sources.pop(task)
    elif case=="duplicate":selection[1]=selection[0].copy()
    elif case=="history":selection[0]["frame_rows"][-1]+=1
    elif case=="population":selection[0]["source_sequence_count"]+=1
    else:sources[task]["partition"]="development"
    with pytest.raises(ValueError):reader(tmp_path,values)


def test_chunk_neighbor_rows_never_enter_effective_output(tmp_path):
    values=fixture(tmp_path);task,sources,_,sealed,*_=values
    prefix=sources[task]["sensor"]
    source=zarr.open_group(str(tmp_path/prefix),mode="a")
    source["range_m"][5]=np.full((16,720),49.,dtype=np.float32)  # not selected; shares selected physical chunk0
    rehash(tmp_path,sealed,prefix+"/range_m/0.0.0")
    output=reader(tmp_path,values).read_task(task)
    assert not np.any(output.ranges_m==49.)
    assert output.read_report["selected_observations"]==16


def test_existing_header_not_writer_assumption_controls_actual_shape_validation(tmp_path):
    values=fixture(tmp_path);task,sources,_,sealed,*_=values
    path=sources[task]["sensor"]+"/valid_mask/.zarray"
    original=json.loads((tmp_path/path).read_text());original["chunks"]=[32,8,720]
    (tmp_path/path).write_text(json.dumps(original));rehash(tmp_path,sealed,path)
    with pytest.raises(ValueError,match="tail chunk"):
        reader(tmp_path,values).read_task(task)
