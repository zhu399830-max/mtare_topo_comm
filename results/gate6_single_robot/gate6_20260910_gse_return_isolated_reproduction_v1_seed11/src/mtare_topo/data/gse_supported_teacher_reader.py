"""Strict teacher-only C01 ray/geometry provenance reader.

There is deliberately no student input API: absolute poses, construction IDs
and membership codes stay at the teacher boundary. Source paths are filtered
as text before resolution; only selected rows/unique source frames decode.
"""
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import zarr

from mtare_topo.governance_supported_teacher import (
    TEACHER_FIELDS, SENSOR_FIELDS, validate_supported_teacher_card,
)
from mtare_topo.data.gse_scoped_inventory import EvidenceStore, validate_selection
from mtare_topo.data.gse_local_teacher_audit import FIELDS as OLD_FIELDS, ScopedLocalTeacherReader
from mtare_topo.data.cano_sensor_smoke import NEAR_RANGE_M, MAX_RANGE_M


def _contained(project_root, relative):
    path = (project_root / relative).resolve(strict=True)
    if not path.is_relative_to(project_root): raise PermissionError("source escapes project")
    return path


def selected_source_index(project_root, card):
    report = validate_supported_teacher_card(card)
    if not report.passed: raise ValueError(report.errors)
    root = Path(project_root).resolve(strict=True); expected = {}
    for role, seal in card["source_seals"].items():
        prefixes = [card["source_roots"][role] + "/" + task + ".zarr/" for task in card["tasks"]]
        fields = SENSOR_FIELDS if role == "sensor" else TEACHER_FIELDS
        exact_json = set(card["task_json_sha256"]) if role == "sensor" else set()
        seal_path = _contained(root, seal)
        data = seal_path.read_bytes()
        if hashlib.sha256(data).hexdigest() != card["sealed_sources"][seal]: raise ValueError("source seal drift")
        for line in data.decode().splitlines():
            digest, path = line.split(None, 1)
            selected = path in exact_json
            if not selected:
                suffix = next((path[len(p):] for p in prefixes if path.startswith(p)), None)
                if suffix in (".zgroup", ".zattrs"): selected = True
                elif suffix is not None:
                    parts = suffix.split("/")
                    selected = (len(parts)==2 and parts[0] in fields and
                        (parts[1] in (".zarray", ".zattrs") or re.fullmatch(r"[0-9]+(?:\.[0-9]+)*", parts[1]) is not None))
            if not selected: continue
            if re.fullmatch(r"[a-f0-9]{64}", digest) is None: raise ValueError("invalid selected source SHA256")
            # The selected key is lexical-safe by the filters above. Avoid
            # strict existence here: EvidenceStore reports sealed missing
            # chunks instead of Zarr silently synthesizing fill values.
            candidate = root / path
            resolved = candidate.resolve()
            if not resolved.is_relative_to(root): raise PermissionError("selected source symlink escape")
            key = str(candidate)
            if key in expected and expected[key]!=digest: raise ValueError("conflicting source seals")
            expected[key] = digest
    for path, digest in card["task_json_sha256"].items():
        if expected.get(str(root/path)) != digest: raise ValueError("task JSON missing from matching sealed P1a source")
    return expected


class ScopedSupportedTeacherReader:
    def __init__(self, project_root, card, *, expected_sha256):
        report=validate_supported_teacher_card(card)
        if not report.passed: raise ValueError(report.errors)  # before all file I/O
        if not isinstance(expected_sha256,dict): raise ValueError("verified source index required")
        self.selection=validate_selection(card["selected_rows"])
        self.project_root=Path(project_root).resolve(strict=True);self.card=card
        self.roots={role:_contained(self.project_root,path) for role,path in card["source_roots"].items()}
        self.expected_sha256=expected_sha256;self.opened={}
        self.teacher_reader=ScopedLocalTeacherReader(self.roots["teacher"],card["selected_rows"],expected_sha256=expected_sha256)
        self.teacher_reader.opened=self.opened

    def _open(self, role, task, fields):
        path=self.roots[role]/(task+".zarr")
        if path.resolve().parent!=self.roots[role]: raise PermissionError("task symlink escapes exact root")
        return zarr.open_group(store=EvidenceStore(path,fields,self.opened,self.expected_sha256),mode="r")

    def _json(self,role,task):
        path=self.roots[role]/(task+".json")
        if path.resolve().parent!=self.roots[role]:raise PermissionError("task JSON symlink escapes root")
        data=path.read_bytes();digest=hashlib.sha256(data).hexdigest()
        relative=self.card["source_roots"][role]+"/"+task+".json"
        if self.expected_sha256.get(str(path))!=digest or self.card["task_json_sha256"].get(relative)!=digest:
            raise ValueError("unsealed or changed task JSON")
        self.opened[str(path)]=digest
        value=json.loads(data)
        if (not isinstance(value,dict) or value.get("parent_id")!=task.split("__")[0]
                or value.get("geometry_realization")!="c1_mixed"):raise ValueError("task JSON identity drift")
        return value

    def read_task(self, task):
        if task not in self.selection:raise PermissionError("unselected task")
        records=self.selection[task]
        teacher={key:value for key,value in self.teacher_reader.read_task(task).items() if key in OLD_FIELDS}
        for name in ("frame_row","source_global_sequence_index","primitive_index","primitive_mask","support_ray_count","temporal_visibility"):
            if teacher[name].dtype.kind not in "iu":raise ValueError("integer teacher metadata required")
        indices=np.asarray([r["row_index"] for r in records],dtype=np.int64)
        motion_fields=frozenset(("relative_translation_current_sensor_m","relative_yaw_current_sensor_deg"))
        group=self._open("teacher",task,motion_fields)
        for name,shape in (("relative_translation_current_sensor_m",(18,5,3)),("relative_yaw_current_sensor_deg",(18,5))):
            value=np.asarray(group[name].oindex[indices])
            if value.shape!=shape or value.dtype!=np.float32 or not np.isfinite(value).all() or np.any(value[:,-1]!=0):
                raise ValueError("relative odometry contract drift")
            teacher[name]=value
        frame_rows=teacher["frame_row"]
        if frame_rows.dtype.kind not in "iu" or not np.all(frame_rows[:,1:]>frame_rows[:,:-1]):
            raise ValueError("integer strictly causal frame rows required")
        unique=np.unique(frame_rows)
        if len(unique)!=90:raise ValueError("same18 windows must reference90 unique original frames per task")
        sensor_group=self._open("sensor",task,SENSOR_FIELDS)
        attrs=sensor_group.attrs
        if (attrs.get("parent_id")!=task.split("__")[0] or attrs.get("partition")!="fit"
                or attrs.get("geometry_realization")!="c1_mixed" or attrs.get("sensor_shape")!=[16,720]
                or attrs.get("maximum_range_m")!=50. or attrs.get("student_pose_input_forbidden") is not True):
            raise ValueError("sensor identity/shape contract drift")
        frames=sensor_group["range_m"].shape[0]
        if np.any(unique>=frames):raise ValueError("selected source frame outside sensor shard")
        expected_shapes={"range_m":(frames,16,720),"valid_mask":(frames,16,720),
            "primitive_membership_code":(frames,16,720),"sensor_xyz_m":(frames,3),"yaw_deg":(frames,)}
        sensor={}
        for name,shape in expected_shapes.items():
            array=sensor_group[name]
            if array.shape!=shape:raise ValueError(f"sensor shape drift: {name}")
            sensor[name]=np.asarray(array.oindex[unique])
        ranges,valid,codes=sensor["range_m"],sensor["valid_mask"],sensor["primitive_membership_code"]
        if (ranges.dtype!=np.float32 or valid.dtype!=np.uint8 or codes.dtype!=np.uint16
                or not np.isfinite(ranges).all() or np.any(ranges<NEAR_RANGE_M) or np.any(ranges>MAX_RANGE_M)
                or not np.isin(valid,(0,1)).all() or not np.array_equal(codes==0,valid==0)):
            raise ValueError("range/valid/source membership contract drift")
        for name in ("sensor_xyz_m","yaw_deg"):
            if sensor[name].dtype not in (np.float32,np.float64) or not np.isfinite(sensor[name]).all():
                raise ValueError("finite floating teacher-only absolute pose required")
        construction=self._json("construction",task);codebook=self._json("codebook",task)
        primitives=construction.get("realized_primitives")
        if not isinstance(primitives,list) or not primitives:raise ValueError("nonempty realized construction required")
        primitive_ids=[p["primitive_id"] for p in primitives]
        if len(set(primitive_ids))!=len(primitive_ids) or codebook.get("primitive_ids")!=primitive_ids:
            raise ValueError("codebook/construction primitive order drift")
        source_sets=codebook.get("source_sets")
        if (not isinstance(source_sets,list) or not source_sets or source_sets[0]!=[]
                or any(not isinstance(s,list) or not s and i>0 or
                       any(type(v) is not int or not 0<=v<len(primitives) for v in s) or s!=sorted(set(s))
                       for i,s in enumerate(source_sets)) or np.any(codes>=len(source_sets))):
            raise ValueError("invalid source-set codebook or out-of-bounds membership code")
        return {"teacher":teacher,"sensor":sensor,"sensor_frame_rows":unique,"frame_rows":frame_rows,
                "construction":construction,"codebook":codebook,"records":records}
