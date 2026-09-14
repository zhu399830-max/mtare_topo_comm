"""Teacher-only exact source reader. Caller must preflight an approved run.

The source plan alone does not authorize an experiment. Recompiled equality
and sealed-byte checks protect correspondence, not approval or label validity.
No model API, source-ID candidate selection, annotation or mesh rerender here.
"""
from copy import deepcopy
import io
from pathlib import Path

import numpy as np
import zarr

from mtare_topo.data.gse_surface_input_export_v1 import _ExactStore, _json_object
from mtare_topo.data.gse_surface_teacher_scope_v1 import compile_teacher_source_scope, FIELDS, plan_field
from mtare_topo.data.primitive_relation_sequences import causal_relative_odometry
from mtare_topo.governance_identity_inventory import P1A
from mtare_topo.governance_surface_material import read_pinned
from mtare_topo.governance_surface_selection import digest


def verify_alignment(sensor, student, construction, codebook, source):
    """Verify source identity/motion before any downstream teacher geometry."""
    parent, variant = source["task"].split("__")
    for document in (construction,codebook):
        if document.get("parent_id")!=parent or document.get("geometry_realization")!=variant:
            raise ValueError("construction/codebook task identity mismatch")
    primitives = construction.get("realized_primitives")
    if type(primitives) is not list or not primitives:
        raise ValueError("missing realized primitive inventory")
    ids = [p.get("primitive_id") for p in primitives]
    if any(type(i) is not str or not i for i in ids) or len(set(ids))!=len(ids) or codebook.get("primitive_ids")!=ids:
        raise ValueError("source primitive order mismatch")
    sets = codebook.get("source_sets")
    if type(sets) is not list or not sets or sets[0]!=[]:
        raise ValueError("source code zero must mean no valid return")
    for i,s in enumerate(sets):
        if type(s) is not list or (i>0 and not s) or any(type(k) is not int or not 0<=k<len(ids) for k in s) or s!=sorted(set(s)):
            raise ValueError("invalid multi-source provenance; never force unique ownership")
    codes = sensor["primitive_membership_code"]
    if codes.shape!=(5,16,720) or codes.dtype!=np.uint16 or np.any(codes>=len(sets)):
        raise ValueError("source code shape/dtype/index mismatch")
    if not np.array_equal(codes==0,student["valid_mask"]==0):
        raise ValueError("source code/first-return validity differs")
    for name,shape in (("sensor_xyz_m",(5,3)),("yaw_deg",(5,))):
        a = sensor[name]
        if a.shape!=shape or a.dtype!=np.float64 or not np.isfinite(a).all():
            raise ValueError("finite original float64 teacher pose required")
    motion = causal_relative_odometry(sensor["sensor_xyz_m"],sensor["yaw_deg"])
    for name in ("relative_translation_current_sensor_m","relative_yaw_current_sensor_deg"):
        # Original window writer stores float32. No tolerance widening.
        if not np.array_equal(getattr(motion,name.removeprefix("relative_")).astype(np.float32),student[name]):
            raise ValueError("absolute/relative pose binding differs at original storage precision")


class SurfaceTeacherReader:
    def __init__(self, root, source_plan):
        self.root = Path(root).resolve(strict=True)
        if digest(compile_teacher_source_scope(self.root))!=digest(source_plan):
            raise ValueError("teacher scope differs from current sealed selection")
        self.plan = deepcopy(source_plan)
        self.rows = {r["task"]:r for r in self.plan["selected"]}
        self.opened = {}; self.completed = set()

    def read_task(self, task):
        if task not in self.rows or task in self.completed:
            raise ValueError("task outside scope or already consumed")
        row = self.rows[task]; source = row["source"]
        sensor = {}
        for field in FIELDS:
            prefix = P1A+"/artifacts/dataset/fit/"+task+".zarr/"+field
            store = _ExactStore(self.root,prefix,self.plan["file_sha256"],self.opened)
            plan = self.plan["array_access"][prefix]
            store.allowed = {".zarray",*plan["chunk_keys"]}
            header = _json_object(store[".zarray"])
            if plan_field(header,field,source["frame_rows"],source["source_frame_count"])!=plan:
                raise ValueError("planned array access drift")
            array = zarr.Array(store=store,read_only=True)
            sensor[field] = np.asarray(array.oindex[source["frame_rows"]])
        p = row["input_path"]; h = self.plan["existing_sensor_inputs_sha256"][p]
        raw = read_pinned(self.root,p,h);self.opened[p]=h
        with np.load(io.BytesIO(raw),allow_pickle=False) as archive:
            if set(archive.files)!={"ranges_m","valid_mask","relative_translation_current_sensor_m",
                "relative_yaw_current_sensor_deg","frame_rows","source_sequence_ids"}:
                raise ValueError("sealed input field contract differs")
            student = {k:archive[k][row["input_row"]].copy() for k in archive.files}
        if (student["source_sequence_ids"].item()!=source["source_sequence_id"]
            or student["frame_rows"].tolist()!=source["frame_rows"]):
            raise ValueError("sealed input observation mismatch")
        documents = {}
        for role in ("constructions","codebooks"):
            path = P1A+"/artifacts/"+role+"/fit/"+task+".json"
            h = self.plan["file_sha256"][path]
            documents[role] = _json_object(read_pinned(self.root,path,h));self.opened[path]=h
        verify_alignment(sensor,student,documents["constructions"],documents["codebooks"],source)
        self.completed.add(task)
        return {"sensor_teacher_only":sensor,"student":student,"construction_teacher_only":documents["constructions"],
            "codebook_teacher_only":documents["codebooks"],"source":deepcopy(source)}
