"""Label/score-independent C02 selection; no LiDAR or geometric targets read."""
from pathlib import Path
import re
import numpy as np
import zarr
from mtare_topo.data.gse_scoped_inventory import EvidenceStore

FIELDS=frozenset(("frame_row","source_global_sequence_index","primitive_mask"))
TASK=re.compile(r"S(?:0[1-9]|10)_[a-z0-9_]+_C02__c1_mixed$")


def quantile_rows(count):
    if type(count) is not int or count<18:raise ValueError("at least18 source rows required")
    return [((2*k+1)*count)//36 for k in range(18)]


class HeadDevelopmentMetadataReader:
    def __init__(self,root,tasks,expected_sha256):
        if (not isinstance(tasks,list) or len(tasks)!=10 or not all(isinstance(t,str) and TASK.fullmatch(t) for t in tasks)
                or tasks!=sorted(set(tasks)) or {t[:3] for t in tasks}!={f"S{i:02d}" for i in range(1,11)}):
            raise ValueError("exact10 C02 parents required before IO")
        self.tasks=tuple(tasks);self.root=Path(root).resolve(strict=True);self.expected=expected_sha256;self.opened={}

    def read_task(self,task):
        if task not in self.tasks:raise PermissionError("unregistered task")
        path=self.root/(task+".zarr")
        if path.resolve().parent!=self.root:raise PermissionError("shard escapes root")
        group=zarr.open_group(store=EvidenceStore(path,FIELDS,self.opened,self.expected),mode="r")
        attrs=group.attrs.asdict()
        expected={"parent_id":task.split("__")[0],"partition":"fit","geometry_realization":"c1_mixed",
                  "maximum_slots":32,"window_frames":5,"student_identity_input_forbidden":True}
        if any(attrs.get(k)!=v for k,v in expected.items()):raise ValueError("metadata identity/split drift")
        count=int(group["frame_row"].shape[0]);rows=quantile_rows(count)
        shapes={"frame_row":(count,5),"source_global_sequence_index":(count,),"primitive_mask":(count,32)}
        for name,shape in shapes.items():
            if group[name].shape!=shape or group[name].dtype.kind not in "iu":raise ValueError("metadata shape/dtype drift")
        traversals=attrs.get("traversal_ids")
        if not isinstance(traversals,list) or len(traversals)!=count or any(not isinstance(v,str) or not v for v in traversals):raise ValueError("traversal metadata drift")
        # Row selection above depends only on N. Targets and IDs cannot select it.
        data={name:np.asarray(group[name].oindex[rows]) for name in FIELDS}
        frames,source,mask=(data[k] for k in ("frame_row","source_global_sequence_index","primitive_mask"))
        if (np.any(frames<0) or np.any(np.diff(frames.astype(np.int64),axis=1)!=1)
                or np.any(source<0) or np.any(np.diff(source.astype(np.int64))<=0)
                or np.any((mask!=0)&(mask!=1)) or np.any(mask.sum(axis=1)<1)):
            raise ValueError("invalid causal references or visible mask")
        selected=[{"task":task,"row_index":row,"source_global_sequence_index":int(source[i]),
            "frame_rows":frames[i].astype(int).tolist(),"visible_fragments":int(mask[i].sum()),
            "traversal_id_metadata_only":traversals[row]} for i,row in enumerate(rows)]
        summary={"task":task,"available_sequences":count,"selected_observations":18,"unique_source_frames":int(np.unique(frames).size),
            "visible_fragments":int(mask.sum()),"selected_traversals":len({r["traversal_id_metadata_only"] for r in selected}),
            "minimum_selected_row_gap":min(np.diff(rows).tolist()),"selected_frame_references":90,
            "nominal_source_arc_spacing_m":1.,"physical_spacing_measured":False}
        return selected,summary
