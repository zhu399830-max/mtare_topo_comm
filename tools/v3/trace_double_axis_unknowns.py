"""Trace existing reference rejection for the exact seven recorded unknowns."""
import json
import linecache
from pathlib import Path
import sys
import numpy as np
import open3d as o3d
from mtare_topo.data.gse_synthetic_matrix import matrix,construction_document
from mtare_topo.data.primitive_relation_materialization import load_p1a_realized_construction
from mtare_topo.teacher.csg_mesh_provenance import mesh_swept_superellipse
from mtare_topo.teacher.bound_ray_diagnostic import BoundRayDiagnostic
from mtare_topo.teacher.csg_interval_raycaster_v2 import verified_interval_hit


def main():
    records=[json.loads(s) for s in Path('docs/evidence/double_axis_reference_20260908.jsonl').read_text().splitlines()]
    unknown=[r for r in records if r.get('status')=='unknown'];assert len(unknown)==7
    contract=json.loads(Path('configs/v3/gate3/double_axis_software_contract_v1.json').read_text())
    cases={c['case_id']:c for c in matrix()};bound=None;key=None
    for row in unknown:
        case=cases[row['case']]
        if key!=row['case']:
            _,primitives=load_p1a_realized_construction(construction_document(case))
            bound=BoundRayDiagnostic([mesh_swept_superellipse(p,axial_spacing_m=.05,angular_segments=64) for p in primitives]);key=row['case']
        origin=case['poses_world_m'][row['frame']];direction=contract['directions'][row['axis']]
        checked=bound._origin.check(origin);assert checked.status=='origin_checked'
        raw={k:v.numpy() for k,v in bound._caster.scene.list_intersections(o3d.core.Tensor(np.array([origin+direction],dtype=np.float32))).items()}
        ids=[bound._caster.geometry_to_operand[int(g)] for g in raw['geometry_ids']]
        trace=[]
        def hook(frame,event,arg):
            if frame.f_code.co_name=='interval_winding_exit' and event=='return':
                trace.append({'line':frame.f_lineno,'code':linecache.getline(frame.f_code.co_filename,frame.f_lineno).strip(),
                              'operand':getattr(frame.f_locals.get('mesh'),'primitive_id',None)})
            return hook
        sys.settrace(hook)
        try:result=verified_interval_hit(bound._meshes,origin,direction,checked.inside,raw['t_hit'],ids)
        finally:sys.settrace(None)
        assert result is None and len(trace)==1
        print(json.dumps({'case':key,'frame':row['frame'],'axis':row['axis'],'rejection':trace[0],
                          'native_distances':raw['t_hit'].tolist(),'native_operands':ids}),flush=True)


if __name__=='__main__':main()
