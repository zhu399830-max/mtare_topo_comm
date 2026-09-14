"""Same seven recorded native streams: attribution and separate root trial."""
import json
from pathlib import Path
import numpy as np
from mtare_topo.data.gse_synthetic_matrix import matrix,construction_document
from mtare_topo.data.primitive_relation_materialization import load_p1a_realized_construction
from mtare_topo.teacher.csg_mesh_provenance import mesh_swept_superellipse
from mtare_topo.teacher.mesh_origin_check import PreparedOriginCheck
from mtare_topo.teacher.exact_hit_attribution import attribute_hit
from mtare_topo.teacher.csg_interval_raycaster_v2 import verified_interval_hit


def main():
    rows=[json.loads(s) for s in Path('docs/evidence/unknown_hit_planes_20260908.jsonl').read_text().splitlines()];assert len(rows)==7
    config=json.loads(Path('configs/v3/gate3/double_axis_software_contract_v1.json').read_text())
    cases={c['case_id']:c for c in matrix()};key=None;replacements=0
    for row in rows:
        case=cases[row['case']]
        if key!=row['case']:
            _,primitives=load_p1a_realized_construction(construction_document(case))
            meshes=[mesh_swept_superellipse(p,axial_spacing_m=.05,angular_segments=64) for p in primitives]
            origin_check=PreparedOriginCheck(meshes);key=row['case']
        origin=case['poses_world_m'][row['frame']];direction=config['directions'][row['axis']]
        checked=origin_check.check(origin);assert checked.status=='origin_checked'
        attrs=[attribute_hit(meshes[h['operand']],h['triangle'],origin,direction) for h in row['hits']]
        assert all(a is not None for a in attrs)
        replacements+=sum(a.replaced for a in attrs)
        native=[h['native_m'] for h in row['hits']];ids=[h['operand'] for h in row['hits']]
        # Attribution alone leaves the actual interval inputs unchanged.
        old=verified_interval_hit(meshes,origin,direction,checked.inside,native,ids)
        assert old is None
        roots=[float(a.exact_t) for a in attrs]
        collision=any(roots[i]==roots[j] and attrs[i].exact_t!=attrs[j].exact_t
                      for i in range(len(attrs)) for j in range(i))
        trial=verified_interval_hit(meshes,origin,direction,checked.inside,roots,ids)
        print(json.dumps({'case':key,'frame':row['frame'],'axis':row['axis'],
            'attribution_replacements':sum(a.replaced for a in attrs),'attribution_only_reference':None,
            'float_conversion_collapses_distinct_exact_roots':collision,
            'separate_refined_root_trial_m':None if trial is None else trial.distance_m,
            'attributions':[{'original_triangle':a.original_triangle,'valid_triangles':a.valid_triangles,
                             'exact_t':str(a.exact_t),'replaced':a.replaced} for a in attrs]}),flush=True)
    assert replacements==18


if __name__=='__main__':main()
