"""Read-only same45 source/target census; no model, new teacher or export."""
import _bootstrap
import json
import time
from pathlib import Path
from mtare_topo.data.gse_partition_cache_binding import load_bound_partition_observations
from mtare_topo.data.gse_synthetic_fit_scope import declared_cases
from mtare_topo.data.gse_block_targets import located_targets
from mtare_topo.evaluation.gse_synthetic_field_scoring import expected_geometry
from mtare_topo.data.gse_structure_review_v1 import canonical_sha


def main():
    start=time.monotonic();cases={v['case_id']:v for v in declared_cases()};rows=[]
    for bound in load_bound_partition_observations(Path(__file__).resolve().parents[2],methods=('r0','r1','r2')):
        ref=expected_geometry(cases[bound['observation_id']]);target=located_targets(ref)
        if not target.anchors_complete or not target.openings_complete:raise ValueError('declared fixture coverage drift')
        row=dict(case_id=bound['observation_id'],points=len(bound['source_flat_ray_index']),
            groups={m:len(v['blocks'].block_ids) for m,v in bound['methods'].items()},
            anchors=len(target.anchors_m),openings=len(target.openings_m),directions=int(target.direction_known.sum()),
            reference_sha256=canonical_sha(ref))
        rows.append(row)
    if len(rows)!=45:raise ValueError('exact45 required')
    print(json.dumps(dict(observations=45,frames=225,roi_returns=sum(r['points'] for r in rows),
        groups={m:sum(r['groups'][m] for r in rows) for m in ('r0','r1','r2')},
        anchors=sum(r['anchors'] for r in rows),openings=sum(r['openings'] for r in rows),
        known_directions=sum(r['directions'] for r in rows),anchor_empty_observations=sum(r['anchors']==0 for r in rows),
        maximum_anchors=max(r['anchors'] for r in rows),maximum_openings=max(r['openings'] for r in rows),
        elapsed_s=time.monotonic()-start,rows_sha256=canonical_sha(rows),optimizer_steps=0,rows=rows)),flush=True)


if __name__=='__main__':main()
