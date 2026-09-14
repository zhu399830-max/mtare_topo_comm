"""Read-only sealed45 actual feature join, no weights or target access."""
import _bootstrap
import hashlib
import json
import resource
import time
from pathlib import Path
from mtare_topo.data.gse_partition_cache_binding import load_bound_partition_observations


def main():
    start=time.monotonic();rows=[]
    for value in load_bound_partition_observations(Path(__file__).resolve().parents[2]):
        row=dict(observation_id=value['observation_id'],encoder_sha256=value['binding'].encoder_sha256,
            points=len(value['source_flat_ray_index']),projection_difference_m=value['maximum_projection_difference_m'],
            methods={m:dict(blocks=len(v['blocks'].block_ids),shape=list(v['context'].shape),
                context_sha256=hashlib.sha256(v['context'].tobytes()).hexdigest()) for m,v in value['methods'].items()})
        rows.append(row)
    print(json.dumps(dict(observations=len(rows),points=sum(r['points'] for r in rows),
        maximum_projection_difference_m=max(r['projection_difference_m'] for r in rows),
        encoder_states=sorted(set(r['encoder_sha256'] for r in rows)),
        groups={m:sum(r['methods'][m]['blocks'] for r in rows) for m in ('r1','r2')},
        elapsed_s=time.monotonic()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        optimizer_steps=0,encoder_inference=0,cases=rows)))


if __name__=='__main__':main()
