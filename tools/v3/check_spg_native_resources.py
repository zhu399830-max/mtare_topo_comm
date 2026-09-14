"""Fixed synthetic CPU resource checks, no researcher data or training."""
import _bootstrap
import json
import resource
import time
import numpy as np
import libply_c
import libcp
from mtare_topo.representation.gse_spg_extractor import extract_spg


def main():
    rows=[]
    for side in (64,240):
        x,y=np.meshgrid(np.arange(side)*.04,np.arange(side)*.04)
        points=np.column_stack((x.ravel(),y.ravel(),np.zeros(side*side))).astype('float32')
        frames=np.arange(len(points))%5
        start=time.monotonic()
        result=extract_spg(points,frames,geof_backend=libply_c.compute_geof,partition_backend=libcp.cutpursuit)
        peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        if peak>32*1024**3:
            raise MemoryError('32GiB host cap exceeded')
        assert len(result['original_point_to_component'])==len(points)
        assert result['voxels'].frame_counts.sum()==len(points)
        row=dict(points=len(points),voxels=len(result['voxels'].centers_m),status=result['status'],
                 connected_components=len(result['connected'].components),elapsed_s=time.monotonic()-start,
                 process_peak_rss_bytes=peak)
        rows.append(row);print(json.dumps(row),flush=True)
    print(json.dumps(dict(software_checks_complete=True,cases=rows,optimizer_steps=0,
                         real_observations=0,scientific_gate_pass=False)),flush=True)


if __name__=='__main__':
    main()
