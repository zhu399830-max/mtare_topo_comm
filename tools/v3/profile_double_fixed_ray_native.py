"""Bounded CPU profile of the pinned diagnostic; no data export or training."""
import cProfile
import json
import pstats
from check_double_fixed_ray_native import main


if __name__=='__main__':
    profiler=cProfile.Profile()
    profiler.runcall(main)
    rows=[]
    for (file,line,name),(primitive,total,self_s,cumulative_s,callers) in pstats.Stats(profiler).stats.items():
        if any(key in file or key in name for key in (
            'oriented_winding','interval_winding_exit','ray_exit_hits',
            'list_intersections','add_triangles','mesh_swept_superellipse')):
            rows.append(dict(file=file,line=line,function=name,calls=total,
                             self_s=self_s,cumulative_s=cumulative_s))
    print(json.dumps({'profile':sorted(rows,key=lambda r:-r['cumulative_s']),
        'scope':'single_pinned_ray_not_population_throughput'},indent=2))
